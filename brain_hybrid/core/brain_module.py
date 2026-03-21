import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from ncps.torch import CfC
from ncps.wirings import NCP


class BrainModule(nn.Module):
    """
    Module hybride CfC + SNN greffé sur une couche Qwen.

    Reçoit  : représentation d'une couche Qwen (1, seq_len, 2560)
    Produit : prédiction de la couche suivante + spikes
    Apprend : via STDP sur les traces pre/post synaptiques

    CfC  → dynamiques continues, état persiste entre appels
    SNN  → spikes discrets via surrogate gradient (snntorch)
    STDP → mis à jour externement via STDPLearning
    """

    def __init__(
        self,
        input_dim: int = 2560,
        hidden_dim: int = 64,
        device: torch.device = None
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # Projection : 2560 → hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim).to(self.device)

        # CfC — dynamiques liquides continues
        # Câblage NCP inspiré du connectome C. elegans
        wiring = NCP(
            inter_neurons=16,
            command_neurons=8,
            motor_neurons=hidden_dim // 2,
            sensory_fanout=4,
            inter_fanout=4,
            recurrent_command_synapses=6,
            motor_fanin=4,
        )
        self.cfc = CfC(hidden_dim, wiring, batch_first=True).to(self.device)

        # SNN — Leaky Integrate-and-Fire via surrogate gradient
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.lif = snn.Leaky(
            beta=0.95,
            spike_grad=spike_grad,
        ).to(self.device)

        # Tête de prédiction : project vers l'espace Qwen
        self.prediction_head = nn.Linear(
            hidden_dim // 2, input_dim
        ).to(self.device)

        # États persistants — survivent entre les appels
        self.cfc_state = None
        self.mem = None

        # Traces synaptiques pour STDP
        self.pre_trace = None
        self.post_trace = None

    def forward(self, x: torch.Tensor):
        """
        x : (batch, seq_len, 2560)
        Retourne : (prediction, spikes)
          - prediction : (batch, seq_len, 2560)
          - spikes     : (batch, seq_len, hidden_dim//2)
        """
        x = x.to(self.device).float()

        # Projection vers l'espace du module
        h = self.input_proj(x)          # (batch, seq_len, hidden_dim)

        # CfC — état liquide persiste entre appels
        cfc_out, self.cfc_state = self.cfc(h, self.cfc_state)
        if self.cfc_state is not None:
            self.cfc_state = self.cfc_state.detach()

        # SNN — traiter chaque pas de temps séquentiellement
        batch, seq_len, feat = cfc_out.shape

        if self.mem is None:
            self.mem = self.lif.init_leaky()

        spk_list = []
        for t in range(seq_len):
            spk_t, self.mem = self.lif(cfc_out[:, t, :], self.mem)
            spk_list.append(spk_t)

        spk = torch.stack(spk_list, dim=1)  # (batch, seq_len, hidden_dim//2)
        self.mem = self.mem.detach()

        # Prédiction de la couche suivante
        prediction = self.prediction_head(spk)

        # Mise à jour des traces synaptiques pour STDP
        self._update_traces(h.detach(), spk.detach())

        return prediction, spk

    def compute_prediction_error(
        self,
        prediction: torch.Tensor,
        actual: torch.Tensor
    ) -> torch.Tensor:
        """Erreur de prédiction = actual - prediction."""
        return actual.to(self.device).float() - prediction

    def _update_traces(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        decay: float = 0.95
    ):
        """
        Traces exponentielles pour STDP.
        Encodent l'historique récent d'activité pre/post synaptique.
        """
        if (self.pre_trace is None
                or self.pre_trace.shape != pre.shape
                or self.post_trace.shape != post.shape):
            self.pre_trace = torch.zeros_like(pre)
            self.post_trace = torch.zeros_like(post)

        self.pre_trace = decay * self.pre_trace + pre.abs()
        self.post_trace = decay * self.post_trace + post.abs()

    def reset_state(self):
        """Reset complet — à appeler entre sessions distinctes si besoin."""
        self.cfc_state = None
        self.mem = None
        self.pre_trace = None
        self.post_trace = None
