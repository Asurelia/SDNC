"""
Cortex Cingulaire Antérieur artificiel (ACC).

Détecte les conflits entre signaux internes (erreurs de prédiction,
dopamine, entropie) et déclenche des réponses adaptatives :
- Conflit détecté → arousal boost + prompt enrichi
- Exploration → légère augmentation dopamine
- Stable → mode normal

Inspiré de Botvinick et al. (2001) — Conflict Monitoring Theory.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from collections import deque


@dataclass
class ACCOutput:
    """Sortie du module ACC."""
    conflict_score: float
    is_conflict: bool
    is_exploration: bool
    conflict_prompt_fragment: str


class ACCModule(nn.Module):
    """
    Anterior Cingulate Cortex — moniteur de conflit.

    MLP 3 couches avec apprentissage Hebbian local.
    Pas de backprop, pas d'optimizer.
    """

    def __init__(
        self,
        n_modules: int = 4,
        conflict_threshold: float = 0.7,
        exploration_threshold: float = 0.4,
        lr: float = 0.001,
        device: torch.device = None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.conflict_threshold = conflict_threshold
        self.exploration_threshold = exploration_threshold
        self.lr = lr

        # Input = n_modules erreurs + n_modules dopamines + 1 entropie
        input_dim = n_modules * 2 + 1

        self.fc1 = nn.Linear(input_dim, 32).to(self.device)
        self.fc2 = nn.Linear(32, 32).to(self.device)
        self.fc3 = nn.Linear(32, 1).to(self.device)

        # Historique pour tendance
        self.history = deque(maxlen=200)

        # Running mean pour l'apprentissage Hebbian
        self._input_ema = None

    def forward(
        self,
        prediction_errors: list,
        dopamine_signals: list,
        action_entropy: float = 0.0,
    ) -> ACCOutput:
        """
        Calcule le score de conflit.

        prediction_errors : liste de floats (une par module CfC)
        dopamine_signals  : liste de floats (un par stdp_learner)
        action_entropy    : float (0.0 si pas disponible)
        """
        # Construire le vecteur d'entrée
        input_vals = list(prediction_errors) + list(dopamine_signals) + [action_entropy]
        x = torch.tensor(input_vals, dtype=torch.float32, device=self.device)

        # Forward MLP
        with torch.no_grad():
            h = torch.relu(self.fc1(x))
            h = torch.relu(self.fc2(h))
            conflict_score = torch.sigmoid(self.fc3(h)).item()

        # Apprentissage Hebbian local
        self._hebbian_update(x, conflict_score)

        # Déterminer l'état
        is_conflict = conflict_score > self.conflict_threshold
        is_exploration = conflict_score > self.exploration_threshold and not is_conflict

        # Fragment de prompt
        if is_conflict:
            mean_err = sum(prediction_errors) / len(prediction_errors) if prediction_errors else 0
            fragment = f"[CONFLIT DÉTECTÉ: erreur={mean_err:.3f}, arousal={conflict_score:.2f}] "
        else:
            fragment = ""

        # Historique
        self.history.append(conflict_score)

        return ACCOutput(
            conflict_score=conflict_score,
            is_conflict=is_conflict,
            is_exploration=is_exploration,
            conflict_prompt_fragment=fragment,
        )

    def _hebbian_update(self, x: torch.Tensor, conflict_score: float):
        """
        Règle Hebbian simplifiée : Δw = lr * conflict_score * (input - mean_input)
        Pas de backprop. Mise à jour directe sous torch.no_grad().
        """
        with torch.no_grad():
            # Mise à jour de la moyenne mobile
            if self._input_ema is None:
                self._input_ema = x.clone()
            else:
                self._input_ema = 0.99 * self._input_ema + 0.01 * x

            deviation = x - self._input_ema

            # Mise à jour des poids fc1 uniquement (première couche)
            delta = self.lr * conflict_score * deviation
            # Clamp pour la stabilité
            delta = torch.clamp(delta, -1e-4, 1e-4)
            self.fc1.weight.data += delta.unsqueeze(0).expand_as(self.fc1.weight)

    def trend(self) -> str:
        """Tendance du conflict_score sur les 200 derniers appels."""
        if len(self.history) < 20:
            return "stable"
        recent = list(self.history)[-10:]
        older = list(self.history)[-20:-10]
        recent_mean = sum(recent) / len(recent)
        older_mean = sum(older) / len(older)
        if recent_mean > older_mean * 1.1:
            return "rising"
        if recent_mean < older_mean * 0.9:
            return "falling"
        return "stable"

    def save_state(self) -> dict:
        """Sauvegarde l'état de l'ACC."""
        return {
            "state_dict": self.state_dict(),
            "history": list(self.history),
            "input_ema": self._input_ema.cpu() if self._input_ema is not None else None,
        }

    def load_state(self, state: dict):
        """Restaure l'état de l'ACC."""
        self.load_state_dict(state["state_dict"])
        self.history = deque(state["history"], maxlen=200)
        if state["input_ema"] is not None:
            self._input_ema = state["input_ema"].to(self.device)
