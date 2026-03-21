# BRAIN-HYBRID — Architecture cerveau artificiel
## Spécification complète pour Claude Code — Version finale (mars 2026)

> **Lis ce document en entier avant d'écrire une seule ligne de code.**
> L'ordre de lecture est critique. Chaque section dépend de la précédente.
> Ce document est autonome — aucune référence externe requise.

---

## 1. Contexte et vision

### Ce qu'on construit

Un système d'IA hybride qui s'approche du fonctionnement du cerveau humain en combinant :

- **Qwen3-4B** (HuggingFace, bfloat16) — noyau de connaissance stable et gelé, "cortex préfrontal"
- **4 modules CfC+SNN** greffés sur les couches internes du LLM — "cortex adaptatif"
- **STDP + neuromodulation dopamine-like** — apprentissage local timing-dépendant
- **Sparse Distributed Memory** — mémoire épisodique persistante, "hippocampe"

### Analogie cerveau → architecture

| Cerveau humain | Architecture BRAIN-HYBRID |
|---|---|
| Cortex préfrontal | Qwen3-4B gelé (connaissance du monde) |
| Cortex sensoriel adaptatif | 4 modules CfC+SNN |
| Plasticité synaptique | STDP local |
| Dopamine (saillance, importance) | Neuromodulateur global |
| Hippocampe | Sparse Distributed Memory |

### Principes fondamentaux — Non négociables

1. **Le Qwen ne s'entraîne jamais** — tous ses poids sont gelés définitivement
2. **L'apprentissage est local** — uniquement dans les modules CfC+SNN via STDP
3. **Pas de backpropagation globale** sur l'ensemble du système
4. **L'hippocampe ne se remet jamais à zéro** — il s'enrichit en continu
5. **Les erreurs de prédiction** entre couches sont le seul signal d'apprentissage

### Pourquoi pas llama-cpp-python

llama-cpp-python ne permet pas l'accès aux hidden states des couches intermédiaires.
`get_layer_representations()` serait impossible à implémenter.
**HuggingFace Transformers avec `output_hidden_states=True` est la seule solution.**

### Specs réelles Qwen3-4B (vérifiées mars 2026)

- 36 transformer layers
- hidden_size = 2560
- `hidden_states[0]` = embeddings d'entrée
- `hidden_states[1..36]` = représentations après chaque layer
- Points d'intercept choisis : couches **8, 16, 24, 36**
- VRAM en bfloat16 : ~8 GB sur 16 GB disponibles (marge confortable ~6.5 GB)

---

## 2. Composants techniques — Code complet

### 2.1 QwenWrapper — Noyau LLM

```python
# llm/qwen_wrapper.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class QwenWrapper:
    """
    Wrapper Qwen3-4B avec accès aux représentations internes.

    Gelé définitivement — aucun poids ne sera jamais modifié.
    Fournit :
      - get_layer_representations() → hidden states aux couches voulues
      - generate() → génération de texte standard
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-4B"):
        print(f"Chargement {model_name} en bfloat16...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,      # ROCm optimal
            device_map="auto",               # auto-placement GPU
            output_hidden_states=True,
            trust_remote_code=True
        )
        self.model.eval()

        # Geler TOUS les poids — non négociable
        for p in self.model.parameters():
            p.requires_grad = False

        self.device = next(self.model.parameters()).device
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Modèle chargé sur {self.device} | VRAM totale : {vram:.1f} GB")

    def get_layer_representations(
        self,
        prompt: str,
        layers: list = None
    ) -> list:
        """
        Extrait les représentations internes aux couches demandées.

        layers : liste d'indices (0=embeddings, 1-36=layers)
        Retourne : liste de tenseurs (1, seq_len, 2560)
        """
        if layers is None:
            layers = [8, 16, 24, 36]

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # hidden_states : tuple de 37 tenseurs (1, seq_len, 2560)
        hidden = outputs.hidden_states
        return [hidden[i] for i in layers]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        **kwargs
    ) -> str:
        """Génération de texte standard."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        # Retourner uniquement les nouveaux tokens
        new_tokens = out[0][inputs['input_ids'].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
```

---

### 2.2 BrainModule — Module CfC + SNN hybride

```python
# core/brain_module.py

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
        # Nécessaire — CfC ne gère pas les très grandes dimensions
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
            beta=0.95,               # constante de décroissance membrane
            spike_grad=spike_grad,
            init_hidden=True
        ).to(self.device)

        # Tête de prédiction : project vers l'espace Qwen
        self.prediction_head = nn.Linear(
            hidden_dim // 2, input_dim
        ).to(self.device)

        # États persistants — survivent entre les appels
        self.cfc_state = None        # état interne CfC
        self.mem = None              # potentiel de membrane LIF

        # Traces synaptiques pour STDP
        self.pre_trace = None        # activité pré-synaptique
        self.post_trace = None       # activité post-synaptique (spikes)

    def forward(self, x: torch.Tensor):
        """
        x : (batch, seq_len, 2560)
        Retourne : (prediction, spikes)
          - prediction : (batch, seq_len, 2560) — prédit la couche suivante
          - spikes     : (batch, seq_len, hidden_dim//2)
        """
        x = x.to(self.device)

        # Projection vers l'espace du module
        h = self.input_proj(x)          # (batch, seq_len, hidden_dim)

        # CfC — état liquide persiste entre appels
        cfc_out, self.cfc_state = self.cfc(h, self.cfc_state)
        # Détacher l'état pour éviter les graphes de gradient infinis
        if self.cfc_state is not None:
            self.cfc_state = self.cfc_state.detach()

        # SNN — initialiser la membrane au premier appel
        if self.mem is None:
            self.mem = self.lif.init_leaky()
            if hasattr(self.mem, 'to'):
                self.mem = self.mem.to(self.device)

        spk, self.mem = self.lif(cfc_out, self.mem)
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
        return actual.to(self.device) - prediction

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
        if self.pre_trace is None:
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
```

---

### 2.3 STDPLearning — Apprentissage local + Dopamine

```python
# core/stdp.py

import torch


class STDPLearning:
    """
    Spike-Timing-Dependent Plasticity + Neuromodulation dopamine-like.

    Règle biologique STDP :
      - Pré s'active AVANT post → renforcement (LTP)
      - Post s'active AVANT pré  → affaiblissement (LTD)

    Neuromodulation dopamine-like :
      - Signal entre 0 et 1
      - Haute erreur de prédiction → surprise → plus de dopamine
      - Plus de dopamine → amplitude d'apprentissage plus forte
      - Inspiré de la théorie TD (Temporal Difference)
    """

    def __init__(
        self,
        lr_plus: float = 0.01,    # amplitude LTP
        lr_minus: float = 0.01    # amplitude LTD
    ):
        self.lr_plus = lr_plus
        self.lr_minus = lr_minus
        self.dopamine_signal = 0.5    # entre 0.0 et 1.0

    def update_dopamine(
        self,
        prediction_error: float,
        threshold: float = 0.3
    ):
        """
        Met à jour le signal dopamine depuis l'erreur de prédiction.
        Haute erreur = surprise = plus de dopamine = apprentissage renforcé.
        """
        normalized = min(1.0, abs(prediction_error) / threshold)
        # Mise à jour lente — moyenne exponentielle
        self.dopamine_signal = 0.9 * self.dopamine_signal + 0.1 * normalized
        # Clamp entre 0 et 1
        self.dopamine_signal = max(0.0, min(1.0, self.dopamine_signal))

    def apply(
        self,
        weight: torch.Tensor,
        pre_trace: torch.Tensor,
        post_trace: torch.Tensor
    ) -> torch.Tensor:
        """
        Applique STDP sur un tenseur de poids.

        Version stable avec outer product propre et interpolation
        pour gérer les dimensions variables des poids CfC.
        Modulé par le signal dopamine.
        """
        with torch.no_grad():
            # Réduire les traces à des vecteurs 1D
            pre_mean = pre_trace.detach().float().mean(0).mean(0)    # (dim,)
            post_mean = post_trace.detach().float().mean(0).mean(0)  # (dim,)

            out_dim = weight.shape[0]
            in_dim = weight.shape[1]

            # Interpoler vers les dimensions du poids cible
            pre_r = torch.nn.functional.interpolate(
                pre_mean.unsqueeze(0).unsqueeze(0),
                size=in_dim,
                mode='linear',
                align_corners=False
            ).squeeze()

            post_r = torch.nn.functional.interpolate(
                post_mean.unsqueeze(0).unsqueeze(0),
                size=out_dim,
                mode='linear',
                align_corners=False
            ).squeeze()

            # LTP — pré avant post = renforcement
            delta = self.lr_plus * self.dopamine_signal * (
                post_r.unsqueeze(1) * pre_r.unsqueeze(0)
            )

            # LTD — post avant pré = affaiblissement
            delta -= self.lr_minus * self.dopamine_signal * (
                pre_r.unsqueeze(1) * post_r.unsqueeze(0)
            )

            # Clamp strict pour la stabilité numérique
            delta = torch.clamp(delta, -1e-3, 1e-3)

            weight.data += delta.to(weight.device)

        return weight
```

---

### 2.4 HippocampalMemory — Mémoire épisodique SDM

```python
# memory/hippocampus.py

import torch
import torch.nn.functional as F


class HippocampalMemory:
    """
    Sparse Distributed Memory inspirée de Kanerva (1988).

    Principe :
      Les souvenirs sont stockés de façon DISTRIBUÉE sur N locations.
      Récupération par SIMILARITÉ PARTIELLE :
        fragment → souvenir complet (comme l'hippocampe biologique)
      Jamais effacée — s'enrichit en continu.

    Différence avec un vectorstore classique :
      - Adresses binaires aléatoires (hard locations)
      - Lecture/écriture dans un rayon de Hamming
      - Récupération par vote majoritaire
    """

    def __init__(
        self,
        address_dim: int = 256,
        content_dim: int = 2560,      # dimension hidden Qwen
        n_locations: int = 10000,
        activation_radius: int = 115  # ~45% de bits → ~activation_radius/address_dim
    ):
        self.address_dim = address_dim
        self.content_dim = content_dim
        self.n_locations = n_locations
        self.activation_radius = activation_radius

        # Hard locations — adresses binaires fixes et aléatoires
        self.addresses = torch.randint(
            0, 2, (n_locations, address_dim)
        ).float()

        # Contenu — s'accumule à chaque write
        self.contents = torch.zeros(n_locations, content_dim)

        # Compteurs d'accès par location
        self.access_counts = torch.zeros(n_locations)

        # Historique des métadonnées
        self.metadata = []

    def _to_binary_address(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Convertit un embedding continu en adresse binaire.
        Prend les address_dim premières dimensions, binarisées au seuil médian.
        """
        flat = embedding.detach().cpu().float().flatten()

        # Adapter à address_dim
        if flat.shape[0] > self.address_dim:
            flat = flat[:self.address_dim]
        elif flat.shape[0] < self.address_dim:
            flat = F.pad(flat, (0, self.address_dim - flat.shape[0]))

        return (flat > flat.median()).float()

    def _get_active_locations(self, address: torch.Tensor) -> torch.Tensor:
        """
        Retourne les indices des locations dans le rayon d'activation.
        Rayon défini par distance de Hamming ≤ activation_radius.
        """
        diffs = (self.addresses - address).abs().sum(dim=1)
        return (diffs <= self.activation_radius).nonzero(as_tuple=True)[0]

    def write(
        self,
        embedding: torch.Tensor,
        content: torch.Tensor,
        metadata: dict = None
    ):
        """
        Écrit un souvenir distribué sur toutes les locations actives.
        Écriture additive — les souvenirs s'accumulent.
        """
        address = self._to_binary_address(embedding)
        active = self._get_active_locations(address)

        if len(active) == 0:
            return

        # Préparer le contenu
        content_flat = content.detach().cpu().float().flatten()
        if content_flat.shape[0] != self.content_dim:
            content_flat = F.interpolate(
                content_flat.unsqueeze(0).unsqueeze(0),
                size=self.content_dim,
                mode='linear',
                align_corners=False
            ).squeeze()

        # Écriture distribuée
        self.contents[active] += content_flat
        self.access_counts[active] += 1

        if metadata:
            self.metadata.append(metadata)

    def read(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Lit la mémoire depuis un embedding partiel.
        Retourne le souvenir reconstruit par vote majoritaire.
        """
        address = self._to_binary_address(embedding)
        active = self._get_active_locations(address)

        if len(active) == 0:
            return torch.zeros(self.content_dim)

        # Vote majoritaire sur les locations actives
        return self.contents[active].mean(dim=0)

    def consolidate(self, threshold: int = 5):
        """
        Élagage synaptique — supprime les locations peu accédées.
        Inspiré de la consolidation mémorielle pendant le sommeil.
        """
        low_access = self.access_counts < threshold
        self.contents[low_access] = 0
        self.access_counts[low_access] = 0

    def stats(self) -> dict:
        """Statistiques de la mémoire."""
        return {
            'memories_stored': len(self.metadata),
            'active_locations': (self.access_counts > 0).sum().item(),
            'total_locations': self.n_locations,
            'usage_pct': (self.access_counts > 0).float().mean().item() * 100
        }
```

---

### 2.5 BrainConfig et utils

```python
# config.py

from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class BrainConfig:
    # LLM
    model_name: str = "Qwen/Qwen3-4B"
    llm_hidden_dim: int = 2560           # Qwen3-4B hidden size
    intercept_layers: List[int] = field(
        default_factory=lambda: [8, 16, 24, 36]
    )

    # Modules CfC+SNN
    module_hidden_dim: int = 64          # dimension interne des modules
    n_modules: int = 4                   # 1 par point d'intercept

    # STDP
    stdp_lr_plus: float = 0.01
    stdp_lr_minus: float = 0.01
    dopamine_threshold: float = 0.3      # seuil de saillance

    # Hippocampe
    memory_address_dim: int = 256
    memory_n_locations: int = 10000
    salience_threshold: float = 0.3      # erreur min pour mémoriser

    # État global
    state_dim: int = 512


# utils/device.py

def get_device() -> torch.device:
    """
    Détection automatique du GPU.
    ROCm (AMD) se présente comme CUDA dans PyTorch.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU détecté : {name} ({vram_gb:.1f} GB)")

        if vram_gb < 15.0:
            print("⚠️  VRAM < 15 GB — Qwen3-4B risque OOM")
            print("   → Fallback recommandé : Qwen/Qwen3-4B")

        return device

    print("⚠️  Pas de GPU détecté — fallback CPU (très lent)")
    return torch.device("cpu")
```

---

### 2.6 BrainHybridModel — Assemblage complet

```python
# model.py

import torch
import torch.nn as nn
from .config import BrainConfig
from .llm.qwen_wrapper import QwenWrapper
from .core.brain_module import BrainModule
from .core.stdp import STDPLearning
from .memory.hippocampus import HippocampalMemory
from .utils.device import get_device


class BrainHybridModel(nn.Module):
    """
    Modèle principal — assemble Qwen + modules CfC+SNN + hippocampe.

    Flux de traitement :
      1. Input → QwenWrapper → hidden states couches 8, 16, 24, 36
      2. Chaque BrainModule reçoit une couche et prédit la suivante
      3. Erreur de prédiction → signal dopamine → STDP
      4. Si saillant → stockage dans l'hippocampe
      5. Qwen génère la réponse finale

    Ce qui apprend : uniquement les modules CfC+SNN (via STDP)
    Ce qui reste gelé : le Qwen entier
    """

    def __init__(self, config: BrainConfig = None):
        super().__init__()
        self.config = config or BrainConfig()

        # Noyau LLM — gelé
        self.llm = QwenWrapper(self.config.model_name)
        self.device = self.llm.device

        # 4 modules CfC+SNN aux points d'intercept
        self.brain_modules = nn.ModuleList([
            BrainModule(
                input_dim=self.config.llm_hidden_dim,
                hidden_dim=self.config.module_hidden_dim,
                device=self.device
            )
            for _ in range(self.config.n_modules)
        ])

        # STDP indépendant pour chaque module
        self.stdp_learners = [
            STDPLearning(
                lr_plus=self.config.stdp_lr_plus,
                lr_minus=self.config.stdp_lr_minus
            )
            for _ in range(self.config.n_modules)
        ]

        # Hippocampe — mémoire épisodique persistante
        self.hippocampus = HippocampalMemory(
            address_dim=self.config.memory_address_dim,
            content_dim=self.config.llm_hidden_dim,
            n_locations=self.config.memory_n_locations
        )

        # État global persistant — jamais resetté
        self.global_state = torch.zeros(
            self.config.state_dim,
            device=self.device
        )

        # Historique des métriques
        self.error_history = []
        self.step_count = 0

    def forward(self, prompt: str, learn: bool = True) -> dict:
        """
        Traitement complet d'un input avec apprentissage STDP optionnel.

        prompt : texte d'entrée
        learn  : si True, applique STDP après le forward
        """
        # Extraire les représentations internes du Qwen
        layer_reps = self.llm.get_layer_representations(
            prompt,
            layers=self.config.intercept_layers    # [8, 16, 24, 36]
        )

        errors = []

        # Passer dans les 4 modules
        for i, (module, stdp) in enumerate(
            zip(self.brain_modules, self.stdp_learners)
        ):
            prediction, spikes = module(layer_reps[i])

            # Erreur de prédiction vs couche suivante
            if i < len(layer_reps) - 1:
                error = module.compute_prediction_error(
                    prediction, layer_reps[i + 1]
                )
                error_val = error.abs().mean().item()
                errors.append(error_val)

                # Signal dopamine depuis l'erreur
                stdp.update_dopamine(error_val, self.config.dopamine_threshold)

                # STDP si learning activé
                if learn and module.pre_trace is not None:
                    for param in module.cfc.parameters():
                        # Uniquement les matrices de poids 2D
                        if param.requires_grad and len(param.shape) == 2:
                            stdp.apply(
                                param,
                                module.pre_trace,
                                module.post_trace
                            )

        # Mettre à jour l'état global (moyenne pondérée)
        last_rep_mean = layer_reps[-1].mean(dim=1).squeeze().detach()
        if last_rep_mean.shape[0] >= self.config.state_dim:
            last_rep_mean = last_rep_mean[:self.config.state_dim]
        self.global_state = (
            0.95 * self.global_state +
            0.05 * last_rep_mean.to(self.device)
        )

        # Stocker dans l'hippocampe si saillant
        mean_error = sum(errors) / len(errors) if errors else 0.0
        self.error_history.append(mean_error)
        self.step_count += 1

        if learn and mean_error > self.config.salience_threshold:
            self.hippocampus.write(
                embedding=layer_reps[0].mean(dim=1).squeeze(),
                content=layer_reps[-1].mean(dim=1).squeeze(),
                metadata={
                    'step': self.step_count,
                    'prompt': prompt[:100],
                    'error': mean_error
                }
            )

        # Générer la réponse avec le Qwen
        response = self.llm.generate(prompt)

        return {
            'response': response,
            'prediction_errors': errors,
            'mean_error': mean_error,
            'dopamine': [s.dopamine_signal for s in self.stdp_learners],
            'memories_stored': len(self.hippocampus.metadata),
            'global_state_norm': self.global_state.norm().item(),
            'step': self.step_count
        }

    def remember(self, query: str) -> torch.Tensor:
        """
        Interroge l'hippocampe depuis une requête textuelle partielle.
        Retourne le vecteur de souvenir le plus proche.
        """
        query_reps = self.llm.get_layer_representations(query, layers=[8])
        query_emb = query_reps[0].mean(dim=1).squeeze()
        return self.hippocampus.read(query_emb)

    def save_state(self, path: str):
        """Sauvegarde complète — modules + hippocampe + état global."""
        torch.save({
            'brain_modules': self.brain_modules.state_dict(),
            'hippocampus_contents': self.hippocampus.contents,
            'hippocampus_counts': self.hippocampus.access_counts,
            'hippocampus_metadata': self.hippocampus.metadata,
            'global_state': self.global_state,
            'error_history': self.error_history,
            'step_count': self.step_count,
            'stdp_dopamine': [s.dopamine_signal for s in self.stdp_learners],
            'config': self.config,
        }, path)
        print(f"💾 Checkpoint sauvegardé : {path}")

    def load_state(self, path: str):
        """Reprend depuis un checkpoint — aucune perte de mémoire."""
        ckpt = torch.load(path, map_location=self.device)
        self.brain_modules.load_state_dict(ckpt['brain_modules'])
        self.hippocampus.contents = ckpt['hippocampus_contents']
        self.hippocampus.access_counts = ckpt['hippocampus_counts']
        self.hippocampus.metadata = ckpt['hippocampus_metadata']
        self.global_state = ckpt['global_state'].to(self.device)
        self.error_history = ckpt['error_history']
        self.step_count = ckpt['step_count']
        for i, d in enumerate(ckpt['stdp_dopamine']):
            self.stdp_learners[i].dopamine_signal = d
        print(f"✅ Reprise depuis step {self.step_count}")
```

---

## 3. Structure de fichiers à créer

```
brain_hybrid/
├── pyproject.toml
├── AGENTS.md                          # Ce fichier
├── brain_hybrid/
│   ├── __init__.py
│   ├── config.py                      # BrainConfig
│   ├── model.py                       # BrainHybridModel
│   ├── core/
│   │   ├── __init__.py
│   │   ├── brain_module.py            # BrainModule (CfC + SNN)
│   │   └── stdp.py                    # STDPLearning + dopamine
│   ├── memory/
│   │   ├── __init__.py
│   │   └── hippocampus.py             # HippocampalMemory (SDM)
│   ├── llm/
│   │   ├── __init__.py
│   │   └── qwen_wrapper.py            # QwenWrapper HF
│   ├── utils/
│   │   ├── __init__.py
│   │   └── device.py                  # get_device()
│   └── eval/
│       ├── __init__.py
│       └── continual_test.py          # Tests d'apprentissage continu
├── tests/
│   ├── test_qwen_wrapper.py
│   ├── test_brain_module.py
│   ├── test_stdp.py
│   ├── test_hippocampus.py
│   └── test_model.py
└── notebooks/
    └── colab_brain_hybrid.ipynb
```

---

## 4. pyproject.toml

```toml
[project]
name = "brain-hybrid"
version = "0.1.0"
description = "Brain-inspired hybrid architecture : Qwen3 + CfC + SNN + STDP + SDM"
requires-python = ">=3.11"

dependencies = [
    "torch>=2.2",
    "transformers>=4.40",
    "ncps>=1.0",
    "snntorch>=0.9",
    "accelerate>=0.27",
    "numpy",
    "tqdm",
]

[project.optional-dependencies]
dev = ["pytest", "matplotlib", "jupyter", "ipywidgets"]
quantized = ["torchao"]    # Phase 2 : si besoin de réduire VRAM

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:BuildBackend"
```

---

## 5. Notebook Google Colab

```python
# notebooks/colab_brain_hybrid.ipynb

# ── Cellule 1 — Mount Google Drive + Installation ─────────────────
from google.colab import drive
drive.mount('/content/drive')

!pip install transformers torch ncps snntorch accelerate tqdm -q

import os, glob, torch

DRIVE_PATH    = '/content/drive/MyDrive/brain_hybrid/'
CHECKPOINT_DIR = DRIVE_PATH + 'checkpoints/'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"GPU : {torch.cuda.get_device_name(0)}")
print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Cellule 2 — Cloner le repo et charger le modèle ───────────────
# Remplacer par ton repo GitHub
!git clone https://github.com/TON_USER/brain_hybrid /content/brain_hybrid
%cd /content/brain_hybrid
!pip install -e . -q

from brain_hybrid.model import BrainHybridModel, BrainConfig

config = BrainConfig(
    model_name="Qwen/Qwen3-4B"    # HF télécharge automatiquement ~15GB
    # 
)
model = BrainHybridModel(config)
print("✅ Modèle chargé")

# ── Cellule 3 — Reprise depuis checkpoint (si existant) ───────────
checkpoints = sorted(glob.glob(CHECKPOINT_DIR + 'step_*.pt'))
if checkpoints:
    model.load_state(checkpoints[-1])
else:
    print("Nouveau run — aucun checkpoint trouvé")

# ── Cellule 4 — Boucle d'apprentissage continue ───────────────────
prompts = [
    "Explique comment fonctionne la photosynthèse",
    "Qu'est-ce que la conscience selon les neurosciences ?",
    "Comment le cerveau consolide-t-il les souvenirs ?",
    "Décris le fonctionnement d'un neurone biologique",
    "Quelle est la différence entre mémoire courte et longue durée ?",
    # Ajouter tes propres prompts ici
]

start = model.step_count

for step in range(start, start + 500):
    prompt = prompts[step % len(prompts)]
    result = model.forward(prompt, learn=True)

    if step % 10 == 0:
        errs = [f"{e:.3f}" for e in result['prediction_errors']]
        dops = [f"{d:.3f}" for d in result['dopamine']]
        print(f"\nStep {step}")
        print(f"  Erreurs     : {errs}")
        print(f"  Dopamine    : {dops}")
        print(f"  Mémoires    : {result['memories_stored']}")
        print(f"  Réponse     : {result['response'][:80]}...")

    # Sauvegarde toutes les 50 steps
    if step % 50 == 0 and step > start:
        path = f"{CHECKPOINT_DIR}step_{step:06d}.pt"
        model.save_state(path)

# ── Cellule 5 — Test mémoire épisodique ───────────────────────────
# Apprendre un fait précis
model.forward("Mon chat s'appelle Luna et il est roux", learn=True)
model.forward("Luna adore jouer avec des balles de laine", learn=True)

# Interroger avec un fragment
recalled = model.remember("chat Luna")
print(f"Souvenir récupéré — norme : {recalled.norm().item():.3f}")
print(f"Mémoires totales : {len(model.hippocampus.metadata)}")
print(f"Stats hippocampe : {model.hippocampus.stats()}")

# ── Cellule 6 — Visualiser l'évolution des erreurs ────────────────
import matplotlib.pyplot as plt

if len(model.error_history) > 10:
    plt.figure(figsize=(12, 4))
    plt.plot(model.error_history)
    plt.title("Erreur de prédiction moyenne au fil du temps")
    plt.xlabel("Step")
    plt.ylabel("Erreur")
    plt.grid(True, alpha=0.3)
    plt.show()
    print(f"Erreur initiale  : {model.error_history[0]:.4f}")
    print(f"Erreur finale    : {model.error_history[-1]:.4f}")
    print(f"Amélioration     : {(1 - model.error_history[-1]/model.error_history[0])*100:.1f}%")
```

---

## 6. Tests

```python
# tests/test_qwen_wrapper.py
import torch, pytest
from brain_hybrid.llm.qwen_wrapper import QwenWrapper

def test_hidden_states_distincts():
    """Les representations des couches 8 et 36 doivent être différentes."""
    w = QwenWrapper()
    reps = w.get_layer_representations("Le chat mange", layers=[8, 36])
    sim = torch.nn.functional.cosine_similarity(
        reps[0].mean(dim=1), reps[1].mean(dim=1)
    ).item()
    assert sim < 0.95, f"Couches trop similaires : {sim:.3f}"

def test_generate():
    """La génération retourne un string non vide."""
    w = QwenWrapper()
    resp = w.generate("Bonjour, comment vas-tu ?", max_new_tokens=32)
    assert isinstance(resp, str) and len(resp) > 0


# tests/test_brain_module.py
import torch, pytest
from brain_hybrid.core.brain_module import BrainModule

def test_forward_shape():
    """Le module retourne la bonne forme."""
    m = BrainModule(input_dim=2560, hidden_dim=64)
    x = torch.randn(1, 10, 2560)
    pred, spk = m(x)
    assert pred.shape == (1, 10, 2560)

def test_state_persists():
    """L'état CfC doit persister entre appels."""
    m = BrainModule(input_dim=2560, hidden_dim=64)
    x = torch.randn(1, 5, 2560)
    m(x)
    state_after_1 = m.cfc_state.clone() if m.cfc_state is not None else None
    m(x)
    state_after_2 = m.cfc_state
    if state_after_1 is not None:
        assert not torch.allclose(state_after_1, state_after_2), \
            "L'état CfC ne change pas entre les appels"


# tests/test_stdp.py
import torch
from brain_hybrid.core.stdp import STDPLearning

def test_poids_changent():
    """Les poids doivent changer après apply()."""
    stdp = STDPLearning()
    w = torch.randn(32, 32)
    w_orig = w.clone()
    pre = torch.randn(1, 10, 64)
    post = torch.randn(1, 10, 32)
    stdp.apply(w, pre, post)
    assert not torch.allclose(w, w_orig), "Les poids n'ont pas changé"

def test_changement_stable():
    """Les changements de poids doivent être < 0.1% de la norme."""
    stdp = STDPLearning()
    w = torch.randn(32, 32)
    w_orig = w.clone()
    pre = torch.randn(1, 10, 64)
    post = torch.randn(1, 10, 32)
    stdp.apply(w, pre, post)
    rel_change = (w - w_orig).abs().mean() / w_orig.abs().mean()
    assert rel_change < 0.01, f"Changement trop grand : {rel_change:.4f}"

def test_dopamine_clamp():
    """Le signal dopamine doit rester entre 0 et 1."""
    stdp = STDPLearning()
    for err in [0.0, 0.1, 0.5, 1.0, 10.0, -5.0]:
        stdp.update_dopamine(err)
        assert 0.0 <= stdp.dopamine_signal <= 1.0


# tests/test_hippocampus.py
import torch
from brain_hybrid.memory.hippocampus import HippocampalMemory

def test_write_read_roundtrip():
    """Un souvenir écrit doit être récupérable."""
    h = HippocampalMemory(address_dim=64, content_dim=128, n_locations=1000)
    emb = torch.randn(128)
    content = torch.randn(128)
    h.write(emb, content)
    recalled = h.read(emb)
    sim = torch.nn.functional.cosine_similarity(
        content.unsqueeze(0), recalled.unsqueeze(0)
    ).item()
    assert sim > 0.5, f"Recall trop faible : {sim:.3f}"

def test_metadata_stockee():
    """Les métadonnées doivent être stockées après write."""
    h = HippocampalMemory(address_dim=64, content_dim=128, n_locations=1000)
    h.write(torch.randn(128), torch.randn(128), metadata={'test': True})
    assert len(h.metadata) == 1
```

---

## 7. Ordre d'implémentation STRICT

```
Étape 1 : config.py + utils/device.py
  → Aucune dépendance
  → Valider get_device() retourne cuda sur AMD ROCm

Étape 2 : llm/qwen_wrapper.py          ← PRIORITÉ ABSOLUE
  → Valider AVANT TOUT AUTRE CODE
  → Test : hidden_states[8] vs hidden_states[36] cosine_sim < 0.95
  → 

Étape 3 : core/brain_module.py
  → Valider forward : (1, seq_len, 2560) → (1, seq_len, 2560)
  → Valider état CfC persiste entre appels

Étape 4 : core/stdp.py
  → Valider poids changent après apply()
  → Valider changement < 1% de la norme des poids

Étape 5 : memory/hippocampus.py
  → Valider write/read round-trip cosine_sim > 0.5

Étape 6 : model.py
  → Assemblage complet
  → Valider forward() retourne tous les champs

Étape 7 : tests/ complets
  → Tous les tests doivent passer avant de continuer

Étape 8 : notebooks/colab_brain_hybrid.ipynb
  → Notebook autonome, fonctionne sans setup local
```

---

## 8. Critères de succès Phase 1

| Test | Condition de succès |
|---|---|
| Hidden states distincts | cosine_sim(layer8, layer36) < 0.95 |
| Prédiction s'améliore | erreur(step 100) < erreur(step 1) × 0.9 |
| STDP stable | changement poids < 1% par step |
| Hippocampe recall | cosine_sim(write, read) > 0.5 |
| Mémoire épisodique | "Luna" récupérable après write + read partiel |
| Pas d'OOM | run complet 100 steps sans CUDA OOM |

---

## 9. Règles absolues — À respecter sans exception

1. **Ne jamais modifier les poids du Qwen** — `requires_grad=False` partout
2. **Si OOM avec Qwen3-4B** → basculer immédiatement sur `Qwen/Qwen3-4B`
3. **Détacher les états CfC** après chaque forward (`.detach()`)
4. **STDP uniquement sous `torch.no_grad()`**
5. **Hippocampe jamais resetté** entre sessions (save/load systématique)
6. **Dopamine signal** toujours clampé entre 0.0 et 1.0
7. **Tester l'étape 2 en premier** — si les hidden states ne fonctionnent pas, tout s'arrête

---

## 10. Budget VRAM

```
Qwen3-4B bfloat16    : ~8 GB   ← marge confortable sur 16 GB
4 modules CfC+SNN    :  ~0.5 GB
Hippocampe (SDM)     :  ~0.2 GB
États + buffers      :  ~0.5 GB
─────────────────────────────────
TOTAL estimé         : ~9.5 GB / 16 GB
Marge             :  ~6.5 GB  ✅ confortable

Si OOM → Qwen/Qwen3-4B (~7 GB) → marge ~8 GB confortable
Phase 2 option : torchao 6-bit sur Qwen3-4B pour récupérer ~2 GB
```

---

*Document BRAIN-HYBRID — Architecture cerveau artificiel*
*Auteur : Sylvain | Corrections : Grok (mars 2026)*
*Implémentation : Claude Code*
*Hardware cible : AMD RX 7800 XT 16 GB, ROCm 6.4, Windows 11*
*Version : 1.0 finale autonome*
