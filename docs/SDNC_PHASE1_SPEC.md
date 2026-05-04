# SDNC — Sparse Dynamic Neural Circuits
## Spécification d'architecture complète pour implémentation

> **Document destiné à Claude Code.**
> Lis ce document en entier avant d'écrire une seule ligne de code.
> L'ordre de lecture est important — chaque section dépend de la précédente.

---

## 1. Contexte et philosophie

### Ce qu'on construit

Une architecture de modèle d'IA fondamentalement différente des transformers actuels.
Pas une modification de transformer. Pas un MoE classique. Une nouvelle architecture.

### Pourquoi pas les transformers

Les transformers ont trois défauts fondamentaux pour notre objectif :
1. **Poids statiques après entraînement** — le modèle n'apprend plus une fois déployé
2. **Attention globale dense** — tout parle à tout, coûteux et non-biologique
3. **Gradient global destructif** — le fine-tuning écrase ce qui existait (catastrophic forgetting)

### L'inspiration biologique

Deux sources directes :

**C. elegans (ver, 302 neurones)**
→ Origine des Liquid Neural Networks (MIT, Hasani 2020)
→ Démontre qu'un très petit nombre de neurones liquides produit des comportements complexes et adaptatifs

**Drosophile (mouche, connectome complet 2023)**
→ 140 000 neurones, 50M synapses cartographiées
→ Révèle que les circuits biologiques sont : sparse, spécialisés, récurrents, câblés précisément
→ Chaque module fait UNE chose. L'intelligence émerge des connexions entre modules.

### Le principe central

```
Pas un gros modèle dense qui sait tout.
Des millions de micro-circuits ultra-spécialisés et légers.
Chacun répond à UN concept précis.
À tout moment : seulement 1-5% s'activent (sparsité).
Les circuits actifs se parlent entre eux (récurrence dynamique).
Chaque circuit est un LNN — son état évolue PENDANT l'inférence.
```

---

## 2. Principes fondamentaux — Non négociables

Ces principes ne peuvent pas être compromis sans trahir l'architecture :

### P1 — Sparsité
À tout moment, maximum 5% des circuits sont actifs.
Une image de chat active : [chat, oreilles, fourrure, couleur, animal]
Une image de voiture active : [voiture, métal, roues, couleur, objet]
Les circuits "chat" ne s'activent PAS pour une voiture.

### P2 — Liquidité
Chaque micro-circuit est un LNN (Liquid Neural Network).
Son état interne change PENDANT le traitement de l'input — pas après.
Le réseau est différent à t=0 et t=100ms sur le même input.

### P3 — Localité de l'apprentissage
Les mises à jour de poids sont LOCALES.
Seuls les circuits impliqués dans une expérience sont modifiés.
Pas de backpropagation globale qui écrase tout.
Règle inspirée Hebbian : "circuits qui s'activent ensemble se renforcent ensemble"

### P4 — Séparation des mémoires
Trois systèmes distincts, comme le cerveau :
- **Mémoire de travail** : contexte immédiat, dans l'état LNN
- **Mémoire épisodique** : souvenirs d'expériences, base vectorielle externe
- **Mémoire procédurale** : savoir-faire, patterns consolidés dans les poids

### P5 — Câblage sparse inspiré connectome
Les connexions entre circuits ne sont PAS fully-connected.
Chaque circuit est connecté à un sous-ensemble précis d'autres circuits.
Ce câblage est appris, pas fixé à l'avance.

---

## 3. Architecture complète

### Vue d'ensemble

```
┌─────────────────────────────────────────────────────────┐
│                    SDNC MODEL                           │
│                                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │CIRCUIT   │ │CIRCUIT   │ │CIRCUIT   │ │CIRCUIT   │  │
│  │TEXTE     │ │VISION    │ │VOIX      │ │SIGNAUX   │  │
│  │LNN       │ │LNN       │ │LNN       │ │LNN       │  │
│  │fréq:lent │ │fréq:fps  │ │fréq:kHz  │ │fréq:var  │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│       └────────────┼────────────┼─────────────┘        │
│                    ↓            ↓                       │
│           ┌────────────────────────┐                   │
│           │  INTÉGRATEUR TEMPOREL  │                   │
│           │  LNN récurrent central │                   │
│           │  synchronise rythmes   │                   │
│           │  état persiste ∞       │                   │
│           └───────────┬────────────┘                   │
│                       │                                 │
│           ┌───────────▼────────────┐                   │
│           │  BANQUE DE MICRO-      │                   │
│           │  CIRCUITS SPÉCIALISÉS  │                   │
│           │  N circuits LNN        │                   │
│           │  activation sparse 5%  │                   │
│           │  câblage NCP           │                   │
│           └───────────┬────────────┘                   │
│                       │                                 │
│           ┌───────────▼────────────┐                   │
│           │  MÉMOIRE EXTERNE       │                   │
│           │  épisodique/sémantique │                   │
│           │  procédurale           │                   │
│           └───────────┬────────────┘                   │
│                       │                                 │
│           ┌───────────▼────────────┐                   │
│           │  CIRCUIT DÉCISION      │                   │
│           │  génère output/action  │                   │
│           └────────────────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

### Les circuits d'entrée (encodeurs modaux)

Chaque modalité a sa propre fréquence naturelle et son circuit dédié :

| Circuit | Input | Fréquence | Encodeur recommandé |
|---------|-------|-----------|---------------------|
| Texte | tokens | événementiel | embeddings légers |
| Vision | frames | 10-30 fps | CLIP/SigLIP frozen |
| Voix | audio | 16kHz → features | Whisper tiny frozen |
| Signaux | vecteurs numériques | variable | MLP custom |

**Important** : Les encodeurs (CLIP, Whisper) sont **gelés**. Ils servent de traducteurs vers l'espace vectoriel. L'apprentissage se passe dans les LNN, pas dans les encodeurs.

### La banque de micro-circuits

C'est le cœur de l'architecture.

```python
# Concept — chaque micro-circuit est un LNN minimal
class MicroCircuit:
    """
    Répond à UN concept ou pattern précis.
    Ultra-léger individuellement.
    Fort collectivement via connexions.
    """
    units: int = 8          # très petit intentionnellement
    wiring: NCP             # câblage sparse, pas fully-connected
    state: Tensor           # état liquide persistant
    specialization: str     # concept appris (émergent, pas défini)
    activation_threshold: float  # seuil d'activation sparse
```

**Activation sparse** : un routeur décide quels circuits activer selon l'input.
Maximum 5% actifs simultanément.
Les circuits actifs s'échangent leurs états via le câblage NCP.

### L'intégrateur temporel

Problème clé : texte arrive à des rythmes différents de la vision et de l'audio.
L'intégrateur synchronise sans perdre d'information.

```python
# Il maintient UN état persistant entre tous les appels
# Cet état encode "ce qui se passe en ce moment"
# Il ne repart jamais de zéro
integrator_state: Tensor  # persiste entre sessions si nécessaire
```

### La mémoire externe

Trois compartiments distincts :

```
Mémoire de travail  → état interne des LNN actifs (implicite)
Mémoire épisodique  → base vectorielle (PostgreSQL + pgvector)
                      "hier j'ai vu X dans ce contexte"
Mémoire procédurale → patterns consolidés après N répétitions
                      "quand je vois X, faire Y"
```

**Mécanisme de consolidation** : après chaque interaction,
un module de saillance décide si l'expérience mérite d'être stockée.
Pas tout stocker — seulement ce qui est nouveau ou important.

---

## 4. Apprentissage — Comment le modèle évolue

### Apprentissage local (en continu, pendant l'inférence)

```
Règle de base (inspirée Hebbian + Oja pour stabilité) :

Si circuit A et circuit B s'activent ensemble :
    → renforcer leur connexion
    → magnitude : proportionnelle à la co-activation
    → normalisation Oja : évite la divergence des poids

Si circuit C ne s'active pas depuis N steps :
    → affaiblir progressivement ses connexions (élagage)
```

### Création de nouveaux circuits

Quand un pattern inconnu est rencontré :
1. Aucun circuit existant ne s'active fortement
2. Un nouveau micro-circuit est instancié
3. Il commence avec des poids aléatoires légers
4. Il se spécialise progressivement via activation répétée

### Ce qui NE change PAS

- Les encodeurs (CLIP, Whisper) : gelés
- L'architecture globale : fixe
- Le câblage de base : fixe (évolue lentement)

---

## 5. Prototype minimal — Phase 1

### Objectif de la Phase 1

Valider le principe sur une tâche simple avant de scaler.

**Tâche** : reconnaissance et mémorisation de patterns visuels simples
**Input** : vision uniquement (pas encore multimodal)
**Mesure de succès** : le modèle reconnaît un pattern après 1-3 expositions

### Stack technique Phase 1

```
Python 3.11+
PyTorch 2.x
ncps (Neural Circuit Policies — lib officielle MIT)
numpy
```

Installation :
```bash
pip install torch ncps numpy
```

### Structure de fichiers Phase 1

```
sdnc/
├── core/
│   ├── micro_circuit.py      # Un LNN minimal, un concept
│   ├── circuit_bank.py       # La banque de N micro-circuits
│   ├── sparse_router.py      # Décide quels circuits activer
│   ├── integrator.py         # Synchronise les modalités
│   └── hebbian_update.py     # Mise à jour locale des poids
├── memory/
│   ├── working_memory.py     # État LNN courant
│   ├── episodic_memory.py    # Stockage vectoriel
│   └── consolidation.py      # Quoi mémoriser ?
├── encoders/
│   └── vision_encoder.py     # CLIP frozen → vecteurs
├── train/
│   └── phase1_vision.py      # Entraînement phase 1
└── eval/
    └── one_shot_test.py       # Test : reconnaît après 1 expo ?
```

### Code de départ — micro_circuit.py

```python
import torch
import torch.nn as nn
from ncps.torch import LTC
from ncps.wirings import NCP

class MicroCircuit(nn.Module):
    """
    Un micro-circuit spécialisé.
    Répond à UN concept via un LNN minimal.
    Son état persiste entre les appels — il a une mémoire.
    """
    
    def __init__(self, input_dim: int, circuit_id: int):
        super().__init__()
        self.circuit_id = circuit_id
        
        # Câblage sparse inspiré connectome
        # Petit intentionnellement — la force vient du collectif
        self.wiring = NCP(
            inter_neurons=6,
            command_neurons=4,
            motor_neurons=2,
            sensory_fanout=3,
            inter_fanout=2,
            recurrent_command_synapses=3,
            motor_fanin=3,
        )
        
        # Le cœur liquide — état change pendant l'inférence
        self.ltc = LTC(
            input_size=input_dim,
            wiring=self.wiring,
            batch_first=True
        )
        
        # État persistant — ne repart pas de zéro entre appels
        self.hidden_state = None
        
        # Seuil d'activation pour la sparsité
        self.activation_score = 0.0
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        x : (batch, seq, input_dim)
        Retourne : (output, activation_score)
        L'état hidden persiste entre les appels.
        """
        output, self.hidden_state = self.ltc(x, self.hidden_state)
        
        # Score d'activation — force de réponse à cet input
        self.activation_score = output.abs().mean().item()
        
        return output, self.activation_score
    
    def reset_state(self):
        """Reset de l'état — à appeler entre sessions si nécessaire."""
        self.hidden_state = None
        self.activation_score = 0.0
```

### Code de départ — sparse_router.py

```python
import torch
import torch.nn as nn
from typing import List
from .micro_circuit import MicroCircuit

class SparseRouter(nn.Module):
    """
    Décide quels micro-circuits activer pour un input donné.
    Maintient la sparsité à maximum SPARSITY_RATIO.
    """
    
    SPARSITY_RATIO = 0.05  # max 5% de circuits actifs
    
    def __init__(self, input_dim: int, n_circuits: int):
        super().__init__()
        self.n_circuits = n_circuits
        self.max_active = max(1, int(n_circuits * self.SPARSITY_RATIO))
        
        # Réseau de routage léger
        # Projette l'input vers les scores d'activation
        self.router = nn.Sequential(
            nn.Linear(input_dim, n_circuits),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> List[int]:
        """
        x : vecteur d'input
        Retourne : indices des circuits à activer
        Garantit : max SPARSITY_RATIO circuits actifs
        """
        scores = self.router(x.mean(dim=1))  # (batch, n_circuits)
        
        # Top-k sparse — seulement les plus pertinents
        _, top_indices = scores.topk(self.max_active, dim=-1)
        
        return top_indices[0].tolist()  # liste d'indices actifs
```

### Code de départ — hebbian_update.py

```python
import torch

def oja_update(weight: torch.Tensor,
               pre: torch.Tensor,
               post: torch.Tensor,
               lr: float = 1e-4) -> torch.Tensor:
    """
    Règle d'Oja — version stable de Hebbian.
    "Neurons that fire together wire together"
    + normalisation pour éviter la divergence.
    
    Mise à jour LOCALE — ne touche que ce circuit.
    Pas de gradient global.
    
    w_new = w + lr * post * (pre - post * w)
    """
    delta = lr * torch.outer(
        post.mean(0),
        pre.mean(0) - post.mean(0) @ weight
    )
    return weight + delta

def apply_hebbian_to_circuit(circuit, activations: dict, lr: float = 1e-4):
    """
    Applique la mise à jour Oja uniquement sur les circuits actifs.
    Appelé APRÈS chaque forward pass.
    """
    with torch.no_grad():
        for layer_name, (pre, post) in activations.items():
            if hasattr(circuit.ltc, layer_name):
                layer = getattr(circuit.ltc, layer_name)
                if hasattr(layer, 'weight'):
                    layer.weight.data = oja_update(
                        layer.weight.data, pre, post, lr
                    )

def synaptic_pruning(circuit, threshold: float = 1e-5):
    """
    Élagage synaptique — supprime les connexions trop faibles.
    Inspiré de l'élagage synaptique biologique.
    Libère de la capacité pour de nouveaux apprentissages.
    """
    with torch.no_grad():
        for param in circuit.ltc.parameters():
            mask = param.data.abs() > threshold
            param.data *= mask.float()
```

---

## 6. Ce que Claude Code doit faire — Instructions précises

### Ordre d'implémentation STRICT

1. `core/micro_circuit.py` — valider qu'un LNN minimal fonctionne
2. `core/sparse_router.py` — valider la sparsité
3. `core/circuit_bank.py` — N circuits ensemble
4. `core/hebbian_update.py` — apprentissage local
5. `encoders/vision_encoder.py` — CLIP frozen en test
6. `eval/one_shot_test.py` — test de validation phase 1

### Règles absolues

- **Ne pas utiliser de backpropagation globale** pour l'apprentissage continu
- **Ne pas utiliser de transformer** comme composant central
- **Toujours utiliser NCP wiring** (pas AutoNCP sauf pour tests rapides)
- **L'état LNN persiste** entre les appels — ne pas reset à chaque inférence
- **Maximum 5% de circuits actifs** à tout moment — vérifier dans les tests

### Ce qu'on NE fait PAS en Phase 1

- Pas d'intégration multimodale complète
- Pas de mémoire épisodique persistante (base de données)
- Pas d'interface utilisateur
- Pas de génération de texte

Phase 1 = valider le principe de sparsité + liquidité sur vision simple.

### Critères de succès Phase 1

```python
# Test one-shot : montrer UN exemple, tester la reconnaissance
# Succès si accuracy > 70% après 1-3 expositions seulement
# Sur un dataset simple (MNIST ou CIFAR-10 subset)

def one_shot_test(model, support_image, query_images):
    """
    1. Montrer support_image une seule fois
    2. Tester sur query_images (même classe + distracteurs)
    3. Mesurer : est-ce que les bons circuits s'activent ?
    """
    pass
```

---

## 7. Hardware cible

```
GPU  : AMD RX 7800 XT (16GB VRAM) — ROCm / DirectML
CPU  : Intel i5-12600K
RAM  : 64GB DDR4 3200MHz
OS   : Windows 11
```

**Contraintes importantes** :
- Utiliser PyTorch avec DirectML ou ROCm pour AMD
- Pas de CUDA-only features
- Le modèle Phase 1 doit tenir en < 4GB VRAM
- Préférer float16 pour les calculs

```python
# Device setup pour AMD
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")  # ROCm se présente comme CUDA
else:
    # Fallback DirectML
    try:
        import torch_directml
        device = torch_directml.device()
    except:
        device = torch.device("cpu")
```

---

## 8. Questions ouvertes — à résoudre en Phase 2

Ces questions ne bloquent pas la Phase 1 mais doivent être gardées en tête :

1. **Combien de micro-circuits ?** On commence à 1000 pour la Phase 1. On scale ensuite.
2. **Comment initialiser les spécialisations ?** Émergente (aléatoire + entraînement) ou guidée ?
3. **Rythme de consolidation** : après combien d'expositions un pattern devient-il "stable" ?
4. **Câblage entre circuits** : comment le faire évoluer sans tout casser ?

---

## 9. Références techniques

```
ncps (Neural Circuit Policies) :
  pip install ncps
  https://github.com/mlech26l/ncps
  Papier : "Closed-form Continuous-time Neural Networks" (Hasani et al.)

Liquid Time-Constant Networks :
  Hasani et al., MIT, 2020
  https://arxiv.org/abs/2006.04439

Oja's Learning Rule :
  Oja, 1982 — version stable de Hebbian
  Évite la divergence des poids

NCP Wiring (Neural Circuit Policies) :
  Lechner et al., NeurIPS 2019
  Câblage sparse inspiré C. elegans
```

---

*Document généré pour le projet SDNC — Architecture par Sylvain*
*Implémentation : Claude Code*
*Version : 0.1 — Phase 1*
