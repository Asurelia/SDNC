# Partie 5 — Référence API, Configuration & Déploiement

## Table des matières

1. [Installation](#1-installation)
2. [Référence API — SDNC v2](#2-reference-api-sdnc-v2)
3. [Référence API — Brain-Hybrid](#3-reference-api-brain-hybrid)
4. [Référence API — Pipeline](#4-reference-api-pipeline)
5. [Configuration complète — SDNCConfig](#5-sdncconfig)
6. [Configuration complète — BrainConfig](#6-brainconfig)
7. [Configuration complète — DualModelConfig](#7-dualmodelconfig)
8. [Tests](#8-tests)
9. [Glossaire](#9-glossaire)
10. [Références scientifiques](#10-references-scientifiques)

---

## 1. Installation

### 1.1 Prérequis

- Python ≥ 3.11
- PyTorch ≥ 2.2 avec support CUDA ou ROCm
- 16 GB RAM minimum (80 GB pour le pipeline de distillation)
- GPU avec ≥ 8 GB VRAM pour Brain-Hybrid

### 1.2 Installation standard

```bash
# Clone du repository
git clone <repository-url>
cd SDNC-refactor-predictive-coding

# Installation en mode éditable
pip install -e .

# Avec dépendances de développement
pip install -e ".[dev]"

# Avec support quantization (Linux/CUDA uniquement)
pip install -e ".[quantized]"
```

### 1.3 Dépendances

```toml
[project]
name = "brain-hybrid"
version = "0.2.0"
requires-python = ">=3.11"

dependencies = [
    "torch>=2.2",
    "transformers>=4.45",
    "ncps>=1.0",
    "snntorch>=0.9",
    "accelerate>=0.27",
    "qwen-vl-utils>=0.0.8",
    "numpy",
    "tqdm",
    "Pillow",
]

[project.optional-dependencies]
dev = ["pytest", "matplotlib", "jupyter", "ipywidgets", "gradio"]
quantized = ["bitsandbytes>=0.43"]
```

### 1.4 Vérification de l'installation

```bash
# Vérifier le GPU
python check_gpu.py

# Test rapide
python run_quick_test.py

# Tests unitaires
pytest tests/ -v
```

---

## 2. Référence API — SDNC v2

### 2.1 `SDNCModel`

**Module** : `sdnc.model`

```python
class SDNCModel(nn.Module):
    def __init__(self, config: SDNCConfig | None = None)
```

**Paramètres** :
- `config` : Instance de `SDNCConfig`. Si `None`, utilise les valeurs par défaut.

#### Méthodes principales

##### `forward(**kwargs) → dict`

Passe avant prédictive. Accepte une seule modalité à la fois.

**Arguments nommés** :
| Argument | Type | Description |
|----------|------|-------------|
| `images` | `Tensor (B, C, H, W)` | Images préprocessées CLIP |
| `text_tokens` | `Tensor (B, seq_len)` | Tokens texte CLIP |
| `audio` | `Tensor (B, mel_features)` | Spectrogramme mel |
| `signals` | `Tensor (B, 8)` | Signaux numériques bruts |

**Retour** : `dict` avec les clés suivantes :
| Clé | Type | Dimensions | Description |
|-----|------|-----------|-------------|
| `encoder_features` | `Tensor` | `(B, 512)` | Features de l'encodeur gelé |
| `circuit_repr` | `Tensor` | `(B, 128)` | Représentation circuit normalisée L2 |
| `active_indices` | `Tensor` | `(B, k)` | Indices des circuits actifs |
| `weights` | `Tensor` | `(B, k)` | Poids de routage |
| `circuit_outputs_raw` | `Tensor` | `(B, 32)` | Sortie brute des circuits |
| `hidden_states` | `Tensor` | `(B, state_size)` | États cachés CfC |
| `prediction` | `Tensor` | `(B, 512)` | Prédiction de l'input suivant |
| `prediction_error` | `Tensor` | `(B, 512)` | Erreur = actual - predicted |
| `prediction_error_norm` | `Tensor` | `scalar` | Norme moyenne de l'erreur |
| `global_state_summary` | `dict` | — | Diagnostic du GlobalState |
| `stream_stats` | `dict` | — | Statistiques du TemporalStream |

##### `learn(labels: Tensor | None = None, **kwargs) → dict`

Entraînement avec double signal (prototypical + Hebbian).

**Arguments** :
- `labels` : `Tensor (B,)` — labels de classe (entiers). Si `None`, seul l'Hebbian s'applique.
- `**kwargs` : mêmes arguments que `forward()`.

**Retour** : `dict` (même structure que `forward()`, tenseurs détachés).

##### `recognize(support_labels, support_kwargs, query_kwargs) → Tensor`

Classification few-shot dans l'espace circuit.

**Arguments** :
- `support_labels` : `Tensor (n_way * k_shot,)` — labels du support set
- `support_kwargs` : `dict` — arguments pour les exemples support
- `query_kwargs` : `dict` — arguments pour les requêtes

**Retour** : `Tensor (n_query, n_way)` — logits (distances négatives aux prototypes).

##### `get_prediction_error(**kwargs) → dict`

Obtient l'erreur de prédiction sans apprentissage.

**Retour** :
| Clé | Type | Description |
|-----|------|-------------|
| `prediction_error_norm` | `float` | Norme moyenne |
| `prediction_error` | `Tensor` | Erreur complète |
| `prediction` | `Tensor` | Prédiction |

##### `prune()`

Élagage synaptique (poids faibles → 0) + élagage inter-circuits.

##### `reset()`

Reset complet : circuits, intégrateur, global state, stream.

##### `soft_reset()`

Reset partiel : circuits et intégrateur seulement. Garde global state et stream.

##### `get_circuit_wiring_stats() → dict`

Statistiques de câblage inter-circuits.

**Retour** :
| Clé | Type | Description |
|-----|------|-------------|
| `nonzero_connections` | `int` | Connexions non-nulles |
| `total_possible` | `int` | Connexions possibles |
| `density` | `float` | Ratio de densité |
| `mean_weight` | `float` | Poids moyen des connexions actives |
| `max_weight` | `float` | Poids maximum |

##### `get_predictive_stats() → dict`

Statistiques de prédiction à travers les circuits.

---

### 2.2 `PredictiveCircuitBank`

**Module** : `sdnc.core.predictive_circuits`

```python
class PredictiveCircuitBank(nn.Module):
    def __init__(self, config: SDNCConfig)
```

#### Attributs

| Attribut | Type | Description |
|----------|------|-------------|
| `circuit` | `PredictiveCircuit` | Circuit partagé (poids communs) |
| `proc_states` | `Tensor (N, proc_state_size)` | États processing par circuit |
| `pred_states` | `Tensor (N, pred_state_size)` | États prediction par circuit |
| `inter_circuit_weights` | `Tensor (N, N)` | Matrice de connexions inter-circuits |
| `prediction_error_history` | `Tensor (N,)` | Historique des erreurs |
| `activation_history` | `Tensor (N,)` | Compteur d'activations |

#### Méthodes

##### `forward_token_choice(x, indices, weights) → dict`

Forward pass avec routage TokenChoice.

##### `forward_expert_choice(x, routing_info, top_weights) → dict`

Forward pass avec routage ExpertChoice.

##### `reset_states()`

Reset tous les états des circuits (proc + pred).

---

### 2.3 `GlobalState`

**Module** : `sdnc.core.global_state`

```python
class GlobalState(nn.Module):
    def __init__(self, config: SDNCConfig)
```

| Méthode | Signature | Description |
|---------|-----------|-------------|
| `read` | `(x: Tensor) → Tensor` | Lecture par attention |
| `write` | `(x: Tensor) → None` | Écriture gated |
| `apply_decay` | `() → None` | Oubli graduel (×0.995) |
| `get_state_summary` | `() → dict` | Diagnostic |
| `reset` | `() → None` | Remise à zéro complète |

---

### 2.4 `TemporalStream`

**Module** : `sdnc.core.temporal_stream`

```python
class TemporalStream(nn.Module):
    def __init__(self, config: SDNCConfig)
```

| Méthode | Signature | Description |
|---------|-----------|-------------|
| `push` | `(x: Tensor) → None` | Ajoute au buffer circulaire |
| `get_history` | `() → Tensor` | Historique ordonné (old→new) |
| `get_context` | `(x: Tensor) → Tensor` | Fusion input + historique |
| `get_stream_stats` | `() → dict` | Statistiques du buffer |
| `reset` | `() → None` | Vide le buffer |

---

### 2.5 `SparseRouter`

**Module** : `sdnc.core.sparse_router`

```python
class SparseRouter(nn.Module):
    def __init__(self, input_dim, n_circuits, k=1, noise_std=0.1)
```

| Méthode | Signature | Description |
|---------|-----------|-------------|
| `forward` | `(x: Tensor) → (indices, weights)` | Top-k routing |
| `entropy_loss` | `(x: Tensor) → Tensor` | Negative entropy loss |
| `update_bias` | `(gamma: float) → None` | Correction de biais |
| `sparsity_ratio` | `→ float` | k / n_circuits |

### 2.6 `ExpertChoiceRouter`

```python
class ExpertChoiceRouter(nn.Module):
    def __init__(self, input_dim, n_circuits, capacity_factor=1.25)
```

| Méthode | Signature | Description |
|---------|-----------|-------------|
| `forward` | `(x: Tensor) → (routing_info, top_weights, token_counts)` | Expert-choice routing |
| `sparsity_ratio` | `→ float` | capacity_factor / n_circuits |

---

### 2.7 Fonctions Hebbian

**Module** : `sdnc.core.hebbian_update`

| Fonction | Signature | Description |
|----------|-----------|-------------|
| `oja_update` | `(weight, pre, post, lr, mask) → Tensor` | Règle d'Oja stabilisée |
| `apply_hebbian_to_cfc` | `(cfc_module, input, hidden, lr) → None` | Oja couche par couche |
| `apply_hebbian_to_circuit` | `(circuit, input, output, lr) → None` | Oja sur modules linéaires |
| `synaptic_pruning` | `(module, threshold) → None` | Élagage synaptique |
| `strengthen_co_active_circuits` | `(co_act, weights, lr, decay) → Tensor` | Inter-circuit Hebbian |

---

### 2.8 `EpisodicMemory`

**Module** : `sdnc.memory.episodic_memory`

```python
class EpisodicMemory:
    def __init__(self, embedding_dim: int, max_memories: int = 10000)
```

| Méthode | Signature | Description |
|---------|-----------|-------------|
| `store` | `(embedding, label, activation_pattern) → None` | Stocker un épisode |
| `retrieve` | `(query, top_k=5) → list[Episode]` | Récupérer par similarité |
| `consolidate` | `(threshold=0.95) → None` | Fusionner les épisodes similaires |
| `__len__` | `→ int` | Nombre d'épisodes stockés |

---

## 3. Référence API — Brain-Hybrid

### 3.1 `BrainHybridModel`

**Module** : `brain_hybrid.model`

```python
class BrainHybridModel(nn.Module):
    def __init__(self, config: BrainConfig = None)
```

#### Attributs principaux

| Attribut | Type | Description |
|----------|------|-------------|
| `llm` | `QwenWrapper` | LLM gelé |
| `brain_modules` | `ModuleList[BrainModule]` | 4 modules CfC+SNN |
| `stdp_learners` | `list[STDPLearning]` | 4 learners STDP |
| `injection_gates` | `ModuleList[InjectionGate]` | 4 gates d'injection |
| `hook_manager` | `BrainHookManager` | Gestionnaire de hooks |
| `hippocampus` | `HippocampalMemory` | Mémoire épisodique SDM |
| `hippocampal_injector` | `HippocampalPrefixInjector` | Prefix tokens |
| `acc` | `ACCModule` | Cortex cingulaire antérieur |
| `pc` | `HierarchicalPC` | Codage prédictif 4 couches |
| `scheduler` | `StepScheduler` | Rythmes cérébraux |
| `distilled_input` | `DistilledInputLayer` | Fusion teacher-student |

---

### 3.2 `BrainModule`

**Module** : `brain_hybrid.core.brain_module`

```python
class BrainModule(nn.Module):
    def __init__(self, input_dim=2560, hidden_dim=64, device=None)
```

| Attribut | Type | Description |
|----------|------|-------------|
| `input_proj` | `Linear(2560, 64)` | Projection d'entrée |
| `cfc` | `CfC(64, NCP)` | Réseau liquide |
| `lif` | `snn.Leaky` | Neurone LIF |
| `output_proj` | `Linear(64, 2560)` | Projection de sortie |
| `cfc_state` | `Tensor` | État persistant CfC |
| `mem` | `Tensor` | Potentiel membranaire SNN |
| `pre_trace` | `Tensor` | Trace pré-synaptique |
| `post_trace` | `Tensor` | Trace post-synaptique |

| Méthode | Retour | Description |
|---------|--------|-------------|
| `forward(x)` | `(prediction, spikes)` | Prédiction + spikes |

---

### 3.3 `STDPLearning`

**Module** : `brain_hybrid.core.stdp`

```python
class STDPLearning:
    def __init__(self, lr_plus=0.01, lr_minus=0.01)
```

| Méthode | Description |
|---------|-------------|
| `update_dopamine(error, threshold)` | Met à jour le signal dopamine |
| `apply(weight, pre_trace, post_trace)` | Applique STDP modulé par dopamine |

| Attribut | Type | Description |
|----------|------|-------------|
| `dopamine_signal` | `float` | Signal dopamine courant [0, 1] |
| `lr_plus` | `float` | Taux LTP |
| `lr_minus` | `float` | Taux LTD |

---

### 3.4 `ACCModule`

**Module** : `brain_hybrid.core.acc`

```python
class ACCModule(nn.Module):
    def __init__(self, n_modules=4, conflict_threshold=0.7,
                 exploration_threshold=0.4, lr=0.001, device=None)
```

| Méthode | Retour | Description |
|---------|--------|-------------|
| `forward(errors, dopamines, entropy)` | `ACCOutput` | Calcul du conflit |
| `trend()` | `str` | "rising", "falling", ou "stable" |
| `save_state()` | `dict` | Sauvegarde |
| `load_state(state)` | `None` | Restauration |

```python
@dataclass
class ACCOutput:
    conflict_score: float          # [0, 1]
    is_conflict: bool              # score > 0.7
    is_exploration: bool           # 0.4 < score ≤ 0.7
    conflict_prompt_fragment: str  # Fragment de prompt si conflit
```

---

### 3.5 `HippocampalMemory`

**Module** : `brain_hybrid.memory.hippocampus`

```python
class HippocampalMemory:
    def __init__(self, address_dim=256, content_dim=2560,
                 n_locations=10000, activation_radius=115)
```

| Méthode | Signature | Description |
|---------|-----------|-------------|
| `write` | `(embedding, content, metadata) → None` | Écriture distribuée |
| `read` | `(embedding) → Tensor` | Lecture par reconstruction |
| `consolidate` | `(threshold=5) → None` | Élagage pendant le sommeil |
| `stats` | `() → dict` | Statistiques mémoire |

---

### 3.6 `StepScheduler`

**Module** : `brain_hybrid.utils.step_scheduler`

```python
class StepScheduler:
    def __init__(self, snn_freq=1, cfc_freq=10, hippocampus_freq=6,
                 qwen_freq=100, sleep_freq=5000, arousal_boost_steps=50)
```

| Méthode | Signature | Description |
|---------|-----------|-------------|
| `step` | `() → None` | Incrémente le compteur |
| `should_update` | `(module_name, step=None) → bool` | Le module doit-il s'activer ? |
| `get_phase` | `() → str` | Phase cérébrale |
| `arousal_boost` | `() → None` | Double les fréquences |
| `save_state` / `load_state` | | Sérialisation |

---

## 5. SDNCConfig — Paramètres complets

**Fichier** : `sdnc/config.py`

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| **Circuit Bank** | | | |
| `n_circuits` | `int` | 1000 | Nombre total de circuits |
| `circuit_dim` | `int` | 8 | Dimension interne (legacy) |
| `circuit_output_dim` | `int` | 32 | Dimension de sortie par circuit |
| `circuit_repr_dim` | `int` | 128 | Dimension de l'espace de classification |
| **Input** | | | |
| `input_dim` | `int` | 512 | Dimension des features CLIP |
| **Routage** | | | |
| `sparsity_k` | `int` | 3 | Top-k circuits par input |
| `router_noise_std` | `float` | 0.1 | Bruit d'exploration du routeur |
| `router_bias_gamma` | `float` | 0.01 | Correction de biais du routeur |
| `use_expert_choice` | `bool` | True | Mode ExpertChoice vs TokenChoice |
| `expert_capacity_factor` | `float` | 1.25 | Facteur de capacité ExpertChoice |
| **Inter-circuit** | | | |
| `inter_circuit_hebbian_lr` | `float` | 1e-3 | LR Hebbian inter-circuits |
| `inter_circuit_decay` | `float` | 0.999 | Oubli inter-circuits |
| `inter_circuit_prune_threshold` | `float` | 1e-4 | Seuil d'élagage |
| **Hebbian** | | | |
| `hebbian_lr` | `float` | 1e-4 | LR Hebbian intra-circuit |
| `pruning_threshold` | `float` | 1e-5 | Seuil d'élagage synaptique |
| **NCP Wiring** | | | |
| `ncp_inter_neurons` | `int` | 48 | Neurones inter du NCP |
| `ncp_command_neurons` | `int` | 32 | Neurones de commande |
| `ncp_sensory_fanout` | `int` | 6 | Connexions sensory → inter |
| `ncp_inter_fanout` | `int` | 4 | Connexions inter → command |
| `ncp_recurrent_command_synapses` | `int` | 4 | Récurrence command |
| `ncp_motor_fanin` | `int` | 8 | Connexions command → motor |
| **Temporal** | | | |
| `integrator_state_dim` | `int` | 64 | Dimension de l'intégrateur CfC |
| **Predictive Coding** | | | |
| `prediction_error_weight` | `float` | 1.0 | Poids de l'erreur dans la loss |
| `prediction_hebbian_lr` | `float` | 5e-4 | LR Hebbian piloté par l'erreur |
| **Global State** | | | |
| `global_state_slots` | `int` | 8 | Nombre de slots mémoire |
| `global_state_decay` | `float` | 0.995 | Taux d'oubli graduel |
| **Temporal Stream** | | | |
| `temporal_history_len` | `int` | 16 | Taille du buffer circulaire |
| **Mémoire** | | | |
| `memory_capacity` | `int` | 10000 | Capacité max de mémoire épisodique |
| `salience_threshold` | `float` | 0.7 | Seuil de saillance |
| `working_memory_decay` | `float` | 0.99 | Oubli mémoire de travail |
| **Encodeurs** | | | |
| `clip_model_name` | `str` | "ViT-B-32" | Modèle CLIP |
| `clip_pretrained` | `str` | "openai" | Poids pré-entraînés |
| `whisper_model` | `str` | "tiny" | Modèle Whisper |
| `whisper_dim` | `int` | 384 | Dimension Whisper |
| `signal_input_dim` | `int` | 8 | Dimension des signaux bruts |
| `signal_hidden_dim` | `int` | 64 | Dimension cachée du signal encoder |
| **Entraînement** | | | |
| `n_way` | `int` | 5 | Nombre de classes par épisode |
| `k_shot` | `int` | 1 | Exemples par classe |
| `query_per_class` | `int` | 15 | Requêtes par classe |
| `train_episodes` | `int` | 1000 | Épisodes d'entraînement |
| `eval_episodes` | `int` | 600 | Épisodes d'évaluation |
| **Device** | | | |
| `device` | `str` | "auto" | GPU automatique |

**Propriétés calculées** :
- `max_active_circuits` : `max(1, sparsity_k)` = 3
- `sparsity_ratio` : `max_active_circuits / n_circuits` = 0.003

---

## 6. BrainConfig — Paramètres complets

**Fichier** : `brain_hybrid/config.py`

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| **LLM** | | | |
| `model_name` | `str` | "Qwen/Qwen3-VL-8B-Instruct" | Modèle Qwen |
| `llm_hidden_dim` | `int` | 4096 | Dimension cachée |
| `intercept_layers` | `List[int]` | [8, 16, 24, 32] | Couches d'intercept |
| **Modules** | | | |
| `module_hidden_dim` | `int` | 64 | Dimension interne CfC |
| `n_modules` | `int` | 4 | Nombre de modules |
| **STDP** | | | |
| `stdp_lr_plus` | `float` | 0.01 | Taux LTP |
| `stdp_lr_minus` | `float` | 0.01 | Taux LTD |
| `dopamine_threshold` | `float` | 1.5 | Seuil de saillance dopamine |
| **Hippocampe** | | | |
| `memory_address_dim` | `int` | 256 | Dimension des adresses SDM |
| `memory_n_locations` | `int` | 10000 | Nombre de locations mémoire |
| `salience_threshold` | `float` | 0.3 | Seuil de saillance pour mémorisation |
| **État global** | | | |
| `state_dim` | `int` | 512 | Dimension de l'état global |
| **Quantization** | | | |
| `use_quantization` | `bool` | True | Q8 bitsandbytes |
| **Injection** | | | |
| `injection_enabled` | `bool` | True | Activer l'injection CfC → Qwen |
| `injection_gate_rank` | `int` | 16 | Rang low-rank |
| `injection_init_alpha` | `float` | 0.001 | Alpha initial |
| `injection_max_alpha` | `float` | 0.1 | Alpha maximum |
| `injection_n_prefix_tokens` | `int` | 4 | Tokens virtuels hippocampe |
| **ACC** | | | |
| `acc_conflict_threshold` | `float` | 0.7 | Seuil de conflit |
| `acc_exploration_threshold` | `float` | 0.4 | Seuil d'exploration |
| `acc_lr` | `float` | 0.001 | LR Hebbian ACC |
| **Scheduler** | | | |
| `snn_freq` | `int` | 1 | Fréquence SNN |
| `cfc_freq` | `int` | 10 | Fréquence CfC |
| `hippocampus_freq` | `int` | 6 | Fréquence hippocampe (thêta) |
| `qwen_freq` | `int` | 100 | Fréquence Qwen enrichi |
| `sleep_freq` | `int` | 5000 | Fréquence consolidation sommeil |
| `arousal_boost_steps` | `int` | 50 | Durée du boost d'arousal |
| **Predictive Coding** | | | |
| `pc_lr` | `float` | 0.001 | LR codage prédictif |
| **Sleep** | | | |
| `sleep_replay_count` | `int` | 10 | Épisodes rejoués pendant sommeil |
| **Distillation** | | | |
| `bridge_checkpoint_path` | `str` | "" | Chemin vers bridge sauvegardé |
| `use_distilled_input` | `bool` | True | Activer la fusion teacher |
| `distilled_input_alpha_init` | `float` | 0.8 | Alpha initial (student domine) |

---

## 8. Tests

### 8.1 Lancer les tests

```bash
# Tous les tests
pytest tests/ -v

# Un fichier spécifique
pytest tests/test_model.py -v

# Un test spécifique
pytest tests/test_stdp.py::test_dopamine_update -v

# Avec couverture
pytest tests/ -v --cov=sdnc --cov=brain_hybrid
```

### 8.2 Matrice de tests

| Fichier | Module testé | Tests clés |
|---------|-------------|------------|
| `test_model.py` | SDNCModel | forward, learn, recognize, reset |
| `test_circuit_bank.py` | CircuitBank | forward, states, inter-wiring |
| `test_sparse_router.py` | SparseRouter, ExpertChoice | routing, bias, entropy |
| `test_hebbian.py` | oja_update, pruning | update, mask, stability |
| `test_brain_module.py` | BrainModule | forward, state persistence |
| `test_stdp.py` | STDPLearning | LTP, LTD, dopamine |
| `test_injection.py` | InjectionGate, HookManager | delta, hook lifecycle |
| `test_hippocampus.py` | HippocampalMemory | write, read, consolidate |
| `test_pc.py` | HierarchicalPC | forward, errors, update |
| `test_acc.py` | ACCModule | conflict, exploration, trend |
| `test_step_scheduler.py` | StepScheduler | frequencies, arousal, phases |
| `test_distilled_input.py` | DistilledInputLayer | fusion, alpha, gate |
| `test_micro_circuit.py` | MicroCircuit (legacy) | backward compat |
| `test_pipeline.py` | Full pipeline | bridge, distillation |

---

## 9. Glossaire

| Terme | Définition |
|-------|-----------|
| **ACC** | Anterior Cingulate Cortex — module de détection de conflit (Botvinick et al., 2001) |
| **Arousal** | État d'éveil augmenté — double les fréquences des modules pendant N steps |
| **bfloat16** | Brain floating-point 16-bit — format numérique optimal pour GPU Ampere+ |
| **Bottleneck** | Couche de compression dans le ProjectionBridge (5120 → 1024 → 2560) |
| **CfC** | Continuous-time Flow Cell — réseau neuronal liquide à dynamiques continues (Hasani et al., 2021) |
| **Circuit** | Unité de traitement SDNC avec double tête (prediction + processing) |
| **Circuit repr** | Représentation dans l'espace de classification (128-dim, L2-normalisé) |
| **Co-activation** | Deux circuits activés simultanément — renforce leur connexion |
| **Consolidation** | Processus de tri des mémoires (élagage des traces faibles pendant le sommeil) |
| **Dopamine** | Signal de saillance modulant l'amplitude de l'apprentissage STDP |
| **ExpertChoice** | Mode de routage où chaque circuit choisit ses inputs (équilibre structurel) |
| **Free Energy Principle** | Théorie de Friston : le cerveau minimise la surprise en prédisant ses entrées |
| **Gate (injection)** | Module low-rank contrôlant l'influence du CfC sur les hidden states Qwen |
| **Global State** | Mémoire de travail persistante partagée entre tous les circuits |
| **Hamming distance** | Nombre de bits différents entre deux adresses binaires (pour SDM) |
| **Hebbian** | "Neurons that fire together wire together" — apprentissage local sans backprop |
| **Hippocampe** | Mémoire épisodique persistante basée sur la Sparse Distributed Memory |
| **Hook (PyTorch)** | Fonction callback enregistrée sur un module pour intercepter les activations |
| **LIF** | Leaky Integrate-and-Fire — modèle de neurone à spikes |
| **LTD** | Long-Term Depression — affaiblissement synaptique |
| **LTP** | Long-Term Potentiation — renforcement synaptique |
| **NCP** | Neural Circuit Policy — câblage sparse inspiré de C. elegans |
| **Oja's rule** | Version stabilisée de la règle de Hebb : Δw = lr × post × (pre - post·w) |
| **Prediction error** | Différence entre la prédiction et l'observation réelle (= signal d'apprentissage) |
| **Predictive coding** | Architecture où chaque couche prédit l'entrée de la couche inférieure |
| **Precision** | 1/variance(error) — confiance dans les prédictions (codage prédictif) |
| **ProjectionBridge** | Seul module avec backprop — traduit teacher → student |
| **Prototypical network** | Classification par distance aux prototypes de classe (Snell et al., 2017) |
| **Ring buffer** | Buffer circulaire réutilisant les positions de manière cyclique |
| **SDM** | Sparse Distributed Memory — mémoire content-addressable de Kanerva |
| **SNN** | Spiking Neural Network — réseau communiquant par spikes discrets |
| **Sparsity mask** | Masque binaire du câblage NCP — seules les connexions câblées sont mises à jour |
| **STDP** | Spike-Timing-Dependent Plasticity — plasticité dépendante du timing des spikes |
| **Teacher-Student** | Paradigme de distillation : un grand modèle (teacher) enseigne à un petit (student) |
| **Temporal stream** | Buffer circulaire fournissant le contexte historique aux circuits |
| **TokenChoice** | Mode de routage où chaque input choisit ses top-k circuits |

---

## 10. Références scientifiques

### Fondements théoriques

1. **Friston, K.** (2010). *The free-energy principle: a unified brain theory?* Nature Reviews Neuroscience, 11(2), 127-138.
2. **Rao, R. P., & Ballard, D. H.** (1999). *Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects.* Nature Neuroscience, 2(1), 79-87.
3. **Bi, G., & Poo, M.** (1998). *Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type.* Journal of Neuroscience, 18(24), 10464-10472.
4. **Botvinick, M. M., et al.** (2001). *Conflict monitoring and cognitive control.* Psychological Review, 108(3), 624-652.
5. **Kanerva, P.** (1988). *Sparse Distributed Memory.* MIT Press.
6. **Hebb, D. O.** (1949). *The Organization of Behavior.* Wiley.

### Architectures et méthodes

7. **Hasani, R., et al.** (2021). *Liquid Time-constant Networks.* Proceedings of the AAAI Conference on Artificial Intelligence.
8. **Hasani, R., et al.** (2022). *Closed-form continuous-time neural networks.* Nature Machine Intelligence, 4(11), 992-1003.
9. **Snell, J., Swersky, K., & Zemel, R.** (2017). *Prototypical networks for few-shot learning.* Advances in Neural Information Processing Systems.
10. **Fedus, W., Zoph, B., & Shazeer, N.** (2022). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR.
11. **Radford, A., et al.** (2021). *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML.
12. **Oja, E.** (1982). *Simplified neuron model as a principal component analyzer.* Journal of Mathematical Biology, 15(3), 267-273.

### Modèles de fondation utilisés

13. **Qwen Team** (2025). *Qwen3 Technical Report.* Alibaba Group.
14. **Radford, A., et al.** (2022). *Robust Speech Recognition via Large-Scale Weak Supervision (Whisper).* OpenAI.
