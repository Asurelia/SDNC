# Partie 4 — Pipeline de Distillation & Entraînement

## Table des matières

1. [Pipeline de distillation teacher → student](#1-pipeline-de-distillation)
2. [ProjectionBridge — Le seul module avec backprop](#2-projectionbridge)
3. [DualModelConfig — Configuration teacher-student](#3-dualmodelconfig)
4. [Corpus de distillation](#4-corpus-de-distillation)
5. [Phases d'entraînement SDNC v2](#5-phases-dentrainement-sdnc-v2)
6. [Entraînement Brain-Hybrid](#6-entrainement-brain-hybrid)
7. [Études d'ablation](#7-etudes-dablation)
8. [Scripts d'entraînement](#8-scripts-dentrainement)
9. [Notebooks Colab](#9-notebooks-colab)

---

## 1. Pipeline de distillation teacher → student

### 1.1 Concept

Le pipeline de distillation transfère les connaissances d'un modèle **teacher** massif (Qwen3-VL-32B, 35 milliards de paramètres) vers un **student** compact (Qwen3.5-4B, 4 milliards de paramètres) augmenté par le cerveau Brain-Hybrid.

```
┌────────────────────────────────┐     ┌────────────────────────────────┐
│  TEACHER (Qwen3-VL-32B)       │     │  STUDENT (Qwen3.5-4B)         │
│  35B params, gelé, en RAM CPU  │     │  4B params, gelé, en VRAM GPU  │
│                                │     │  + Brain-Hybrid (adaptatif)    │
│  Couches [16, 32, 48, 64]      │     │  Couches [8, 16, 24, 32]       │
│  hidden_dim = 5120             │     │  hidden_dim = 2560             │
│                                │     │                                │
│  Extrait hidden states          │     │  Reçoit hidden states projetés │
└───────────────┬────────────────┘     └───────────────┬────────────────┘
                │                                       │
                │    ┌────────────────────┐             │
                └───►│  ProjectionBridge    │◄───────────┘
                     │  5120 → 1024 → 2560  │
                     │  SEUL module avec     │
                     │  backprop             │
                     └────────────────────┘
```

### 1.2 Flux par épisode de distillation

```python
# 1. Teacher extrait ses représentations (CPU, bfloat16)
teacher_reps = teacher.get_layer_representations(prompt, layers=[16, 32, 48, 64])
soft_targets = teacher.forward(prompt).logits  # Soft targets pour KL

# 2. ProjectionBridge projette dans l'espace student
projected_reps = [bridge(teacher_reps[i]) for i in range(4)]
# (1, S, 5120) → (1, S, 2560) par couche

# 3. Student brain forward avec représentations enrichies
student_reps = student.get_layer_representations(prompt, layers=[8, 16, 24, 32])
enriched, stats = distilled_input(student_reps, projected_reps)

# 4. Loss = MSE(projected, student_reps) + KL(student_logits, soft_targets)
loss_mse = MSE(projected_reps, [s.detach() for s in student_reps])
loss_kl = KL(student_logits / T, soft_targets / T) * T²
loss = alpha * loss_mse + (1 - alpha) * loss_kl

# 5. Backprop UNIQUEMENT sur ProjectionBridge
loss.backward()
optimizer.step()  # AdamW, lr=5e-5

# 6. Student brain apprend via règles LOCALES (pas de backprop)
hierarchical_pc.update_all(errors, dopamine_signals, stdp_learners)
# STDP + dopamine → poids CfC
# PC → modulation précision
# Hippocampe → mémoire épisodique
```

### 1.3 Séparation stricte des apprentissages

| Module | Méthode d'apprentissage | Gradient ? |
|--------|------------------------|-----------|
| Teacher (Qwen-32B) | Aucun (gelé) | Non |
| Student (Qwen-4B) | Aucun (gelé) | Non |
| ProjectionBridge | Adam backprop | Oui |
| BrainModules (CfC) | STDP + dopamine | Non |
| HierarchicalPC | Codage prédictif local | Non |
| Hippocampe | Écriture additive | Non |
| ACC | Hebbian local | Non |
| InjectionGates | Gradient via hooks | Oui (partiel) |
| DistilledInputLayer | Gradient via fusion | Oui (partiel) |

---

## 2. ProjectionBridge

**Fichier** : `pipeline/distillation_engine.py`

### 2.1 Architecture

```
Teacher representation (batch, seq_len, 5120)
                │
                ▼ cast float32 (stabilité gradient)
                │
    ┌────────────────────┐
    │  down: Linear(5120, 1024)  │  Compression → espace latent
    └────────┬───────────┘
             │
    ┌────────▼───────────┐
    │  mid_norm: LayerNorm(1024)  │  Normalisation intermédiaire
    └────────┬───────────┘
             │
    ┌────────▼───────────┐
    │  act: GELU          │  Activation non-linéaire
    └────────┬───────────┘
             │
    ┌────────▼───────────┐
    │  up: Linear(1024, 2560)  │  Expansion → espace student
    └────────┬───────────┘
             │
    ┌────────▼───────────┐
    │  norm: LayerNorm(2560)  │  Normalisation de sortie
    └────────┬───────────┘
             │
             ▼ cast back to original dtype
             │
    Projected representation (batch, seq_len, 2560)
```

### 2.2 Pourquoi un bottleneck ?

Le bottleneck (5120 → **1024** → 2560) force une **compression informationnelle** :

- **5120-dim** : espace teacher riche mais redondant
- **1024-dim** : espace latent compressé — seule l'information essentielle survit
- **2560-dim** : espace student — représentation utilisable

Sans bottleneck, le bridge pourrait simplement copier les dimensions. Le bottleneck force l'abstraction.

### 2.3 Cast float32

```python
def forward(self, teacher_repr):
    orig_dtype = teacher_repr.dtype
    h = self.down(teacher_repr.float())  # ← float32 pour les gradients
    # ... opérations ...
    return self.norm(h).to(orig_dtype)   # ← retour au dtype original
```

Le cast en float32 assure la **stabilité des gradients** — les opérations en bfloat16 peuvent perdre en précision pendant le backprop.

---

## 3. DualModelConfig

**Fichier** : `pipeline/dual_model_config.py`

### 3.1 Paramètres complets

```python
@dataclass
class DualModelConfig:
    # ── Teacher ─────────────────────────────────────
    teacher_model: str = "Qwen/Qwen3-VL-32B-Instruct"
    teacher_hidden: int = 5120          # Dimension cachée du teacher
    teacher_layers: List[int] = [16, 32, 48, 64]  # Points d'intercept

    # ── Student ─────────────────────────────────────
    student_model: str = "Qwen/Qwen3.5-4B"
    student_hidden: int = 2560          # Dimension cachée du student
    student_layers: List[int] = [8, 16, 24, 32]

    # ── ProjectionBridge ────────────────────────────
    distill_projection_dim: int = 1024  # Taille du bottleneck
    distill_temperature: float = 2.0    # Température pour soft targets
    distill_alpha: float = 0.7          # α×MSE + (1-α)×KL

    # ── Sleep distillation ──────────────────────────
    sleep_replay_count: int = 50        # Épisodes rejoués pendant le sommeil
    sleep_distill_lr: float = 5e-5      # LR AdamW pour le bridge

    # ── Hardware ────────────────────────────────────
    teacher_device: str = "cpu"         # Teacher 35B en RAM système
    student_device: str = "cuda"        # Student 4B en VRAM
    teacher_dtype: str = "bfloat16"     # BF16 natif

    # ── Safety ──────────────────────────────────────
    ram_safety_margin_gb: float = 3.0   # Arrêt propre si RAM < 3 GB libre
    log_interval: int = 10              # Log toutes les N steps

    # ── Cloud sync ──────────────────────────────────
    gcs_bucket: str = "sdnc-models"
    gcs_model_prefix: str = "checkpoints"
    auto_push_every_n_distillations: int = 1
```

### 3.2 Placement hardware

```
┌──────────────────────────────────────────────────────┐
│  RAM SYSTÈME (~80 GB minimum)                        │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  Teacher Qwen3-VL-32B (bfloat16) → ~65 GB   │   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────┐   │
│  │  ProjectionBridge (quelques MB)               │   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────┐   │
│  │  Marge de sécurité ≥ 3 GB                    │   │
│  └──────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  VRAM GPU (~16 GB minimum)                           │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  Student Qwen3.5-4B (bfloat16) → ~8 GB      │   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────┐   │
│  │  Brain-Hybrid modules → ~1 GB                │   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────┐   │
│  │  Activations, gradients → ~4 GB              │   │
│  └──────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

---

## 4. Corpus de distillation

**Fichier** : `pipeline/distillation_engine.py` (lignes 68+)

Le corpus contient **300 prompts** couvrant **11 catégories** :

| Catégorie | N prompts | Exemples |
|-----------|-----------|----------|
| Raisonnement logique | 20 | Syllogismes, contraposées, paradoxes |
| Mathématiques | 20 | Probabilités, limites, théorèmes |
| Sciences | 30 | Physique, biologie, chimie |
| Programmation | 30 | Algorithmes, structures de données, Python |
| Créativité | 20 | Métaphores, histoires, poèmes |
| Connaissances générales | 30 | Histoire, géographie, culture |
| Analyse de texte | 20 | Résumés, thèmes, registres |
| Éthique et philosophie | 20 | Dilemmes moraux, libre arbitre |
| Communication | 20 | Emails, présentations, persuasion |
| Multilingue | 20 | Traduction, nuances linguistiques |
| Méta-cognition | 20 | Réflexion sur le raisonnement |

Le corpus est **en français**, diversifié pour couvrir l'ensemble des capacités cognitives du système.

---

## 5. Phases d'entraînement SDNC v2

### Phase 1 — Vision (Few-shot CIFAR-10)

**Script** : `run_train.py`, `run_train_v2.py`
**Module** : `sdnc/train/phase1_vision.py`

```
Objectif : Classification few-shot sur CIFAR-10
Config  : 5-way 1-shot (5 classes, 1 exemple par classe)
Metric  : Accuracy sur 600 épisodes d'évaluation

Pipeline :
  1. Échantillonner un épisode (5 classes × [1 support + 15 query])
  2. Préprocesser avec CLIP transforms
  3. model.learn(labels=support_labels, images=support_images)
  4. logits = model.recognize(support_labels, support_images, query_images)
  5. accuracy = (logits.argmax(-1) == query_labels).float().mean()
```

### Phase 2 — Multimodal (Vision + Texte + Audio)

**Script** : `run_phase2.py`
**Module** : `sdnc/train/phase2_multimodal.py`

```
Objectif : Alignement cross-modal dans l'espace circuit
Modalités : Images CIFAR-10 + descriptions textuelles + audio synthétique
Router  : ExpertChoice (garantit l'équilibre)

Nouveau :
  - Temporal stream actif (historique cross-modal)
  - Global state accumule contexte multimodal
  - Inter-circuit wiring émerge entre modalités
```

### Phase 3 — Quadmodal (+ Signaux)

**Script** : `run_phase3.py`
**Module** : `sdnc/train/phase3_quadmodal.py`

```
Objectif : Intégration de 4 modalités simultanées
Modalités : Vision + Texte + Audio + Signaux numériques
Complexité : Les 4 encodeurs alimentent le même espace circuit

Nouveau :
  - Signal encoder entièrement entraînable
  - Intégrateur temporal synchronise les 4 flux
  - Mémoire épisodique stocke des représentations cross-modales
```

### Progression des phases

```
Phase 1        Phase 2          Phase 3
Vision only    + Texte + Audio   + Signaux
   │                │                │
   ▼                ▼                ▼
CLIP ViT    + CLIP Text       + Whisper       + Signal MLP
   │          + Audio proj      + Signal proj    (trainable)
   ▼                ▼                ▼
TokenChoice    ExpertChoice     ExpertChoice
   │                │                │
   ▼                ▼                ▼
1000 circuits  1000 circuits    1000 circuits
(monoétat)    (multi-état)     (multi-état enrichi)
   │                │                │
   ▼                ▼                ▼
Prototypical  Prototypical     Prototypical
loss seul     + pred error     + cross-modal alignment
```

---

## 6. Entraînement Brain-Hybrid

### 6.1 Flux d'entraînement

```python
# Initialisation
model = BrainHybridModel(BrainConfig())

# Boucle d'entraînement
for step, prompt in enumerate(corpus):
    # 1. Rythmes
    model.scheduler.step()
    
    # 2. Extraire hidden states
    layer_reps = model.llm.get_layer_representations(prompt, layers=[8, 16, 24, 32])
    
    # 3. PC forward → erreurs
    errors = model.pc.forward(layer_reps)
    
    # 4. BrainModules → prédictions
    for i, module in enumerate(model.brain_modules):
        if model.scheduler.should_update("cfc"):
            prediction, spikes = module(layer_reps[i])
            error = prediction - layer_reps[min(i+1, 3)]
            
            # Dopamine
            model.stdp_learners[i].update_dopamine(
                error.abs().mean().item(),
                threshold=model.config.dopamine_threshold,
            )
            
            # STDP
            model.stdp_learners[i].apply(
                list(module.cfc.parameters())[0],
                module.pre_trace,
                module.post_trace,
            )
    
    # 5. PC update
    dopamine_signals = [s.dopamine_signal for s in model.stdp_learners]
    model.pc.update_all(errors, dopamine_signals, model.stdp_learners)
    
    # 6. ACC
    acc_output = model.acc.forward(
        [e.abs().mean().item() for e in errors],
        dopamine_signals,
    )
    if acc_output.is_conflict:
        model.scheduler.arousal_boost()
    
    # 7. Hippocampe (conditionné par scheduler)
    if model.scheduler.should_update("hippocampus"):
        mean_error = sum(e.abs().mean() for e in errors) / len(errors)
        if mean_error > model.config.salience_threshold:
            model.hippocampus.write(
                layer_reps[-1].mean(dim=1).squeeze(),
                layer_reps[-1].mean(dim=1).squeeze(),
            )
    
    # 8. Sleep (rare)
    if model.scheduler.should_update("sleep"):
        model.hippocampus.consolidate()
```

### 6.2 Métriques de suivi

| Métrique | Source | Signification |
|----------|--------|---------------|
| `pc_errors[i]` | HierarchicalPC | Erreur de prédiction par couche |
| `precisions[i]` | HierarchicalPC | Confiance par couche (1/variance) |
| `dopamine_signals[i]` | STDP | Niveau de surprise par module |
| `acc.conflict_score` | ACC | Score de conflit global |
| `acc.trend()` | ACC | Tendance (rising/falling/stable) |
| `hippocampus.stats()` | SDM | Taux d'utilisation mémoire |
| `scheduler.get_phase()` | StepScheduler | Phase cérébrale (gamma/theta/idle) |
| `injection_gates[i].alpha` | InjectionGate | Force d'injection par couche |
| `distilled_input.get_teacher_influence()` | DistilledInput | Influence du teacher |

---

## 7. Études d'ablation

### 7.1 Ablation v3 — Circuit-centric

**Script** : `run_ablation_v3.py`

Compare les performances avec et sans composants clés :

| Condition | Description |
|-----------|-------------|
| **Full SDNC** | Tous les composants actifs |
| **CLIP baseline** | Encodeur CLIP seul (sans circuits) |
| **No temporal** | Sans TemporalStream |
| **No global state** | Sans GlobalState |
| **No prediction** | Sans tête de prédiction (processing seul) |
| **No Hebbian** | Sans apprentissage Hebbian |
| **No inter-wiring** | Sans connexions inter-circuits |

### 7.2 Ablation Brain-Hybrid

**Fichier** : `brain_hybrid/eval/ablation.py`

| Condition | Description |
|-----------|-------------|
| **Full model** | Cerveau complet |
| **No STDP** | Pas de plasticité synaptique |
| **No dopamine** | STDP sans modulation |
| **No ACC** | Pas de détection de conflit |
| **No hippocampus** | Pas de mémoire épisodique |
| **No injection** | CfC ne modifie pas Qwen |
| **No PC** | Pas de codage prédictif |
| **No scheduler** | Tous les modules à chaque step |

---

## 8. Scripts d'entraînement

### 8.1 Récapitulatif complet

| Script | Module | Phase | Description |
|--------|--------|-------|-------------|
| `run_train.py` | SDNC v2 | 1 | Entraînement vision basique |
| `run_train_v2.py` | SDNC v2 | 1 | V2 avec corrections |
| `run_phase2.py` | SDNC v2 | 2 | Multimodal (vision+text+audio) |
| `run_phase3.py` | SDNC v2 | 3 | Quadmodal (+signaux) |
| `run_phase2_hybrid.py` | Brain-Hybrid | 2 | Phase 2 avec cerveau |
| `run_ablation.py` | SDNC v2 | — | Ablation standard |
| `run_ablation_v2.py` | SDNC v2 | — | Ablation v2 |
| `run_ablation_v3.py` | SDNC v2 | — | Ablation circuit-centric |
| `run_quick_test.py` | — | — | Validation rapide |
| `check_gpu.py` | — | — | Diagnostic GPU |
| `check_gpu_minimal.py` | — | — | Check GPU minimal |
| `check_gpu_ops.py` | — | — | Test des opérations GPU |
| `debug_grads.py` | — | — | Visualisation des gradients |
| `inspect_cfc.py` | — | — | Inspection des CfC |

### 8.2 Exécution

```bash
# Phase 1 — Vision only
python run_train.py

# Phase 2 — Multimodal
python run_phase2.py

# Phase 3 — Quadmodal
python run_phase3.py

# Ablation
python run_ablation_v3.py

# Diagnostic
python check_gpu.py
python debug_grads.py
```

---

## 9. Notebooks Colab

### 9.1 Liste des notebooks

| Notebook | GPU recommandé | Contenu |
|----------|---------------|---------|
| `colab_sdnc.ipynb` | T4 (gratuit) | Phase 1-3 complète |
| `colab_a100_full.ipynb` | A100 (payant) | Pipeline complet avec distillation |
| `colab_brain_hybrid.ipynb` | T4/A100 | Démonstration Brain-Hybrid |
| `colab_pipeline.ipynb` | A100 | Pipeline teacher-student |

### 9.2 Workflow Colab typique

```python
# 1. Installation
!pip install -e ".[dev]"

# 2. Vérification GPU
import torch
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 3. Entraînement
from sdnc.train.phase1_vision import train_phase1
results = train_phase1()

# 4. Évaluation
print(f"Accuracy: {results['accuracy']:.1%}")
```

### 9.3 Cloud sync

Le pipeline inclut une synchronisation automatique vers GCS :

```python
# pipeline/cloud_sync.py
gcs_bucket = "sdnc-models"
gcs_prefix = "checkpoints"

# Sauvegarde automatique après chaque distillation
def save_checkpoint(bridge, step):
    path = f"bridge_step{step}.pt"
    torch.save(bridge.state_dict(), path)
    upload_to_gcs(path, f"{gcs_prefix}/{path}")
```
