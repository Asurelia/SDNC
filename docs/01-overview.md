# SDNC — Sparse Dynamic Neural Circuits

## Documentation Technique Complète

> **Version** : 0.2.0 — Mars 2026
> **Auteur** : Architecture conçue et documentée par l'équipe SDNC
> **Licence** : Projet de recherche

---

## 1. Introduction

### 1.1 Qu'est-ce que SDNC ?

**SDNC** (Sparse Dynamic Neural Circuits) est une architecture d'intelligence artificielle bio-inspirée qui combine des modèles de fondation gelés (Qwen, CLIP) avec des circuits neuronaux adaptatifs utilisant des mécanismes d'apprentissage biologiquement plausibles.

Le projet se compose de **deux systèmes complémentaires** :

| Système | Rôle | Analogie |
|---------|------|----------|
| **SDNC v2** | Perception multimodale + few-shot learning | Cortex sensoriel |
| **Brain-Hybrid** | Cognition + génération de texte + mémoire épisodique | Cerveau complet |

### 1.2 Problème résolu

Les architectures traditionnelles de deep learning souffrent de plusieurs limitations fondamentales :

1. **Backpropagation globale** — Biologiquement impossible, coûteux en calcul, incompatible avec l'apprentissage en continu
2. **Catastrophic forgetting** — Les réseaux oublient les tâches précédentes en apprenant de nouvelles
3. **Sample inefficiency** — Des milliers d'exemples nécessaires pour apprendre une catégorie
4. **Opacité** — Les réseaux denses sont des boîtes noires

SDNC répond à ces problèmes en s'inspirant directement du cerveau humain :

| Limitation | Solution SDNC |
|-----------|---------------|
| Backpropagation globale | Apprentissage local uniquement (Oja, STDP, codage prédictif) |
| Catastrophic forgetting | États persistants + mémoire épisodique (jamais reset) |
| Sample inefficiency | Few-shot via circuits prototypiques (5-way 1-shot) |
| Opacité | Circuits sparse + patterns d'activation lisibles |

### 1.3 Principes fondamentaux — Non négociables

Ces principes architecturaux sont **immuables** et guident toute décision de conception :

1. **Le Qwen ne s'entraîne jamais** — Tous ses poids sont gelés définitivement
2. **L'apprentissage est local** — Uniquement dans les modules CfC+SNN via STDP et règle d'Oja
3. **Pas de backpropagation globale** sur l'ensemble du système (sauf le ProjectionBridge)
4. **L'hippocampe ne se remet jamais à zéro** — Il s'enrichit en continu
5. **Les erreurs de prédiction** entre couches sont le **seul** signal d'apprentissage

---

## 2. Vision et inspiration biologique

### 2.1 L'analogie cerveau → architecture

SDNC reproduit fidèlement les mécanismes cérébraux connus :

```
┌──────────��──────────────────────────────────────────────────┐
│                    CERVEAU HUMAIN                            │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Cortex      │    │   Cortex      │    │  Hippocampe   │  │
│  │  préfrontal   │◄──►│  sensoriel    │◄──►│  (mémoire     │  │
│  │  (savoir)     │    │  (adaptation) │    │  épisodique)  │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                    │                    │          │
│  ┌──────▼───────────────────▼──────────────────▼───────┐   │
│  │            Plasticité synaptique (STDP)               │   │
│  │            Dopamine (saillance, récompense)           │   │
│  │            Rythmes cérébraux (gamma, thêta, sommeil)  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↕ mapping
┌─────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE SDNC                         │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Qwen/CLIP   │    │  CfC + SNN    │    │  SDM / Episodic│ │
│  │   (gelé)      │◄──►│  (adaptatif)  │◄──►│  Memory       │  │
│  │               │    │              │    │               │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                    │                    │          │
│  ┌──────▼───────────────────▼──────────────────▼───────┐   │
│  │  STDP + Oja (apprentissage local)                     │   │
│  │  Neuromodulation dopaminergique                        │   │
│  │  StepScheduler (gamma=1, thêta=6, sommeil=5000)       │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Mapping détaillé cerveau → code

| Structure cérébrale | Composant SDNC | Fichier | Fonction |
|---|---|---|---|
| Cortex préfrontal | Qwen3-4B / CLIP ViT-B/32 | `brain_hybrid/llm/qwen_wrapper.py`, `sdnc/encoders/` | Connaissance du monde, extraction de features |
| Cortex sensoriel adaptatif | 4 modules CfC+SNN | `brain_hybrid/core/brain_module.py` | Traitement adaptatif des représentations |
| Plasticité synaptique | STDP + Oja | `brain_hybrid/core/stdp.py`, `sdnc/core/hebbian_update.py` | Renforcement/affaiblissement des connexions |
| Dopamine | Neuromodulateur global | `brain_hybrid/core/stdp.py:update_dopamine()` | Modulation de l'apprentissage par la surprise |
| Hippocampe | Sparse Distributed Memory | `brain_hybrid/memory/hippocampus.py` | Mémoire épisodique persistante |
| Cortex cingulaire antérieur | ACC Module | `brain_hybrid/core/acc.py` | Détection de conflit, arousal |
| Rythmes cérébraux | StepScheduler | `brain_hybrid/utils/step_scheduler.py` | Oscillations gamma/thêta/sommeil |
| Codage prédictif | HierarchicalPC / PredictiveCircuits | `brain_hybrid/core/predictive_coding.py`, `sdnc/core/predictive_circuits.py` | Prédiction top-down + erreur |
| Consolidation nocturne | Sleep replay | `brain_hybrid/utils/step_scheduler.py` | Replay épisodique offline |

### 2.3 Fondements théoriques

L'architecture s'appuie sur des théories neuroscientifiques établies :

- **Free Energy Principle** (Karl Friston, 2010) — Le cerveau minimise la surprise en prédisant ses entrées. L'erreur de prédiction est LE signal d'apprentissage.
- **Predictive Coding** (Rao & Ballard, 1999) — Chaque niveau cortical prédit l'activité du niveau inférieur. Seules les erreurs remontent.
- **STDP** (Bi & Poo, 1998) — Les synapses se renforcent quand le neurone pré-synaptique s'active AVANT le post-synaptique.
- **Conflict Monitoring** (Botvinick et al., 2001) — L'ACC détecte les conflits entre signaux concurrents.
- **Sparse Distributed Memory** (Kanerva, 1988) — Mémoire content-addressable avec stockage distribué.
- **Liquid Neural Networks / CfC** (Hasani et al., 2021) — Réseaux à dynamiques continues, inspirés du connectome de C. elegans.

---

## 3. Architecture globale

### 3.1 Vue d'ensemble des deux systèmes

```
                    ┌─────────────────────────────────────┐
                    │         SDNC ECOSYSTEM               │
                    └─────────────────┬───────────────────┘
                                      │
               ┌──────────────────────┴──────────────────────┐
               │                                              │
    ┌──────────▼──────────┐                    ┌─────────────▼─────────────┐
    │     SDNC v2          │                    │     Brain-Hybrid           │
    │  Predictive Coding   │                    │  Full Brain Simulation     │
    │                      │                    │                           │
    │  • CLIP encoders     │                    │  • Qwen3-4B (gelé)        │
    │  • 1000 circuits     │                    │  • 4 modules CfC+SNN      │
    │  • Sparse routing    │                    │  • STDP + dopamine         │
    │  • Global state      │                    │  • ACC (conflit)           │
    │  • Temporal stream   │                    │  • Hippocampe (SDM)        │
    │  • Few-shot learning │                    │  • Injection CfC → Qwen   │
    │  • Episodic memory   │                    │  • Codage prédictif        │
    │                      │                    │  • Rythmes cérébraux       │
    │  Vision / Text /     │                    │                           │
    │  Audio / Signal      │                    │  Génération de texte       │
    │  → Classification    │                    │  enrichie par le cerveau   │
    └──────────┬──────────┘                    └────────────┬──────────────┘
               │                                            │
               └──────────────────────┬─────────────────────┘
                                      │
                         ┌────────────▼────────────┐
                         │   Distillation Pipeline  │
                         │                         │
                         │  Teacher (32B) → Bridge  │
                         │  → Student (4B) + Brain  │
                         │                         │
                         │  Seul module avec        │
                         │  backprop dans le système │
                         └─────────────────────────┘
```

### 3.2 Stack technologique

| Couche | Technologie | Version | Rôle |
|--------|------------|---------|------|
| **Runtime** | Python | ≥ 3.11 | Langage principal |
| **GPU Framework** | PyTorch | ≥ 2.2 | Tenseurs, autograd, CUDA/ROCm |
| **Modèles fondation** | HuggingFace Transformers | ≥ 4.45 | Qwen, CLIP |
| **Réseaux liquides** | ncps | ≥ 1.0 | CfC (Continuous-time Flow Cells) |
| **Réseaux à spikes** | snntorch | ≥ 0.9 | SNN (Spiking Neural Networks) |
| **Multi-GPU** | Accelerate | ≥ 0.27 | Device placement, quantization |
| **Vision-Language** | qwen-vl-utils | ≥ 0.0.8 | Utilitaires Qwen VL |
| **Encodeurs visuels** | open_clip | implicite | CLIP ViT-B/32 |
| **Quantization** | bitsandbytes | ≥ 0.43 | Q8 (Linux/CUDA uniquement) |

### 3.3 Structure du projet

```
SDNC-refactor-predictive-coding/
│
├── sdnc/                           # SDNC v2 — Circuits prédictifs
│   ├── config.py                   #   SDNCConfig (dataclass centrale)
│   ├── model.py                    #   SDNCModel (413 lignes)
│   ├── core/                       #   Composants fondamentaux
│   │   ├── predictive_circuits.py  #     Circuits à double tête (prédiction + traitement)
│   │   ├── global_state.py         #     État contextuel persistant
│   │   ├── temporal_stream.py      #     Buffer circulaire + attention temporelle
│   │   ├── sparse_router.py        #     Routage TokenChoice + ExpertChoice
│   │   ├── hebbian_update.py       #     Règle d'Oja + élagage synaptique
│   │   ├── integrator.py           #     Intégrateur temporel CfC
│   │   ├── circuit_bank.py         #     Banque de N circuits partagés
│   │   └── micro_circuit.py        #     Legacy (remplacé par predictive)
│   ├── encoders/                   #   Extracteurs de features gelés
│   │   ├── vision_encoder.py       #     CLIP ViT-B/32 → 512-dim
│   │   ├── text_encoder.py         #     CLIP text encoder → 512-dim
│   │   ├── audio_encoder.py        #     Whisper tiny → 384→512-dim
│   │   └── signal_encoder.py       #     Signaux bruts → 8→512-dim
│   ├── memory/                     #   Systèmes de mémoire
│   │   ├── episodic_memory.py      #     Stockage vectoriel d'épisodes
│   │   ├── consolidation.py        #     Détection de saillance
│   │   └── working_memory.py       #     Tampon court-terme
│   ├── train/                      #   Boucles d'entraînement
│   │   ├── phase1_vision.py        #     Few-shot vision (CIFAR-10)
│   │   ├── phase2_multimodal.py    #     Vision + texte + audio
│   │   └── phase3_quadmodal.py     #     4 modalités intégrées
│   ├── eval/                       #   Évaluation
│   │   └── one_shot_test.py        #     Protocole d'évaluation few-shot
│   └── utils/
│       └── device.py               #     Gestion CUDA/ROCm
│
├── brain_hybrid/                   # Brain-Hybrid — Cerveau artificiel complet
│   ├── config.py                   #   BrainConfig (dataclass)
│   ├── model.py                    #   BrainHybridModel (assemblage complet)
│   ├── chat.py                     #   Interface de chat
│   ├── ui.py                       #   Interface utilisateur
│   ├── core/                       #   Modules cérébraux
│   │   ├── brain_module.py         #     CfC + SNN hybride (64-dim)
│   │   ├── stdp.py                 #     STDP + modulation dopaminergique
│   │   ├── injection.py            #     Gates CfC → Qwen + prefix hippocampique
│   │   ├── predictive_coding.py    #     Codage prédictif hiérarchique (Friston)
│   │   ├── acc.py                  #     Cortex cingulaire antérieur (conflit)
│   │   └── distilled_input.py      #     Fusion teacher-student
│   ├── llm/
│   │   └── qwen_wrapper.py         #     Wrapper Qwen3-4B/8B gelé
│   ├── memory/
│   │   └── hippocampus.py          #     Sparse Distributed Memory (Kanerva)
│   ├── eval/                       #   Évaluation
│   │   ├── ablation.py             #     Études d'ablation
│   │   ├── benchmark.py            #     Benchmarks de performance
│   │   └── continual_test.py       #     Tests d'apprentissage continu
│   └── utils/
│       ├── device.py               #     Détection GPU/dtype
│       └── step_scheduler.py       #     Rythmes cérébraux
│
├── pipeline/                       # Pipeline de distillation
│   ├── distillation_engine.py      #   ProjectionBridge + corpus (300 prompts)
│   ├── full_loop.py                #   Boucle complète d'entraînement
│   ├── teacher_loader.py           #   Chargement du modèle teacher
│   ├── teacher_wrapper.py          #   Wrapper teacher
│   ├── local_runner.py             #   Runner local
│   ├── cloud_sync.py               #   Sync GCS/GitHub
│   └── dual_model_config.py        #   Config teacher-student
│
├── tests/                          # 14 fichiers de tests unitaires
│   ├── test_model.py               #   SDNCModel
│   ├── test_circuit_bank.py        #   CircuitBank + inter-wiring
│   ├── test_sparse_router.py       #   TokenChoice + ExpertChoice
│   ├── test_hebbian.py             #   Règle d'Oja
│   ├── test_brain_module.py        #   CfC + SNN
│   ├── test_stdp.py                #   STDP + dopamine
│   ├── test_injection.py           #   Gates d'injection
│   ├── test_hippocampus.py         #   SDM
│   ├── test_pc.py                  #   Codage prédictif
│   ├── test_acc.py                 #   ACC
│   ├── test_step_scheduler.py      #   Rythmes
│   ├── test_distilled_input.py     #   Fusion teacher-student
│   ├── test_micro_circuit.py       #   Legacy
│   └── test_pipeline.py            #   Pipeline complet
│
├── notebooks/                      # Jupyter notebooks
│   ├── colab_sdnc.ipynb            #   Phase 1-3 SDNC
│   ├── colab_a100_full.ipynb       #   Pipeline complet sur A100
│   ├── colab_brain_hybrid.ipynb    #   Brain-Hybrid demo
│   └── colab_pipeline.ipynb        #   Pipeline de distillation
│
├── scripts/
│   └── local_sync.sh               #   Synchronisation locale
│
├── run_train.py                    # Entraînement Phase 1
├── run_train_v2.py                 # Entraînement Phase 1 v2
├── run_phase2.py                   # Entraînement Phase 2
├── run_phase3.py                   # Entraînement Phase 3
├── run_phase2_hybrid.py            # Phase 2 Brain-Hybrid
├── run_ablation.py                 # Études d'ablation v1
├── run_ablation_v2.py              # Études d'ablation v2
├── run_ablation_v3.py              # Études d'ablation v3 (circuit-centric)
├── run_quick_test.py               # Test rapide
├── check_gpu*.py                   # Diagnostics GPU (3 variantes)
├── debug_grads.py                  # Debug des gradients
├── inspect_cfc.py                  # Inspection des CfC
│
├── AGENTS.md                       # Spécification architecturale (38 KB, français)
└── pyproject.toml                  # Métadonnées du projet
```

---

## 4. Hardware et compatibilité

### 4.1 GPU supportés

| GPU | API | Dtype optimal | Quantization | Statut |
|-----|-----|--------------|-------------|--------|
| NVIDIA Ampere+ (A100, RTX 3090, 4090) | CUDA | bfloat16 | Q8 (bitsandbytes) | Optimal |
| NVIDIA Turing (T4) | CUDA | float16 | Q8 | Supporté |
| AMD (RX 7900 XT, MI250) | ROCm | bfloat16 | Q8 (ROCm) | Supporté |
| CPU | — | float32 | Non | Fallback |

### 4.2 Détection automatique

Le système détecte automatiquement le hardware et choisit la configuration optimale :

```python
# sdnc/utils/device.py / brain_hybrid/utils/device.py
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability()
    if capability >= (8, 0):      # Ampere+
        dtype = torch.bfloat16    # bfloat16 natif
    elif capability >= (7, 5):    # Turing (T4)
        dtype = torch.float16     # float16 avec mixed precision
    else:
        dtype = torch.float32
else:
    dtype = torch.float32         # CPU fallback
```

### 4.3 Exigences mémoire

| Composant | VRAM requise | RAM requise |
|-----------|-------------|-------------|
| SDNC v2 (1000 circuits, CLIP) | ~2 GB | ~4 GB |
| Brain-Hybrid (Qwen3-4B bfloat16) | ~8 GB | ~16 GB |
| Brain-Hybrid (Qwen3-8B Q8) | ~10 GB | ~16 GB |
| Pipeline Teacher (Qwen3-32B) | — | ~70 GB (CPU) |
| Pipeline Student + Teacher | ~10 GB | ~80 GB |
