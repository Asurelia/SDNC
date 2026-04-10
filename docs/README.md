# SDNC — Documentation Technique Complète

> **Sparse Dynamic Neural Circuits** — Architecture bio-inspirée combinant modèles de fondation gelés et circuits neuronaux adaptatifs avec apprentissage local.

**Version** : 0.2.0 | **Date** : Mars 2026 | **~8 700 SLOC** Python

---

## Structure de la documentation

| Document | Contenu | Lien |
|----------|---------|------|
| **01 — Vue d'ensemble** | Introduction, vision, analogie cerveau, architecture globale, stack technique, structure du projet, hardware | [01-overview.md](01-overview.md) |
| **02 — SDNC v2** | SDNCModel, PredictiveCircuitBank, GlobalState, TemporalStream, SparseRouter, Hebbian Learning, Integrator, Encoders, Memory, Data Flow complet | [02-sdnc-v2.md](02-sdnc-v2.md) |
| **03 — Brain-Hybrid** | BrainHybridModel, BrainModule, STDP, InjectionGate, HierarchicalPC, ACC, Hippocampus, StepScheduler, DistilledInputLayer, Data Flow complet | [03-brain-hybrid.md](03-brain-hybrid.md) |
| **04 — Pipeline & Entraînement** | Distillation teacher-student, ProjectionBridge, DualModelConfig, Corpus, Phases 1-3, Ablation, Scripts, Notebooks | [04-pipeline-training.md](04-pipeline-training.md) |
| **05 — Référence API** | Installation, API complète (toutes classes/méthodes), Configuration (SDNCConfig, BrainConfig, DualModelConfig), Tests, Glossaire, Références scientifiques | [05-api-reference.md](05-api-reference.md) |

---

## Quick Start

```bash
# Installation
pip install -e ".[dev]"

# Vérification GPU
python check_gpu.py

# Phase 1 — Vision few-shot
python run_train.py

# Tests
pytest tests/ -v
```

## Architecture en un coup d'oeil

```
         ┌─────────────┐              ┌──────────────────┐
         │  SDNC v2     │              │  Brain-Hybrid     │
         │              │              │                  │
         │  CLIP → 1000 │              │  Qwen + 4×CfC   │
         │  circuits    │              │  + STDP + SDM    │
         │  prédictifs  │              │  + ACC + PC      │
         └──────┬──────┘              └────────┬─────────┘
                │                               │
                └──────────┬────────────────────┘
                           │
                  ┌────────▼─────────┐
                  │  Distillation     │
                  │  Pipeline         │
                  │  Teacher → Bridge │
                  │  → Student+Brain  │
                  └──────────────────┘
```

## Principes fondamentaux

1. **Le Qwen ne s'entraîne jamais** — tous ses poids sont gelés
2. **L'apprentissage est local** — STDP, Oja, codage prédictif (pas de backprop global)
3. **Les états persistent** — jamais de reset pendant l'entraînement
4. **L'hippocampe s'enrichit en continu** — consolidation pendant le sommeil
5. **L'erreur de prédiction** est le seul signal d'apprentissage
