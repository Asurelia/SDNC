# SDNC - Sparse Dynamic Neural Circuits

SDNC est un projet de recherche Python pour construire un systeme d'IA local,
leger et apprenant par interaction. L'objectif n'est pas de remplacer le coeur
par un transformer plus petit, mais d'explorer une autre voie:

```text
signaux sensoriels
-> circuits sparse locaux
-> memoire persistante
-> outils reels
-> feedback utilisateur
-> plasticite locale
-> experts vivants verifies
```

Le systeme peut fonctionner sans telecharger de LLM. Qwen, Gemma, Claude,
Codex, Gemini CLI ou d'autres modeles peuvent etre utilises comme sources
externes d'hypotheses, mais ils ne sont pas le centre de SDNC.

## Etat Actuel

SDNC contient deux couches:

- `sdnc/core`: experiences de circuits liquid/CfC/NCP.
- `sdnc/agent`: systeme autonome runnable avec memoire SQLite, outils,
  apprentissage local, perception multimodale compacte et interface web.

L'agent actuel sait:

- traiter une interaction texte sans LLM local;
- activer seulement une petite fraction de circuits;
- persister ses episodes, procedures, lacunes, experts et traces;
- utiliser des outils simples: memoire, recherche fichier, lecture fichier,
  calculatrice, recherche web optionnelle;
- classer des actions possibles avec un planner d'active inference leger:
  repondre, rappeler la memoire, utiliser des outils, demander du feedback ou
  noter une lacune;
- recevoir du feedback et modifier seulement les circuits actifs;
- ingerer des observations texte, image, audio et video sous forme de signaux
  sensoriels compacts;
- gerer une file locale de fichiers d'entrainement depuis l'interface web;
- creer, tester, promouvoir, refroidir ou rejeter des experts avec payloads
  compresses, checksums et decode `L0/L1/L2`;
- lancer une evaluation locale qui mesure routing outil, memoire, surprise,
  latence, budget chaud et respect du ratio sparse;
- lancer un cycle de sommeil/replay borne pour consolider les episodes utiles;
- extraire des regles neuro-symboliques avec provenance et contre-exemples;
- journaliser les evenements pour le monitoring local.

## Installation

```powershell
python -m pip install -e ".[dev]"
```

Dependances optionnelles:

```powershell
python -m pip install -e ".[datasets]"
python -m pip install -e ".[sync]"
python -m pip install -e ".[directml]"
```

## Commandes

Lancer les tests:

```powershell
python -m pytest -q
```

Lancer une interaction autonome unique:

```powershell
python -m sdnc.agent.cli --no-web --once "calcule 2 + 2"
```

Lancer l'evaluation locale SDNC:

```powershell
python -m sdnc.agent.eval --workspace . --reset
```

Lancer l'agent interactif en console:

```powershell
python -m sdnc.agent.cli --workspace .
```

Dans la console, `/sleep-preview` inspecte les episodes qui seraient rejoues et
`/sleep` lance une consolidation bornee.

`/rules` lance une consolidation neuro-symbolique: les traces verifiees peuvent
devenir des regles locales, et les contre-exemples les affaiblissent au lieu de
les remplacer silencieusement.

Lancer l'interface web locale:

```powershell
python -m sdnc.agent.web --host 127.0.0.1 --port 8787 --workspace .
```

Puis ouvrir:

```text
http://127.0.0.1:8787/
```

## Interface Web

Le tableau de bord local sert a piloter SDNC sans ligne de commande:

- zone de dialogue;
- drag and drop de fichiers;
- file `a traiter`, `en cours`, `termine`, `a revoir`;
- boutons pour traiter un fichier, lancer un lot, declarer une lacune ou
  declencher une amelioration;
- sliders de budget cognitif et taille de lot;
- monitoring des circuits, memoire, experts, traces et evenements.

Les fichiers importes restent locaux. Les donnees runtime sont conservees dans
`data/`, qui n'est pas versionne par Git.

## Architecture Courte

```text
Observation
  -> LocalMultimodalEncoder / PerceptionBus
  -> HashingExperienceEncoder
  -> BudgetManager + ContextLODCompressor
  -> LocalCircuitLearner
  -> Sparse Cognitive Core / Global Workspace
  -> ExpertManager
  -> ExpertAtlas payload decode
  -> ToolRegistry
  -> PersistentMemory
  -> SleepConsolidationCycle
  -> RuleEngine
  -> feedback / self-improvement
```

Le centre cognitif sparse ne sait pas tout. Il garde l'etat mental courant,
selectionne ce qui merite attention, compare predictions et realite, puis
orchestre memoire, experts, outils et feedback.

Principes non negociables:

- pas de teacher model au centre;
- pas de centre monolithique qui stocke toute la connaissance;
- pas de distillation Qwen/Gemma comme moteur principal;
- pas de backpropagation globale pendant l'interaction;
- activation sparse bornee a `max_active_ratio <= 0.05`;
- SQLite et l'event log local restent la source de verite;
- un expert n'est promu qu'apres consensus et experience sandbox;
- les rejets sont conserves comme apprentissage.

## Recherche Compression / Experts

La piste SDNC pour se rapprocher d'une grande capacite avec peu de VRAM n'est
pas de stocker litteralement des trillions de poids en 10-12 GB. La piste
plausible est:

```text
coeur chaud 6-9B
+ atlas froid d'experts compresses
+ micro-experts actives ponctuellement
+ memoire vectorielle / SQLite
+ verification locale
+ decompression L0/L1/L2 a la demande
```

Le premier atlas concret vit dans `sdnc/agent/expert_atlas.py`. Il supporte
des payloads `procedure`, `prototype`, `low_rank`, `sparse_delta` et
`codebook`, avec checksum, taille en octets, cout de decode estime et hints
RAM/VRAM pour eviter de chauffer un expert trop cher.

Voir:

- [Compression Research](docs/SDNC_COMPRESSION_RESEARCH.md)
- [Non-Transformer Roadmap](docs/NON_TRANSFORMER_ROADMAP.md)
- [Efficiency Strategy](docs/EFFICIENCY_STRATEGY.md)
- [Architecture](docs/ARCHITECTURE.md)

## Documentation

- [Architecture actuelle](docs/ARCHITECTURE.md)
- [Plan d'implementation](docs/IMPLEMENTATION_PLAN.md)
- [Roadmap complete du modele](docs/COMPLETE_MODEL_ROADMAP.md)
- [Strategie d'efficacite](docs/EFFICIENCY_STRATEGY.md)
- [Roadmap non-transformer](docs/NON_TRANSFORMER_ROADMAP.md)
- [Sources de datasets](docs/DATASET_SOURCES.md)
- [Dashboard d'entrainement](docs/TRAINING_DASHBOARD.md)
- [Recherche compression](docs/SDNC_COMPRESSION_RESEARCH.md)
- [Notes de recherche](docs/RESEARCH_NOTES.md)
- [Specification initiale archivee](docs/SDNC_PHASE1_SPEC.md)

## Donnees Locales

Ne pas committer:

- `data/`
- `checkpoints/`
- `__pycache__/`
- `.pytest_cache/`
- environnements virtuels

Ces chemins sont ignores par `.gitignore`.

## Statut

SDNC est experimental. Le but est de construire progressivement un systeme qui
apprend par essais, memoire, outils et feedback, plutot qu'un gros modele
fige apres entrainement.
