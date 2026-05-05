# SDNC

Sparse Dynamic Neural Circuits is a Python research codebase for a lightweight
interaction-learning AI system: sparse local circuits, persistent memory, tools,
and feedback-driven plasticity.

## Tech Stack
- Python 3.11+, PyTorch, `ncps`, numpy.
- `sdnc/agent` uses only stdlib + numpy at runtime so it can run without a downloaded LLM.
- Legacy visual experiments use frozen OpenCLIP/Whisper encoders as translators, not as the learning core.

## Commands
- Install dev deps: `pip install -e ".[dev]"`
- Run all tests: `pytest`
- Run autonomous loop once: `python -m sdnc.agent.cli --no-web --once "calcule 2 + 2"`
- Run interactive learner: `python -m sdnc.agent.cli --workspace .`
- Run local web UI: `python -m sdnc.agent.web --host 127.0.0.1 --port 8787 --workspace .`
- Run SDNC eval harness: `python -m sdnc.agent.eval --workspace . --reset`
- Optional dataset deps: `pip install -e ".[datasets]"`

## Important Files
- `sdnc/agent/system.py` - observe -> sparse circuits -> tools -> memory -> local learning.
- `sdnc/agent/cognitive_core.py` - sparse global workspace that organizes attention, belief state, prediction traces, experts, memory, tools, and feedback without storing all knowledge itself.
- `sdnc/agent/budget.py` - cognitive modes and hot/cold RAM/VRAM estimates.
- `sdnc/agent/context_lod.py` - long-context compression into summaries/prototypes.
- `sdnc/agent/expert_atlas.py` - compressed expert payloads, L0/L1/L2 decode, checksums, hot cost hints.
- `sdnc/agent/experts.py` - lifecycle for self-created experts and hot/cold residency.
- `sdnc/agent/plasticity.py` - Oja-style local circuit updates and sparse activation.
- `sdnc/agent/memory.py` - SQLite episodic/procedural memory.
- `sdnc/agent/learning.py` - lacune detection, source hypotheses, consensus, local verification.
- `sdnc/agent/multimodal.py` - local signatures for text/image/audio/video observations.
- `sdnc/agent/perception.py` - sensory signal binding into sparse multimodal events.
- `sdnc/agent/planner.py` - active-inference-style action planner for answer/tool/feedback/gap choices.
- `sdnc/agent/dataset_ingestion.py` - local/Hugging Face dataset rows as experiences.
- `sdnc/agent/experience_packs.py` - compressed prototypes for experience packs.
- `sdnc/agent/eval.py` - deterministic learning/sparsity/tool/memory evaluation harness.
- `sdnc/agent/sync.py` - optional event mirrors; local SQLite stays authoritative.
- `sdnc/agent/self_improvement.py` - consensus, sandbox experiments, growth, pruning.
- `sdnc/agent/tools.py` - real tool registry: memory, web, file search/read, calculator.
- `sdnc/agent/web.py` and `sdnc/agent/static/` - local browser UI and JSON API.
- `sdnc/core/` - liquid/CfC/NCP circuit experiments.
- `docs/ARCHITECTURE.md` - current architecture.
- `docs/COMPLETE_MODEL_ROADMAP.md` - precise phase-by-phase plan to finish the SDNC model.
- `docs/EFFICIENCY_STRATEGY.md` - how SDNC avoids transformer-style dense memory/compute.
- `docs/NON_TRANSFORMER_ROADMAP.md` - translates transformer-era compression into SDNC-native research milestones.
- `docs/DATASET_SOURCES.md` - Hugging Face datasets suitable for SDNC-style experience learning.
- `docs/TRAINING_DASHBOARD.md` - local training cockpit, file queue, monitoring, and UI controls.
- `docs/SDNC_PHASE1_SPEC.md` - archived original long specification.

## Rules
- **CRITICAL**: Do not put Qwen, Gemma, or any LLM at the center of SDNC. A language model may only be an optional peripheral tool.
- **CRITICAL**: SDNC's center is a sparse cognitive coordinator / global workspace, not a monolithic model or source of truth.
- **CRITICAL**: The autonomous interaction system must learn through local updates, memory, tools, and feedback; no teacher/distillation pipeline.
- **CRITICAL**: New circuits/skills must pass consensus plus sandbox experimentation before promotion.
- **CRITICAL**: External tools/models are hypothesis sources only. Local tests, tool traces, memory, and user feedback decide what is learned.
- **CRITICAL**: Keep sparse activation bounded to `max_active_ratio <= 0.05` unless a test explicitly covers a research exception.
- **CRITICAL**: SQLite/local event log is the source of truth. Convex or any cloud sync is optional best-effort mirroring only.
- **CRITICAL**: Prefer hot/cold budgets and context LOD over dense always-on context or experts.
- **CRITICAL**: Multimodal input must flow through sensory signals, bindings, and memory; do not flatten everything into one giant prompt.
- **CRITICAL**: Experts are not fixed assets; maintain lifecycle status, utility, hot/cold residency, and retirement evidence.
- **NEVER** reintroduce `brain_hybrid`, `pipeline`, `teacher_wrapper`, `distillation_engine`, or Qwen wrapper code.
- **NEVER** reset circuit state during a normal interaction; reset only for tests or explicit session boundaries.
- **ALWAYS** persist learned interaction state through `AutonomousConfig.memory_path` and `state_path`.
- **ALWAYS** keep interaction, feedback, and improvement events append-only through `sync_events` when changing web/runtime behavior.
- **ALWAYS** persist multimodal observations in `sensory_bindings` when changing perception/runtime behavior.
- **ALWAYS** use `training_files` plus `sync_events` for dashboard file ingestion and processing state.
- **ALWAYS** record lacunes in `learning_gaps` and update `source_stats` when adding source-backed learning behavior.
- **ALWAYS** update `docs/EFFICIENCY_STRATEGY.md` when changing budget, context compression, or expert residency behavior.
- **ALWAYS** create or reinforce experts only from verified learning evidence, not from one untested source claim.
- **ALWAYS** record rejected self-improvement experiments; rejection is learning, not failure.
- **ALWAYS** add or update tests when changing routing, plasticity, memory persistence, or tool selection.
- Frozen encoders are allowed for perception; learning belongs in circuits/memory/procedures.

## Workflow
- Prefer extending `sdnc/agent` for interaction learning.
- Keep research docs current with source URLs and verification dates.
- Preserve user data under `data/`; do not delete learned memory or checkpoints unless explicitly asked.
