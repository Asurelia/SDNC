# SDNC Interaction Plan

For the full phase-by-phase roadmap to finish the SDNC model, including
predictive coding, active inference, compressed expert atlas, replay/sleep,
evaluation, UI, packaging, and stretch research, see
`docs/COMPLETE_MODEL_ROADMAP.md`.

## V1 Delivered

- Runnable autonomous interaction package in `sdnc/agent`.
- Local deterministic encoder, no LLM dependency.
- Sparse circuit activation with maximum 5 percent active circuits.
- Oja-style local circuit updates and co-active connection reinforcement.
- Persistent SQLite episodic/procedural/sensory memory.
- Real tool registry: memory recall, web search, file search/read, calculator.
- Feedback API and CLI command `/feedback`.
- Consensus-gated self-improvement with sandbox experiments before promotion.
- Lack-driven learning cycle: low confidence/novelty/tool failures/feedback
  trigger source-backed hypotheses, consensus, local verification, and
  procedural consolidation.
- Bounded circuit growth/recycling and weak skill pruning.
- Local multimodal observations for text, image, audio, and video signatures.
- `PerceptionBus` binds multiple modality samples into one sensory event with
  reliability and binding scores.
- Dataset ingestion turns local or Hugging Face rows into bound sparse
  experiences.
- Experience packs compress many observations into reusable prototypes.
- Cognitive budgets (`fast`, `think`, `max`) control memory recall, tools,
  context LOD, and hot expert estimates.
- Sparse Cognitive Core / Global Workspace keeps a bounded belief packet,
  attention focus, prediction trace, uncertainty, and surprise for every learned
  interaction or observation.
- Active-inference-style planner ranks answer/memory/tool/feedback/gap actions
  and persists the selected policy in interaction metadata and cognitive traces.
- Context LOD replaces dense long-context handling with summaries/prototypes.
- Self-managed expert lifecycle: verified learning can create procedure
  experts, select a hot subset, cool unused experts, and retire weak ones.
- First compressed expert atlas: procedure/prototype/low-rank/sparse-delta/
  codebook payloads with `L0/L1/L2` decode views, checksums, byte size, decode
  cost, and hot RAM/VRAM hints.
- Local web UI with chat, sensory observation, circuits, traces, memory,
  feedback, and improve controls.
- Beginner-oriented training dashboard with drag/drop file queue, batch
  processing controls, live logs, monitoring, and queue statuses.
- Local-first realtime layer: SQLite WAL source of truth, append-only `sync_events`,
  and SSE stream for the frontend.
- Optional Convex mirror is available but disabled by default.
- Tests for encoding, plasticity, memory, and the end-to-end interaction loop.
- Tests for cognitive workspace bounds, cognitive trace persistence, web API
  exposure, and interaction metadata.
- Minimal Phase 7 evaluation harness in `sdnc/agent/eval.py`, with CLI/JSON
  reports for tool routing, memory reuse, surprise, latency, hot VRAM estimate,
  and sparse activation limits.
- Phase 7 advanced probes: `sdnc-eval` also measures sensory prototype recall,
  rule consolidation, and sleep/replay consolidation in `feature_probes`.
- First Phase 4 sleep/replay cycle in `sdnc/agent/replay.py`: salience-ranked
  queue, preview mode, bounded local plasticity, drift guard, negative-feedback
  holdout, repeated procedure strengthening, CLI `/sleep`, and API `/api/sleep`.
- First Phase 6 neuro-symbolic memory in `sdnc/agent/rules.py`: rules with
  provenance/counterexamples/confidence, contradiction weakening, tool-routing
  hints, CLI `/rules`, and API `/api/rules/consolidate`.
- Phase 6 rule links: `rule_links` now attaches matched rules to hot experts
  and sensory prototypes with provenance, runtime attachment counts, and
  `rule_link_summary` in web status.
- First Phase 5 sensory prototypes in `sdnc/agent/sensory_prototypes.py`:
  repeated multimodal events get compact prototype identity, observation count,
  confidence, metadata exposure, and `/api/recent` visibility.
- Archived the original Phase 1 spec in `docs/SDNC_PHASE1_SPEC.md`.

## Near-Term Hardening

1. Expand the benchmark harness:
   - same query after feedback should select the same useful tool faster;
   - incorrect feedback should reduce the same association;
   - memory retrieval should improve across sessions;
   - `/learn` should create fewer gaps after successful consolidation;
   - add multimodal recall, expert promotion/rejection, and replay tests.
2. Harden compressed expert residency:
   - persist decode and eviction events;
   - add LRU timestamps and prefetch scoring;
   - move large cold payloads from SQLite JSON into atlas pack files when they
     outgrow the local database;
   - add optional neural/adapted expert loading once deterministic payloads are
     benchmarked.
3. Add richer local database tooling:
   - compaction/backup command for the SQLite memory file;
   - event replay command from `sync_events`;
   - memory export/import with checksums.
4. Add richer front-end telemetry:
   - timeline of self-improvement experiments;
   - circuit usage heatmap across sessions;
   - memory/procedure browser with pruning controls.
   - sensory binding browser and cross-modal trace inspection.
   - budget/VRAM timeline.
5. Add active exploration tools:
   - run controlled read-only experiments against files/web/calculator;
   - record failed hypotheses as limits;
   - rank future experiments by uncertainty and usefulness.
6. Harden neuro-symbolic rules:
   - add richer inspection for rule-to-expert/prototype links;
   - add disable/enable/manual feedback controls;
   - add provenance-preserving memory compaction;
   - add contradiction forking when two rules both have useful evidence.
7. Upgrade sensory bridges:
   - frozen CLIP image embedding -> autonomous interaction vector;
   - frozen Whisper/audio embedding -> autonomous interaction vector;
   - video keyframe/audio temporal summaries;
   - optional local file ingestion from browser/API paths;
   - no backprop through CLIP;
   - richer web inspection for sensory prototypes.
8. Add salience replay:
   - add scheduled/idle triggering around the manual sleep endpoint;
   - expose richer sleep controls in the web UI;
   - add replay benchmarks that prove performance improves without forgetting.
9. Add tool safety profiles:
   - read-only default;
   - optional explicit write/execute tools later.
10. Add export/import for memory and circuit state.

## Acceptance Criteria

- The system starts from an empty memory file and improves through interaction.
- Learning state survives process restarts.
- No central Qwen/Gemma/teacher/distillation dependency exists.
- Local operation does not require Convex or any cloud database.
- The active circuit ratio remains <= 5 percent.
- New circuits/skills require repeated evidence plus a sandbox experiment.
- Lacune learning uses multiple sources when available and records source
  reliability.
- Dataset ingestion can run from in-memory records without internet; Hugging
  Face streaming is optional.
- Multimodal rows are bound as events, not flattened into one dense prompt.
- Every interaction reports cognitive budget, context compression, and hot
  resource estimate.
- The core interaction tests pass with `pytest tests -k agent`.

## Non-Goals For This Version

- No full chatbot generation model.
- No autonomous shell-writing or destructive tools.
- No claim of AGI or human-level reasoning.
- No replacement of the visual `sdnc/core` research path.
