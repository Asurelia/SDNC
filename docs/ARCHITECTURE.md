# SDNC Interaction Architecture

Last verified with web research: 2026-05-02.

## Goal

SDNC is now split into two layers:

1. `sdnc/core`: liquid/CfC/NCP experiments for sparse neural circuits.
2. `sdnc/agent`: a runnable interaction learner that can operate without a
   large language model.

The agent is not a wrapper around Qwen, Gemma, or a teacher model. Its target
center is a sparse cognitive coordinator, not an all-knowing model:

```text
sensory signals -> PerceptionBus -> bound sensory event
               -> sparse local circuits
               -> Sparse Cognitive Core / Global Workspace
               -> living experts + memory + tools
               -> memory/tool selection -> real tools
               -> feedback -> local Oja-style plasticity + SQLite memory
               -> lack detection -> multi-source hypotheses -> local verification
               -> periodic self-improvement sandbox
```

The Sparse Cognitive Core keeps the current belief packet, attention/salience,
prediction traces, uncertainty, and action proposals. It organizes perception,
memory, experts, tools, and feedback, but it does not store all knowledge and it
does not replace the specialized circuits.

## Runtime Modules

- `HashingExperienceEncoder`: deterministic text/context encoder. It is not a
  transformer and does not need downloaded weights.
- `BudgetManager`: chooses `fast`, `think`, `max`, or `open` cognitive modes.
  `open` is the laboratory mode: it avoids tool truncation so failures and
  loops can be observed before better constraints are learned.
- `ContextLODCompressor`: replaces dense long-context handling with segment
  summaries, prototypes, and compact routing text.
- `CognitiveCore` / `SparseGlobalWorkspace`: a bounded state coordinator that
  receives proposals from perception, circuits, memory, experts, and tools,
  then chooses what deserves attention or action. It persists prediction,
  observation, uncertainty, surprise, and attention focus in `cognitive_traces`.
- `ActionPlanner`: ranks `answer`, `recall_memory`, `use_tools`,
  `ask_feedback`, and `investigate_gap` with a practical expected-free-energy
  approximation. It does not execute hidden reasoning; it exposes the chosen
  action and rejected candidates in interaction metadata and cognitive traces.
  In `open` mode, planner candidates remain visible but execution follows all
  relevant local proposals so the experiment reveals true behavior.
- `ExpertAtlas`: defines compressed expert payloads for `procedure`,
  `prototype`, `low_rank`, `sparse_delta`, and `codebook` experts. Each payload
  has L0/L1/L2 decode views, byte size, decode cost, hot RAM/VRAM hints, and a
  checksum.
- `ExpertManager`: manages self-created experts as living assets with
  probation, active, hot/cold, and retired states.
- `SleepConsolidationCycle`: builds a salience-ranked replay queue, previews or
  replays useful episodes with low-strength local plasticity, rejects negative
  feedback, detects excessive circuit drift, and strengthens repeated
  tool-success procedures.
- `MemoryCompactionCycle`: groups repeated non-protected episodes into compact
  cold prototypes in `memory_compactions`. It preserves source episode ids,
  protected rule evidence, counterexamples, and rule-link provenance; first-slice
  compaction never deletes raw episodes.
- `SensoryPrototypeLearner`: compacts repeated text/image/audio/video events
  into reusable sensory prototypes. Prototype matches are reported during
  observation and help SDNC recognize recurring multimodal situations without
  loading raw media into hot context.
- `RuleEngine`: extracts provenance-backed neuro-symbolic rules from repeated
  verified traces. Rules hold trigger patterns, preconditions, action/tool,
  expected outcome, confidence, provenance, and counterexamples; they can
  influence tool routing only while sufficiently confident. When two useful
  enabled rules share a trigger but propose incompatible tools, the engine stores
  an explicit conflict fork instead of overwriting either rule.
- `LocalCircuitLearner`: maintains circuit keys, liquid state, usage counters,
  and sparse inter-circuit weights. It activates at most 5 percent of circuits.
  State checkpoints are saved by atomic replace and partial reads are ignored so
  the web UI can start while another process is checkpointing.
- `PersistentMemory`: SQLite episodic, procedural, and conversation-example
  memory. Episodes store embeddings, active circuits, salience, and feedback.
  Conversation examples store source-backed prompt/response pairs learned from
  datasets or direct interaction. These examples are retrieval evidence, not a
  single source of truth: response synthesis filters them through lightweight
  intent compatibility before using a learned answer. Procedures store learned
  tool-use patterns. The connection runs in WAL mode with local indexes for
  concurrent UI reads and learning writes. At runtime, it also maintains a
  RAM-side dense vector index over episodes, conversation examples, sensory
  bindings, sensory prototypes, procedures, rules, and experts so large
  ingestion runs do not rescan every SQLite row on each observation. The hot
  index keeps only vectors and small ranking metadata; full text, responses,
  context JSON, and sensory payloads stay cold in SQLite until the top
  candidates are fetched. Long-lived readers incrementally refresh this RAM
  index from SQLite watermarks when another process writes to the same local
  database.
- `ToolRegistry`: real tools for memory recall, web search, workspace file
  search/read, and arithmetic experiments.
- `InteractionLearningSystem`: orchestrates the loop and persists state.
- `SelfDirectedLearner`: detects lacunes, gathers source hypotheses, builds
  consensus, verifies locally, then consolidates only proven procedures.
- `SelfImprovementCycle`: turns discoveries into bounded experiments before
  any new circuit or skill is promoted.
- `LocalMultimodalEncoder`: converts text, image, audio, and video samples into
  the same sparse vector space without making CLIP/Whisper mandatory.
- `PerceptionBus`: binds one or more sensory signals into a compact event with
  modality list, signal reliabilities, binding score, and a fused embedding.
- `DatasetIngestor`: streams local or Hugging Face rows as experiences.
  Local `.parquet` training files are recognized as datasets when Polars is
  installed; rows such as `source`/`target` become sensory text experiences,
  and prompt/label pairs are also stored as conversation examples for later
  response synthesis. Ingestion reports are attached to the visible interaction
  result.
- `speech_training`: local CLI runner for large French conversation corpora. It
  ingests canonical `source`/`target` parquet files into the same memory/circuit
  loop and writes JSONL progress checkpoints outside the browser.
- `ExperiencePack`: compressed prototypes built from many observations for
  later sparse recall.
- `EvaluationHarness`: deterministic probes for tool routing, repeated memory,
  surprise, uncertainty, latency, hot VRAM estimates, and sparse activation
  limits.
- `GuidedCurriculum`: beginner-safe training cycles that run through the real
  interaction loop for calculator routing, memory recall, local file lookup,
  multimodal sensory prototypes, neuro-symbolic rules, and bounded replay. It
  scores each step, applies local feedback, and emits append-only curriculum
  events for the web UI.
- `sync_events`: append-only local event log used by the web UI for realtime
  updates. Convex can mirror these events, but SQLite remains the authority.
- `sdnc.agent.web`: local browser interface and JSON API around the same
  long-lived interaction system.

## Learning Rule

The interaction layer avoids global backpropagation. For each observation:

1. Encode observation into a unit vector.
2. Activate top-k circuits only.
3. Update active circuit keys with an Oja-style local rule.
4. Strengthen only connections between co-active circuits.
5. Store salient episodes and successful tool patterns.

Inactive circuits are untouched. Negative feedback weakens the current
association instead of changing the whole model.

## Conversation Use

SDNC can now reuse learned prompt/response pairs, but only after a small
cognitive gate:

```text
user text -> embedding recall -> candidate examples
          -> intent compatibility
          -> required tool override when applicable
          -> answer or explicit uncertainty
```

This prevents a near vector match from becoming a false answer. For example,
an example that says "translate Bonjour into Spanish" can be retrieved for a
simple greeting, but it is rejected because its intent is translation, not
greeting. Natural arithmetic similarly forces the calculator tool even when a
near episodic memory or conversation example looks confident.

## Efficiency Path

SDNC's anti-transformer path is explicit:

```text
input/context -> cognitive budget -> context LOD -> sparse activation
              -> tool/memory selection -> local learning
```

The current runtime records:

- `cognitive_budget`: selected mode and limits;
- `context_lod`: segment/prototype counts and estimated tokens saved;
- `resource_budget`: estimated hot VRAM/RAM and whether it fits the configured
  hardware target.
- `introspection`: active circuits, proposed/executed/skipped tools, memory
  matches, hot experts, rules, planner candidates, and workspace trace.

More detail lives in `docs/EFFICIENCY_STRATEGY.md`.

## Expert Lifecycle

Experts are not frozen modules. They have a lifecycle:

```text
learning evidence -> candidate/probation expert -> active expert
                  -> hot when useful now
                  -> cold when not needed
                  -> retired if repeated evidence says it is weak
```

The expert registry is stored locally in SQLite:

- `experts.status`: `candidate`, `probation`, `active`, `cold`, or `retired`;
- `experts.utility`: moving usefulness score;
- `experts.hot`: whether it is currently resident in the hot path;
- `estimated_vram_gb` / `estimated_ram_gb`: budget hints;
- `payload_json`: source traces, tool names, evidence, and optional
  `expert_payload` atlas data.

Accepted self-directed learning creates or reinforces procedure experts. During
interaction, `ExpertManager` retrieves the most relevant experts and heats only
the subset allowed by the current cognitive budget. New procedure experts store
a deterministic atlas payload; the UI metadata can inspect L0 identity, L1
sketches, and L2 evidence without making the cognitive core a source of truth.

## Self-Improvement

SDNC does not immediately trust a single discovery. It uses this path:

```text
discovery -> repeated evidence -> consensus gate -> sandbox experiment
          -> promote/reject -> persist outcome
```

The consensus gate requires repeated salient experience plus source diversity
such as novelty, successful tools, feedback, and memory matches. The sandbox
then measures whether a candidate circuit separates its positive examples from
other recent memories and whether existing circuits already cover it. Failed
experiments are stored as useful limits.

Growth is bounded by `max_circuits`. If capacity is full, the system recycles a
quiet low-use circuit rather than growing without limit. Procedures/skills are
promoted from repeated successful tool patterns and pruned when repeated
failures show they are no longer useful.

## Sleep Replay

Replay is a bounded maintenance path, not hidden pretraining:

```text
salient episodes
  -> replay queue
  -> preview or low-strength local plasticity
  -> drift guard
  -> repeated tool-success procedure strengthening
  -> sync_events report
```

Negative-feedback episodes are held out. Useful existing procedures are listed
as protected before replay starts, and any replay that moves active circuit keys
past the configured drift limit is restored and recorded as rejected.

## Neuro-Symbolic Rules

Rules are local routing hints extracted from repeated verified experience:

```text
episodes + tool traces
  -> repeated successful pattern
  -> rule with provenance and expected outcome
  -> future tool-routing hint
  -> counterexample weakens or rejects, never overwrites
  -> incompatible useful rules become explicit forks
```

The `rules` SQLite table stores confidence, status, provenance ids, and
counterexample ids. During interaction, matched enabled rules are reported in
`metadata.neuro_symbolic_rules` and may add their tool to the candidate set.
This helps SDNC explain which learned rule influenced an action without making
that rule a single source of truth.

The `rule_links` SQLite table stores explicit relations from matched rules to
living SDNC assets. Today the supported targets are hot experts and sensory
prototypes. Each link keeps relation type, confidence, provenance ids, and a
small payload such as expert name or prototype key. Runtime metadata reports
attachment counts in `metadata.rule_attachments`, and web status exposes
`rule_link_summary` so the user can inspect which learned hints shaped the
current organization layer.

The `rule_conflicts` SQLite table stores unresolved forks between useful but
incompatible enabled rules. A fork is created only when both sides have enough
provenance and confidence; both rules remain enabled while the conflict records
topic, rule ids, action tools, evidence counts, and reason. `rule_conflict_summary`
is exposed by the API/UI so the cognitive center can keep uncertainty visible
and gather future evidence instead of choosing a premature single truth.

Rules are also controllable through the local API/UI: the user can consolidate,
enable, disable, reject, or add positive/counterexample feedback. These manual
actions append `rules` events and update rule confidence/status; they do not
erase provenance or counterexample history.

## Self-Directed Learning

When SDNC has a lacune, it should learn instead of bluffing. Current triggers:

- confidence below `confidence_threshold`;
- novelty above `lack_novelty_threshold`;
- failed tools;
- negative user feedback.

The learning path is:

```text
lacune -> memory/tool/external advisors -> consensus -> local sandbox
       -> procedure or rejection -> source reliability update
```

External CLI advisors such as Codex, Claude Code, Gemini CLI, or Cursor can be
plugged in through exact command tuples, but they are disabled by default. They
are treated as hypothesis sources. Local evidence, tests, tool traces, and user
feedback remain the judges.

Accepted hypotheses become compact procedural memory. Rejected hypotheses are
stored in `experiments` and source reliability is adjusted in `source_stats`.

## Sensory And Dataset Learning

SDNC's multimodal path is sensor-first, not transformer-first:

```text
text/image/audio/video sample(s)
  -> local sensory signatures
  -> PerceptionBus binding
  -> sensory_bindings + episodic memory
  -> sparse circuit update + expert routing
```

`sensory_bindings` stores the bound event separately from normal episodes:

- event id, source, modalities, sample ids;
- fused embedding and binding score;
- per-signal features/reliability;
- salience and optional linked episode id.

`sensory_prototypes` stores repeated event identity:

- modality set, sources, and sample ids;
- centroid embedding;
- observation count and confidence;
- compact reliability/binding features;
- payload pointing to the most recent event and episode.

The second time SDNC observes a similar multimodal event, the prototype match is
reported in `metadata.sensory_prototypes.matches`; after learning, the updated
prototype is reported in `metadata.sensory_prototypes.learned`.
When an enabled neuro-symbolic rule matches the same sensory event, SDNC stores
a `rule_links` relation to the learned prototype instead of burying that relation
inside a prompt or raw media context.

Datasets are not copied into dense weights. They are streamed as experiences:

```text
row -> ModalitySample(s) -> bound sensory event
    -> sparse circuit update + episodic/sensory memory
    -> optional ExperiencePack prototype
```

The optional Hugging Face path uses `datasets.load_dataset(..., streaming=True)`
when the `datasets` extra is installed. Hugging Face documents streaming as an
`IterableDataset` path for large datasets, and its feature system supports image
and audio decoding:

- https://huggingface.co/docs/datasets/en/stream
- https://huggingface.co/docs/datasets/about_dataset_features

## Memory Separation

- Working memory: `LocalCircuitLearner.circuit_state`, decayed every activation.
- Episodic memory: `episodes` table in SQLite.
- Compact memory: `memory_compactions` table, storing repeated episode
  prototypes with source ids, protected evidence ids, rule ids, and rule-link
  ids.
- Sensory memory: `sensory_bindings` table for cross-modal event bindings.
- Procedural memory: `procedures` table, mapping trigger embeddings to useful
  tools and success rates.
- Neuro-symbolic conflict memory: `rule_conflicts` table, keeping incompatible
  useful rules as inspectable forks with evidence and reason.
- Training queue: `training_files` table, storing local file status, SHA-256,
  modality, preview, linked episode, and processing errors.

## Local-First Sync

SDNC does not need Convex to function. The local source of truth is a SQLite
database configured for local reliability:

- WAL journaling so the web UI can read while the learner writes.
- `episodes`, `procedures`, `experiments`, `rules`, `rule_links`,
  `rule_conflicts`, and `tool_stats` for learning state.
- `learning_gaps` and `source_stats` for lacune tracking and source reliability.
- `sync_events` as an append-only event stream for interactions, feedback, and
  improvement/learning cycles.
- Server-Sent Events at `/api/stream` so the frontend updates without polling.
- `training_files` as the local dashboard queue for files to process, running
  files, completed files, and files needing review.

Convex is supported only as an optional best-effort mirror through
`--sync-to-convex --convex-url <url>`. If the mirror fails, SDNC keeps learning
locally and exposes the last sync error in `/api/status`.

## Tool Policy

Tools are selected from the current observation, confidence, memory, and learned
procedures:

- Always recall memory.
- Search the web for question/research/current-information inputs or low
  confidence.
- Search/read files for project/code/path inputs.
- Use calculator for arithmetic.
- Reuse procedural tool choices that succeeded on similar past interactions.
- Run `/improve` in the CLI to force a maintenance/self-improvement cycle.
- Run `/learn` in the CLI or `POST /api/learn-gap` to investigate the last
  interaction's lacune.

## Boundaries

- A future LLM may be added only as a tool, for example summarization or
  language formatting.
- No teacher-student distillation belongs in the core loop.
- OpenCLIP/Whisper remain acceptable frozen perception encoders for visual/audio
  experiments, not the source of learning.

## Run

```bash
python -m sdnc.agent.cli --no-web --once "calcule 2 + 2"
python -m sdnc.agent.cli --workspace .
python -m sdnc.agent.eval --workspace . --reset
python -m sdnc.agent.web --host 127.0.0.1 --port 8787 --workspace .
python -m sdnc.agent.web --host 127.0.0.1 --port 8787 --workspace . --sync-to-convex --convex-url <url>
```

Web endpoints:

- `GET /` - SDNC Lab UI.
- `GET /api/status` - circuits, sparsity, tools, modalities, memory paths.
- `GET /api/recent` - recent memories and sensory bindings.
- `POST /api/compact` - preview or run provenance-preserving memory compaction.
- `GET /api/events` - recent local sync events.
- `GET /api/stream` - local realtime SSE stream.
- `GET /api/curriculum` - guided local training steps.
- `POST /api/curriculum/run` - run one curriculum step or the full guided cycle.
- `POST /api/interact` - one interaction.
- `POST /api/observe` - one or more text/image/audio/video sensory samples.
- `GET /api/files` - local training file queue.
- `POST /api/files/upload` - add a local browser-selected file to the queue.
- `POST /api/files/status` - move a file between queue states.
- `POST /api/files/process` - process one queued file through SDNC.
- `POST /api/files/process-next` - process the next queued files as a batch.
- `POST /api/feedback` - feedback for the last interaction.
- `POST /api/improve` - forced self-improvement cycle.
- `POST /api/learn-gap` - source-backed learning cycle for the last interaction.
