# SDNC Complete Model Roadmap

Last updated: 2026-05-04.

This roadmap defines what "finish the model" means for SDNC. SDNC is not
finished when it can chat once. It is finished when it can learn from
interaction, use tools, remember, test hypotheses, create bounded experts, and
improve its own behavior without putting a transformer or teacher model at the
center.

## Definition Of Done

SDNC reaches a complete research-model state when all of these are true:

- It runs locally from an empty state and keeps learning across restarts.
- It handles text, files, image/audio/video signatures, and tool results through
  sensory bindings rather than one dense prompt.
- It predicts outcomes, detects surprise, and updates only active circuits.
- It decides when to answer, when to search, when to ask for feedback, and when
  to run a bounded experiment.
- It creates, compresses, heats, cools, prunes, and rejects experts with stored
  evidence.
- It replays and consolidates high-salience experiences during idle cycles.
- It has a measurable benchmark suite showing improvement over time.
- It stays inside explicit hot RAM/VRAM budgets.
- It has a beginner-friendly UI for training, monitoring, memory, experts, and
  failures.
- It never depends on Qwen, Gemma, or any teacher/distillation loop as its core.

## Target Architecture

```text
User/environment/files/datasets
  -> sensory adapters
  -> PerceptionBus
  -> predictive sparse circuits
  -> belief state + context LOD
  -> expert router + compressed expert atlas
  -> tool/action planner
  -> local verifier
  -> memory + feedback + replay
```

The hot path must stay small:

```text
Hot:   active circuits, active experts, current belief packet, verifier heads
Warm:  recent decoded experts, codebooks, route prototypes, source stats
Cold:  SQLite, event log, sensory bindings, experience packs, expert atlas
```

## Phase 0 - Stabilize The Current Base

Goal: make the current autonomous learner reliable enough to build on.

Tasks:

- Lock current CLI and web API contracts.
- Add smoke tests for CLI one-shot, web upload/process, feedback, and improve.
- Add a memory migration/version table for SQLite schema changes.
- Add backup/export/import commands for memory and circuit state.
- Add a `sdnc doctor` or equivalent diagnostics command.
- Add corruption checks for state files and graceful fallback.
- Add structured runtime errors surfaced in UI logs.

Acceptance criteria:

- `python -m pytest -q` passes.
- A fresh workspace can run one interaction, restart, and recall its previous
  memory.
- A corrupt state file is reported and quarantined, not silently reused.
- UI status clearly shows memory path, state path, schema version, event count,
  and last error.

Primary files:

- `sdnc/agent/config.py`
- `sdnc/agent/memory.py`
- `sdnc/agent/system.py`
- `sdnc/agent/cli.py`
- `sdnc/agent/web.py`
- `tests/test_agent_*`

## Phase 1 - Predictive Coding Core

Goal: turn SDNC from reactive learner into prediction-error learner.

Tasks:

- Add predicted confidence, predicted tool usefulness, and predicted outcome to
  each interaction.
- Persist prediction vs observation deltas in SQLite.
- Compute surprise from confidence error, tool result error, novelty, and user
  feedback.
- Feed surprise into local plasticity intensity.
- Add per-circuit prediction error moving averages.
- Separate positive surprise from negative failure.
- Expose prediction traces in CLI and web UI.

Acceptance criteria:

- Every interaction stores a prediction trace.
- Feedback changes future confidence/tool selection on similar inputs.
- Repeated predictable interactions produce lower surprise.
- Novel or failed interactions produce higher surprise and learning gaps.
- Only active circuits are updated.
- `max_active_ratio <= 0.05` remains enforced.

Primary files:

- `sdnc/agent/plasticity.py`
- `sdnc/agent/system.py`
- `sdnc/agent/memory.py`
- `sdnc/agent/types.py`
- `sdnc/agent/static/app.js`

## Phase 2 - Active Inference Planner

Goal: make SDNC explore when useful instead of passively guessing.

Tasks:

- Add an `ActionPlanner` that ranks answer/search/tool/ask-feedback/experiment.
- Score actions with a practical expected-free-energy approximation:

```text
score =
  uncertainty_reduction
  + task_utility
  + user_relevance
  - cost
  - risk
  - memory_pollution_penalty
```

- Add action budgets for `fast`, `think`, and `max`.
- Make lacune handling choose the cheapest useful experiment first.
- Record skipped actions and why they were skipped.
- Add read-only sandbox tools for file/web/calculator experiments.
- Add UI panel for "why SDNC chose this action".

Acceptance criteria:

- Low-confidence tasks produce explainable exploration choices.
- The planner refuses expensive actions when expected value is too low.
- Failed hypotheses are recorded as useful negative evidence.
- Tool use becomes more selective over repeated tasks.

Primary files:

- `sdnc/agent/learning.py`
- `sdnc/agent/tools.py`
- `sdnc/agent/system.py`
- `sdnc/agent/self_improvement.py`
- `sdnc/agent/web.py`

## Phase 3 - Compressed Expert Atlas

Goal: move from metadata-only experts to bounded compressed expert assets.

Tasks:

- Define `ExpertPayload` formats:
  - `procedure`: symbolic/action template;
  - `prototype`: vector prototype and thresholds;
  - `low_rank`: small matrix factors;
  - `sparse_delta`: sparse residual update;
  - `codebook`: quantized block references.
- Add `decode_level`: `L0`, `L1`, `L2`.
- Add expert byte size, decode cost, hot RAM, and hot VRAM estimates.
- Add LRU-style hot/cold residency manager.
- Add codebook storage for compressed experts.
- Add deterministic decompressor first, learned neural decompressor later.
- Add expert integrity checks and payload checksums.
- Persist every promotion, rejection, decode, eviction, and verifier result.

Acceptance criteria:

- Experts have real payloads, not only names and metadata.
- The system can retrieve an expert at `L0`, sketch it at `L1`, and hot-decode
  it at `L2`.
- The budget manager refuses to heat experts that exceed current limits.
- Repeatedly useful experts are prefetched/heated faster.
- Repeatedly weak experts are cooled or retired.

Primary files:

- `sdnc/agent/experts.py`
- `sdnc/agent/budget.py`
- `sdnc/agent/memory.py`
- new `sdnc/agent/expert_atlas.py`
- new `sdnc/agent/decompressor.py`

## Phase 4 - Replay And Sleep Consolidation

Goal: let SDNC improve while idle without polluting memory.

Tasks:

- Add salience-ranked replay queue.
- Replay high-value episodes into local circuits with bounded plasticity.
- Consolidate repeated tool successes into stronger procedures.
- Compare candidate experts against recent failures before promotion.
- Add anti-forgetting checks for existing useful procedures.
- Add drift detection for circuit keys and expert utility.
- Add sleep-cycle UI controls: preview, run, stop, inspect outcome.

Acceptance criteria:

- Replay improves benchmark performance on repeated tasks.
- Replay does not erase previously verified procedures.
- Sleep cycles create explicit reports: strengthened, pruned, rejected.
- User can inspect why an expert changed.

Primary files:

- `sdnc/agent/self_improvement.py`
- `sdnc/agent/memory.py`
- `sdnc/agent/plasticity.py`
- `sdnc/agent/web.py`

## Phase 5 - Multimodal Perception Upgrade

Goal: strengthen text/image/audio/video handling while keeping frozen encoders
as translators only.

Tasks:

- Add optional frozen image/audio/video encoders behind feature flags.
- Keep local signature fallback for no-model operation.
- Add video keyframe/audio segment sampling.
- Add OCR/transcription hooks as optional tools, not learning core.
- Add cross-modal binding tests for repeated event identity.
- Add memory browser for sensory bindings and linked episodes.
- Add dataset ingestion presets for text, image, audio, video, and mixed rows.

Acceptance criteria:

- Text-only mode still works without downloaded models.
- Image/audio/video files produce sensory bindings and linked memories.
- Repeated multimodal observations retrieve related prior events.
- Frozen encoders cannot update core learning weights directly.

Primary files:

- `sdnc/agent/multimodal.py`
- `sdnc/agent/perception.py`
- `sdnc/agent/dataset_ingestion.py`
- `sdnc/agent/web.py`

## Phase 6 - Neuro-Symbolic Memory

Goal: extract verified reusable patterns from experience.

Tasks:

- Add a local rule/procedure graph:
  - trigger pattern;
  - preconditions;
  - action/tool;
  - expected outcome;
  - confidence;
  - provenance;
  - counterexamples.
- Promote repeated verified traces into rules.
- Attach rules to experts and sensory prototypes.
- Add contradiction detection between new evidence and old rules.
- Add memory compaction that preserves provenance and rejection evidence.
- Add UI for rule inspection, disable/enable, and manual feedback.

Acceptance criteria:

- SDNC can explain which memory/rule/expert influenced an answer.
- Contradictory evidence weakens or forks a rule instead of overwriting it.
- Rules improve tool choice on repeated tasks.
- Rejected rules remain searchable.

Primary files:

- `sdnc/agent/memory.py`
- `sdnc/agent/learning.py`
- `sdnc/agent/experts.py`
- new `sdnc/agent/rules.py`

## Phase 7 - Evaluation Harness

Goal: measure whether SDNC is actually learning.

Tasks:

- Add benchmark datasets for:
  - repeated user preference learning;
  - file QA with memory;
  - arithmetic/tool routing;
  - source-backed lacune learning;
  - multimodal event recall;
  - expert promotion/rejection;
  - replay consolidation.
- Add before/after metrics:
  - task success;
  - confidence calibration;
  - tool calls avoided;
  - memory hits;
  - surprise reduction;
  - expert utility;
  - hot RAM/VRAM estimate;
  - latency.
- Add benchmark CLI and JSON reports.
- Add regression gates in tests for sparsity, persistence, and no-LLM core.
- Add long-running soak test with many interactions.

Acceptance criteria:

- A benchmark run shows improvement after feedback/replay.
- Metrics are reproducible with a seed.
- CI/local tests fail if sparse activation exceeds limits.
- Reports show where SDNC learned, failed, or rejected a hypothesis.

Primary files:

- new `sdnc/agent/eval.py`
- new `tests/test_agent_eval.py`
- `docs/RESEARCH_NOTES.md`

## Phase 8 - Beginner Training Studio

Goal: make the local web UI capable of piloting the whole training process.

Tasks:

- Add expert atlas view with hot/cold/filter/status controls.
- Add memory/rule/sensory binding browser.
- Add sleep/replay controls.
- Add benchmark run button and report viewer.
- Add resource budget timeline.
- Add training file dataset presets.
- Add "why this happened" trace view for routing, tools, experts, memory.
- Add safe import/export buttons.

Acceptance criteria:

- A beginner can import files, process them, give feedback, run replay, and see
  what changed.
- Every major automatic action has a visible trace.
- UI never requires cloud sync.
- Runtime errors are understandable and recoverable.

Primary files:

- `sdnc/agent/static/index.html`
- `sdnc/agent/static/app.css`
- `sdnc/agent/static/app.js`
- `sdnc/agent/web.py`

## Phase 9 - Local Performance And Packaging

Goal: make SDNC installable and usable on the target Windows machine.

Tasks:

- Add package scripts:
  - `sdnc-agent`;
  - `sdnc-web`;
  - `sdnc-doctor`;
  - `sdnc-eval`;
  - `sdnc-export`;
  - `sdnc-import`.
- Add Windows GPU/runtime detection for CUDA, DirectML, CPU fallback.
- Add disk/RAM/VRAM budget presets.
- Add compact local config file.
- Add release checklist.
- Add versioned docs matching the package version.

Acceptance criteria:

- Fresh clone + install + web launch works from documented commands.
- `sdnc doctor` reports missing optional dependencies clearly.
- CPU-only mode remains functional.
- GPU mode is optional acceleration, not a requirement.

Primary files:

- `pyproject.toml`
- `sdnc/agent/cli.py`
- new `sdnc/agent/doctor.py`
- docs and README

## Phase 10 - Research Stretch: Learned Decompressor

Goal: test whether a small SDNC-native decompressor can reconstruct useful
expert blocks faster or smaller than deterministic formats.

Tasks:

- Build a dataset of verified expert payloads.
- Train a tiny neural codec on payload chunks.
- Compare deterministic low-rank/codebook decode vs learned decode.
- Measure quality by downstream task success, not just reconstruction error.
- Add strict budget gates and fallback to deterministic decode.
- Reject learned codec if it reduces reliability or explainability too much.

Acceptance criteria:

- Learned decompressor beats deterministic baseline on at least one measured
  metric without breaking reliability.
- Fallback path works when codec is unavailable.
- Codec cannot promote experts without verifier evidence.

Primary files:

- `sdnc/agent/decompressor.py`
- `sdnc/agent/expert_atlas.py`
- `sdnc/agent/eval.py`
- `docs/SDNC_COMPRESSION_RESEARCH.md`

## Suggested Build Order

1. Phase 0: stabilize state, migrations, diagnostics.
2. Phase 1: prediction traces and surprise.
3. Phase 7 minimal: benchmark harness for learning improvement.
4. Phase 2: active inference planner.
5. Phase 3: compressed expert atlas.
6. Phase 4: replay/sleep consolidation.
7. Phase 6: neuro-symbolic rules.
8. Phase 5: stronger multimodal adapters.
9. Phase 8: full training studio UI.
10. Phase 9: packaging and Windows/local runtime polish.
11. Phase 10: learned decompressor research.

This order matters. The evaluation harness must arrive before the more
speculative compression work, otherwise SDNC may look impressive without
measuring whether it is actually learning.

## Minimum Viable Complete Model

The smallest version that deserves to be called a complete SDNC model is:

- Phase 0 complete.
- Phase 1 complete.
- Phase 2 complete for read-only tools.
- Phase 3 complete for `procedure`, `prototype`, and deterministic `low_rank`
  payloads.
- Phase 4 complete for bounded replay.
- Phase 7 complete for repeated-learning benchmarks.
- UI can show memory, experts, surprise, and replay results.

This MVP does not need a learned decompressor, giant datasets, or GPU kernels.
It must prove the loop:

```text
experience -> prediction error -> local update -> verified expert/memory
           -> replay -> better future behavior
```

## Final Complete Model

The full target adds:

- compressed expert atlas with L0/L1/L2 decode;
- active inference planner with tool/search/sandbox choices;
- multimodal sensory binding with optional frozen translators;
- neuro-symbolic memory with provenance and counterexamples;
- replay/sleep consolidation;
- evaluation reports;
- beginner training studio;
- installable local package;
- optional learned decompressor if it proves useful.

## Global Risks

- Compression may hide errors instead of preserving knowledge.
- Experts may multiply without improving behavior.
- Replay may drift if salience and anti-forgetting checks are weak.
- Tool results can pollute memory if provenance is not strict.
- UI can become pretty but not diagnostic.
- Benchmarks can be gamed by memorization.

Mitigations:

- Keep rejected experiments.
- Require verifier tests before promotion.
- Track source reliability.
- Keep sparse activation limits.
- Prefer small measurable steps.
- Treat external LLMs as hypothesis sources only.

## Non-Goals

- No GPT replacement claim.
- No full transformer hidden inside the system.
- No teacher-student distillation as the core path.
- No destructive autonomous tools by default.
- No cloud database dependency.
- No memory ingestion without provenance and salience.
