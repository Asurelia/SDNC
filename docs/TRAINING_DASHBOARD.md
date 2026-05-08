# SDNC Training Dashboard

Last updated: 2026-05-08.

The dashboard borrows the ergonomic shape of defendGPT's training UI: data
preparation, launch controls, resume/review paths, live metrics, logs, and
preview. It does not borrow the transformer architecture. In SDNC, the UI drives
experience learning:

```text
files/signals -> local queue -> PerceptionBus -> sensory memory
              -> sparse circuits -> expert lifecycle -> logs/monitoring
```

## Beginner Flow

1. Drop files into the `Données` area.
2. Files enter the local SQLite-backed queue as `queued`.
3. Use `Suivant` or `Lot` to process them.
4. The file moves through `queued -> running -> done` or `failed`.
5. Monitoring updates confidence, novelty, salience, hot experts, tool traces,
   memories, sensory bindings, and append-only logs.
6. Use `Curriculum` to run one guided exercise or a full beginner cycle when
   you want an immediate local training check without preparing a dataset.
7. When the response is wrong or missing, fill the teaching pair and use
   `Enseigner` or `Corriger dernière`; SDNC stores that pair as conversation
   evidence and later responses show the accepted/rejected recall decision.

No file is sent to a cloud service by this dashboard. Uploaded files are written
under `AutonomousConfig.file_queue_path` and indexed in SQLite.

## Local Storage

- `training_files` table stores file metadata, status, SHA-256, preview,
  modality, error, and linked episode id.
- File bytes are stored under `data/file_queue/<file_id>/<safe_name>`.
- `sync_events` records queue, status, processing, observation, learning, and
  improvement events.

## UI Controls

- `Mode` slider maps to SDNC cognitive budgets: `fast`, `think`, `max`.
- `Ouvert` switches to SDNC's laboratory mode: no tool truncation, large
  recall/context windows, and full introspection of proposals and failures.
- `Lot` slider controls batch size for `/api/files/process-next`.
- `Apprendre` toggles local circuit/memory updates.
- `Outils` permits normal SDNC tool selection during file processing.
- File cards expose action buttons for processing, pause, resume, review, and
  manual status correction.
- `Règles` consolidates verified traces, shows rule links to experts/prototypes,
  and surfaces open rule conflict forks when two useful rules disagree.
- `Curriculum` runs real guided exercises for calculator routing, short memory,
  local file lookup, sensory prototypes, rule consolidation, and replay. It
  shows score, pass/fail notes, per-step metrics, and can keep sleep replay in
  preview mode.
- `Enseigner` stores one prompt/response pair directly in conversation memory,
  emits a teaching event, applies positive local feedback, and makes the next
  matching interaction inspectable through `conversation_decision`.
- `Corriger dernière` uses the last user input as the prompt and the typed
  correction as the desired response, so a bad answer can immediately become
  supervised local evidence without running a dataset job.
- `Trace vivante` exposes inspectable SDNC state in realtime: active circuits,
  proposed/executed/skipped tools, memory hits, experts, rules, and planner
  candidates. It is telemetry, not an external LLM chain-of-thought.
- `Sortie modèle` always shows the latest `InteractionResult.response`, episode
  id, executed tools, active circuits, conversation intent/decision, taught
  example id/source, and dataset ingestion summary when a dataset file is
  processed.
- `.parquet` files are treated as datasets when Polars is available. SDNC reads
  rows such as `source`/`target`, turns them into experiences, and reports rows,
  samples, stored episodes, and errors instead of pretending the binary file was
  useful text.
- Dataset runs emit `dataset` events during ingestion. The event log shows
  progress lines while the final model output keeps the completed report.
- Large speech datasets can also be launched outside the browser with
  `python -m sdnc.agent.speech_training`; this writes a JSONL progress log and
  uses the same SQLite memory/state files as the web UI.

## API

- `GET /api/files`
- `POST /api/files/upload`
- `POST /api/files/status`
- `POST /api/files/process`
- `POST /api/files/process-next`
- `GET /api/curriculum`
- `POST /api/curriculum/run`
- `POST /api/teach`

The dashboard remains intentionally local-first. Convex can mirror events later,
but SQLite is the source of truth.

## Inspiration Boundary

defendGPT is a transformer training dashboard. SDNC uses its UX pattern only:

- data preparation tiles;
- launch/resume controls;
- live metrics and logs;
- preview/debug panels.

The SDNC core remains sparse circuits, sensory memory, local feedback, tools,
and living experts.

Reference:

- https://github.com/anisayari/defendGPT
