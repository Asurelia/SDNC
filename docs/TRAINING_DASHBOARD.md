# SDNC Training Dashboard

Last updated: 2026-05-06.

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

## API

- `GET /api/files`
- `POST /api/files/upload`
- `POST /api/files/status`
- `POST /api/files/process`
- `POST /api/files/process-next`
- `GET /api/curriculum`
- `POST /api/curriculum/run`

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
