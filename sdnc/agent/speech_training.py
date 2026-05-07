"""Launch SDNC conversation learning from local speech datasets.

The runner consumes a canonical parquet file with ``source`` and ``target``
columns. It is intentionally local: no LLM is trained or placed at the center,
and no Hugging Face dependency is required once the parquet exists.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from sdnc.agent.config import AutonomousConfig
from sdnc.agent.dataset_ingestion import DatasetIngestor
from sdnc.agent.system import InteractionLearningSystem


@dataclass(frozen=True)
class SpeechTrainingProgress:
    """One durable progress checkpoint for a speech training run."""

    timestamp: str
    source_path: str
    start_offset: int
    records_seen: int
    run_records_seen: int
    episodes_stored: int
    errors: int
    elapsed_s: float
    rows_per_second: float
    done: bool = False


def run_speech_training(
    dataset_path: Path,
    memory_path: Path,
    state_path: Path,
    workspace_root: Path,
    progress_log: Path,
    *,
    start_offset: int = 0,
    max_rows: int | None = None,
    batch_size: int = 250,
    progress_interval: int = 100,
    auto_improve: bool = True,
) -> SpeechTrainingProgress:
    """Ingest a local canonical speech dataset into SDNC memory/circuits."""

    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError("Install `polars` to run SDNC speech training from parquet.") from exc

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if start_offset < 0:
        raise ValueError("start_offset must be >= 0")

    progress_log.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.read_parquet(dataset_path)
    required = {"source", "target"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"canonical speech dataset is missing columns: {', '.join(missing)}")

    if start_offset:
        frame = frame.slice(start_offset)
    if max_rows is not None:
        frame = frame.head(max_rows)

    config = AutonomousConfig(
        memory_path=memory_path,
        state_path=state_path,
        workspace_root=workspace_root,
        allow_web=False,
        auto_improve_enabled=auto_improve,
    )
    system = InteractionLearningSystem(config)
    ingestor = DatasetIngestor(system)
    started = time.perf_counter()
    records_seen = 0
    episodes_stored = 0
    errors = 0

    try:
        for batch in frame.iter_slices(n_rows=batch_size):
            rows = batch.iter_rows(named=True)
            report = ingestor.ingest_records(
                rows,
                source=f"speech:{dataset_path.name}",
                text_columns=("source", "target"),
                label_columns=("target",),
                build_pack=False,
                learn=True,
                progress_interval=progress_interval,
                emit_done=False,
            )
            records_seen += report.records_seen
            episodes_stored += report.episodes_stored
            errors += len(report.errors)
            progress = _progress(
                dataset_path,
                records_seen,
                episodes_stored,
                errors,
                started,
                start_offset=start_offset,
                done=False,
            )
            _emit_dataset_progress(system, dataset_path, progress)
            _append_progress(progress_log, progress)
            print(json.dumps(asdict(progress), ensure_ascii=False), flush=True)
        final = _progress(
            dataset_path,
            records_seen,
            episodes_stored,
            errors,
            started,
            start_offset=start_offset,
            done=True,
        )
        _emit_dataset_progress(system, dataset_path, final)
        _append_progress(progress_log, final)
        print(json.dumps(asdict(final), ensure_ascii=False), flush=True)
        return final
    finally:
        system.close()


def _progress(
    dataset_path: Path,
    records_seen: int,
    episodes_stored: int,
    errors: int,
    started: float,
    *,
    start_offset: int,
    done: bool,
) -> SpeechTrainingProgress:
    elapsed = max(time.perf_counter() - started, 1e-6)
    return SpeechTrainingProgress(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        source_path=str(dataset_path),
        start_offset=start_offset,
        records_seen=start_offset + records_seen,
        run_records_seen=records_seen,
        episodes_stored=episodes_stored,
        errors=errors,
        elapsed_s=round(elapsed, 3),
        rows_per_second=round(records_seen / elapsed, 3),
        done=done,
    )


def _append_progress(path: Path, progress: SpeechTrainingProgress) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(progress), ensure_ascii=False) + "\n")


def _emit_dataset_progress(
    system: InteractionLearningSystem,
    dataset_path: Path,
    progress: SpeechTrainingProgress,
) -> None:
    emit = getattr(system, "_emit_event", None)
    if emit is None:
        return
    emit(
        "dataset",
        {
            "source": f"speech:{dataset_path.name}",
            "source_path": str(dataset_path),
            "start_offset": progress.start_offset,
            "records_seen": progress.records_seen,
            "run_records_seen": progress.run_records_seen,
            "episodes_stored": progress.episodes_stored,
            "errors_count": progress.errors,
            "elapsed_s": progress.elapsed_s,
            "rows_per_second": progress.rows_per_second,
            "done": progress.done,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local SDNC speech/conversation training.")
    parser.add_argument("--dataset", type=Path, required=True, help="Canonical parquet with source/target columns.")
    parser.add_argument("--memory", type=Path, default=Path("data/autonomous_memory.sqlite3"))
    parser.add_argument("--state", type=Path, default=Path("data/autonomous_circuits.npz"))
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--progress-log", type=Path, default=Path("data/speech_training_progress.jsonl"))
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional explicit cap for smoke tests.")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--no-auto-improve", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    run_speech_training(
        dataset_path=args.dataset,
        memory_path=args.memory,
        state_path=args.state,
        workspace_root=args.workspace,
        progress_log=args.progress_log,
        start_offset=args.start_offset,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        progress_interval=args.progress_interval,
        auto_improve=not args.no_auto_improve,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
