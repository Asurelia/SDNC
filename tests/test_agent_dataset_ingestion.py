import numpy as np
import pytest

from sdnc.agent.config import AutonomousConfig
from sdnc.agent.dataset_ingestion import DatasetIngestor
from sdnc.agent.experience_packs import ExperiencePack
from sdnc.agent.speech_training import run_speech_training
from sdnc.agent.system import InteractionLearningSystem


def test_dataset_ingestor_learns_records_and_builds_pack(tmp_path):
    config = AutonomousConfig(
        input_dim=32,
        n_circuits=20,
        memory_path=tmp_path / "memory.sqlite3",
        state_path=tmp_path / "state.npz",
        workspace_root=tmp_path,
        allow_web=False,
    )
    system = InteractionLearningSystem(config)
    records = [
        {"text": "bug import pytest corrigé", "label": "code"},
        {"caption": "image sombre", "image": np.zeros((4, 4, 3), dtype=np.uint8), "label": "vision"},
        {
            "text": "son court",
            "audio": {"array": np.ones(64, dtype=np.float32), "sampling_rate": 8000},
            "label": "audio",
        },
    ]
    try:
        report = DatasetIngestor(system).ingest_records(records, source="fixture", limit=3)
        events = system.recent_events(limit=20)

        assert report.records_seen == 3
        assert report.samples_seen >= 3
        assert report.episodes_stored >= 1
        assert report.pack is not None
        assert len(report.pack.prototypes) >= 1
        assert any(event.event_type == "observation" for event in events)
    finally:
        system.close()


def test_experience_pack_roundtrip(tmp_path):
    config = AutonomousConfig(
        input_dim=32,
        n_circuits=20,
        memory_path=tmp_path / "memory.sqlite3",
        state_path=tmp_path / "state.npz",
        workspace_root=tmp_path,
        allow_web=False,
    )
    system = InteractionLearningSystem(config)
    try:
        report = DatasetIngestor(system).ingest_records(
            [{"text": "pattern répétable", "label": "repeat"}],
            source="fixture",
        )
        path = tmp_path / "pack.npz"
        report.pack.save(path)
        loaded = ExperiencePack.load(path)

        assert loaded.name == report.pack.name
        assert loaded.prototypes[0].count == report.pack.prototypes[0].count
    finally:
        system.close()


def test_speech_training_runner_ingests_canonical_parquet(tmp_path):
    pl = pytest.importorskip("polars")
    dataset = tmp_path / "speech.parquet"
    progress = tmp_path / "progress.jsonl"
    pl.DataFrame(
        {
            "source": ["bonjour", "explique une addition"],
            "target": ["salut, je t'écoute", "une addition combine deux nombres"],
            "dataset": ["fixture", "fixture"],
        }
    ).write_parquet(dataset)

    report = run_speech_training(
        dataset_path=dataset,
        memory_path=tmp_path / "memory.sqlite3",
        state_path=tmp_path / "state.npz",
        workspace_root=tmp_path,
        progress_log=progress,
        batch_size=1,
        auto_improve=False,
    )

    lines = progress.read_text(encoding="utf-8").strip().splitlines()
    assert report.done
    assert report.records_seen == 2
    assert report.episodes_stored >= 1
    assert len(lines) >= 2

    memory_system = InteractionLearningSystem(
        AutonomousConfig(
            input_dim=256,
            memory_path=tmp_path / "memory.sqlite3",
            state_path=tmp_path / "state_read.npz",
            workspace_root=tmp_path,
            allow_web=False,
        )
    )
    try:
        matches = memory_system.memory.retrieve_conversation_examples(
            memory_system.encoder.encode("bonjour"),
            top_k=1,
        )
        assert matches[0].response == "salut, je t'écoute"
    finally:
        memory_system.close()
