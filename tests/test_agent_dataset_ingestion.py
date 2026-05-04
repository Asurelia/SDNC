import numpy as np

from sdnc.agent.config import AutonomousConfig
from sdnc.agent.dataset_ingestion import DatasetIngestor
from sdnc.agent.experience_packs import ExperiencePack
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
