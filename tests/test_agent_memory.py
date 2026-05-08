import numpy as np

from sdnc.agent.memory import PersistentMemory


def test_persistent_memory_roundtrip(tmp_path):
    memory = PersistentMemory(tmp_path / "memory.sqlite3", embedding_dim=8)
    emb = np.ones(8, dtype=np.float32)
    emb /= np.linalg.norm(emb)

    episode_id = memory.store_episode(
        text="j'ai appris une interaction",
        context={"kind": "test"},
        embedding=emb,
        active_circuits=[1, 2],
        salience=0.8,
    )
    retrieved = memory.retrieve_similar(emb, top_k=1)

    assert retrieved[0].id == episode_id
    assert retrieved[0].active_circuits == [1, 2]
    assert retrieved[0].similarity > 0.99
    memory.close()


def test_vector_index_refreshes_external_writes(tmp_path):
    memory_path = tmp_path / "memory.sqlite3"
    reader = PersistentMemory(memory_path, embedding_dim=4)
    writer = PersistentMemory(memory_path, embedding_dim=4)
    emb = np.array([1, 0, 0, 0], dtype=np.float32)

    episode_id = writer.store_episode(
        text="external training write",
        context={"source": "writer"},
        embedding=emb,
        active_circuits=[1],
        salience=0.9,
    )

    assert reader.vector_index_summary()["episodes"] == 0
    reader._last_data_version = reader._data_version()
    retrieved = reader.retrieve_similar(emb, top_k=1)

    assert retrieved[0].id == episode_id
    assert reader.vector_index_summary()["episodes"] == 1
    assert reader.memory_storage_summary()["episodes"] == 1
    writer.close()
    reader.close()


def test_procedural_memory_retrieves_similar_tool_pattern(tmp_path):
    memory = PersistentMemory(tmp_path / "memory.sqlite3", embedding_dim=4)
    emb = np.array([1, 0, 0, 0], dtype=np.float32)
    memory.upsert_procedure(
        name="use:web_search:docs",
        description="Use web search for docs",
        tool_name="web_search",
        trigger_embedding=emb,
        success=True,
    )

    procedures = memory.retrieve_procedures(emb, top_k=1)

    assert procedures[0].tool_name == "web_search"
    assert procedures[0].success_rate == 1.0
    memory.close()


def test_delete_procedure(tmp_path):
    memory = PersistentMemory(tmp_path / "memory.sqlite3", embedding_dim=4)
    emb = np.array([1, 0, 0, 0], dtype=np.float32)
    memory.upsert_procedure(
        name="weak",
        description="weak procedure",
        tool_name="memory_recall",
        trigger_embedding=emb,
        success=False,
    )

    procedure = memory.list_procedures()[0]
    memory.delete_procedure(procedure.id)

    assert memory.list_procedures() == []
    memory.close()


def test_sync_event_log_roundtrip(tmp_path):
    memory = PersistentMemory(tmp_path / "memory.sqlite3", embedding_dim=4)
    first = memory.append_event("interaction", {"text": "hello"})
    second = memory.append_event("feedback", {"score": 1})

    events = memory.recent_events(limit=10)
    after_first = memory.recent_events(limit=10, after_id=first.id)

    assert [event.event_type for event in events] == ["interaction", "feedback"]
    assert after_first[0].id == second.id
    assert after_first[0].payload["score"] == 1
    memory.close()


def test_memory_uses_wal_for_local_concurrency(tmp_path):
    memory = PersistentMemory(tmp_path / "memory.sqlite3", embedding_dim=4)

    mode = memory.conn.execute("PRAGMA journal_mode").fetchone()[0]
    synchronous = memory.conn.execute("PRAGMA synchronous").fetchone()[0]

    assert mode.lower() == "wal"
    assert synchronous == 1
    memory.close()


def test_learning_gap_and_source_stats_roundtrip(tmp_path):
    memory = PersistentMemory(tmp_path / "memory.sqlite3", embedding_dim=4)
    gap_id = memory.record_learning_gap(
        kind="low_confidence",
        description="needs more evidence",
        severity=0.7,
        payload={"episode": "abc"},
    )
    memory.update_learning_gap_status(gap_id, "tested")
    memory.record_source_feedback("docs", accepted=True)
    memory.record_source_feedback("docs", accepted=False)

    gaps = memory.recent_learning_gaps(limit=1)
    stats = memory.list_source_stats()

    assert gaps[0].id == gap_id
    assert gaps[0].status == "tested"
    assert stats[0].source_name == "docs"
    assert stats[0].proposals == 2
    memory.close()


def test_training_file_queue_roundtrip(tmp_path):
    memory = PersistentMemory(tmp_path / "memory.sqlite3", embedding_dim=4)
    file_id = memory.add_training_file(
        name="sample.txt",
        path=str(tmp_path / "sample.txt"),
        content_type="text/plain",
        size_bytes=12,
        sha256="abc",
        status="queued",
        modality="text",
        preview="hello sdnc",
        payload={"label": "note"},
    )
    memory.update_training_file(file_id, status="done", processed_episode_id="episode-1")

    record = memory.get_training_file(file_id)
    files = memory.list_training_files()
    summary = memory.training_file_summary()

    assert record.name == "sample.txt"
    assert record.status == "done"
    assert record.processed_episode_id == "episode-1"
    assert files[0].payload["label"] == "note"
    assert summary["done"] == 1
    memory.close()


def test_cognitive_trace_roundtrip(tmp_path):
    memory = PersistentMemory(tmp_path / "memory.sqlite3", embedding_dim=4)
    trace_id = memory.store_cognitive_trace(
        trace_id="trace-1",
        episode_id="episode-1",
        input_text="question humaine",
        mode="think",
        prediction={"predicted_action": "use_tools"},
        observation={"successful_tools": ["memory_recall"]},
        attention=["input", "memory"],
        surprise=0.44,
        uncertainty=0.58,
        payload={"slot_count": 2},
    )

    traces = memory.recent_cognitive_traces(limit=1)

    assert traces[0].id == trace_id
    assert traces[0].prediction["predicted_action"] == "use_tools"
    assert traces[0].attention == ["input", "memory"]
    assert traces[0].surprise == 0.44
    memory.close()
