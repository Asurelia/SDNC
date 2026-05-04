from sdnc.agent.budget import BudgetManager
from sdnc.agent.config import AutonomousConfig
from sdnc.agent.context_lod import ContextLODCompressor
from sdnc.agent.encoding import HashingExperienceEncoder
from sdnc.agent.system import InteractionLearningSystem


def test_budget_manager_modes_and_vram_snapshot():
    config = AutonomousConfig(input_dim=32, n_circuits=20)
    manager = BudgetManager(config)

    fast = manager.choose("bonjour", explicit_mode="fast")
    max_budget = manager.choose("analyse architecture profonde", explicit_mode="max")
    snapshot = manager.snapshot(max_budget)

    assert fast.mode == "fast"
    assert max_budget.mode == "max"
    assert max_budget.max_hot_experts > fast.max_hot_experts
    assert snapshot.hot_vram_gb <= snapshot.usable_vram_gb
    assert snapshot.within_budget is True


def test_context_lod_compresses_long_context():
    encoder = HashingExperienceEncoder(dim=32)
    config = AutonomousConfig(input_dim=32, n_circuits=20)
    budget = BudgetManager(config).choose("", explicit_mode="fast")
    compressor = ContextLODCompressor(encoder)
    text = "\n\n".join(
        f"section {index}: objectif important test bug mémoire compression sparse"
        for index in range(40)
    )

    packet = compressor.compress(text, {}, budget)

    assert packet.original_chars == len(text)
    assert len(packet.segments) <= budget.max_context_segments
    assert len(packet.prototypes) >= 1
    assert packet.compression_ratio < 1.0
    assert packet.estimated_tokens_saved > 0


def test_interaction_metadata_exposes_budget_and_context(tmp_path):
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
        result = system.interact("analyse pourquoi ce test échoue", context={"mode": "think"})

        assert result.metadata["cognitive_budget"]["mode"] == "think"
        assert result.metadata["context_lod"]["segment_count"] >= 1
        assert result.metadata["resource_budget"]["within_budget"] is True
    finally:
        system.close()
