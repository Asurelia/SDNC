import numpy as np

from sdnc.agent.config import AutonomousConfig
from sdnc.agent.encoding import HashingExperienceEncoder
from sdnc.agent.multimodal import LocalMultimodalEncoder, ModalitySample
from sdnc.agent.perception import PerceptionBus
from sdnc.agent.system import InteractionLearningSystem


def test_perception_bus_binds_cross_modal_event():
    bus = PerceptionBus(LocalMultimodalEncoder(dim=32), HashingExperienceEncoder(dim=32))
    image = np.zeros((6, 6, 3), dtype=np.uint8)
    image[:, :, 1] = 255

    event = bus.observe(
        [
            ModalitySample("text", text="bouton vert confirme", label="ui", source="fixture", sample_id="a"),
            ModalitySample("image", content=image, text="bouton vert", label="ui", source="fixture", sample_id="a"),
        ],
        context={"scene": "test"},
    )

    assert event.modalities == ["text", "image"]
    assert event.sample_ids == ["a"]
    assert event.source == "fixture"
    assert event.binding_score > 0
    assert event.reliability > 0.5
    assert np.isclose(np.linalg.norm(event.embedding), 1.0)
    assert "sensory_event" in event.summary


def test_system_observe_persists_sensory_binding(tmp_path):
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
        result = system.observe(
            [
                ModalitySample("text", text="le voyant devient rouge", source="fixture", sample_id="s1"),
                ModalitySample(
                    "image",
                    content=np.ones((4, 4, 3), dtype=np.uint8) * 255,
                    text="voyant rouge",
                    source="fixture",
                    sample_id="s1",
                ),
            ],
            learn=True,
        )
        bindings = system.recent_sensory_bindings(limit=5)
        events = system.recent_events(limit=5)

        assert result.learned is True
        assert result.metadata["sensory_event"]["modalities"] == ["text", "image"]
        assert bindings[0].id == result.metadata["sensory_event"]["id"]
        assert bindings[0].episode_id == result.episode_id
        assert any(event.event_type == "observation" for event in events)
    finally:
        system.close()
