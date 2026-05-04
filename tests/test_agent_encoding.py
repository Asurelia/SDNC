import numpy as np

from sdnc.agent.encoding import HashingExperienceEncoder


def test_hashing_encoder_is_stable_and_normalized():
    encoder = HashingExperienceEncoder(dim=64)
    left = encoder.encode("chercher docs liquid neural networks", {"mode": "research"})
    right = encoder.encode("chercher docs liquid neural networks", {"mode": "research"})

    assert np.allclose(left, right)
    assert np.isclose(np.linalg.norm(left), 1.0)


def test_hashing_encoder_context_changes_vector():
    encoder = HashingExperienceEncoder(dim=64)
    left = encoder.encode("same text", {"tool": "web"})
    right = encoder.encode("same text", {"tool": "file"})

    assert not np.allclose(left, right)
