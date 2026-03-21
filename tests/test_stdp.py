import torch
from brain_hybrid.core.stdp import STDPLearning


def test_poids_changent():
    """Les poids doivent changer après apply()."""
    stdp = STDPLearning()
    w = torch.randn(32, 32)
    w_orig = w.clone()
    pre = torch.randn(1, 10, 64)
    post = torch.randn(1, 10, 32)
    stdp.apply(w, pre, post)
    assert not torch.allclose(w, w_orig), "Les poids n'ont pas changé"


def test_changement_stable():
    """Les changements de poids doivent être < 1% de la norme."""
    stdp = STDPLearning()
    w = torch.randn(32, 32)
    w_orig = w.clone()
    pre = torch.randn(1, 10, 64)
    post = torch.randn(1, 10, 32)
    stdp.apply(w, pre, post)
    rel_change = (w - w_orig).abs().mean() / w_orig.abs().mean()
    assert rel_change < 0.01, f"Changement trop grand : {rel_change:.4f}"


def test_dopamine_clamp():
    """Le signal dopamine doit rester entre 0 et 1."""
    stdp = STDPLearning()
    for err in [0.0, 0.1, 0.5, 1.0, 10.0, -5.0]:
        stdp.update_dopamine(err)
        assert 0.0 <= stdp.dopamine_signal <= 1.0
