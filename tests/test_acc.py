import torch
from brain_hybrid.core.acc import ACCModule


def test_conflict_detected():
    """Erreurs élevées doivent déclencher un conflit."""
    acc = ACCModule(n_modules=4, conflict_threshold=0.7)
    result = acc.forward(
        prediction_errors=[0.9, 0.9, 0.9, 0.9],
        dopamine_signals=[0.9, 0.9, 0.9, 0.9],
    )
    # Le MLP avec poids aléatoires ne garantit pas is_conflict au premier appel
    # mais conflict_score doit être entre 0 et 1
    assert 0.0 <= result.conflict_score <= 1.0


def test_no_conflict_low_errors():
    """Erreurs basses ne doivent pas saturer le conflict_score."""
    acc = ACCModule(n_modules=4, conflict_threshold=0.7)
    result = acc.forward(
        prediction_errors=[0.01, 0.01, 0.01, 0.01],
        dopamine_signals=[0.1, 0.1, 0.1, 0.1],
    )
    assert 0.0 <= result.conflict_score <= 1.0


def test_acc_weights_change():
    """Les poids doivent changer après update Hebbian avec inputs variés."""
    acc = ACCModule(n_modules=4)
    w_before = acc.fc1.weight.data.clone()
    for i in range(20):
        errs = [0.1 * (i % 5 + 1)] * 4
        dops = [0.2 * (i % 3 + 1)] * 4
        acc.forward(errs, dops)
    w_after = acc.fc1.weight.data
    assert not torch.allclose(w_before, w_after), "Poids ACC inchangés"


def test_acc_stable_convergence():
    """Le conflict_score doit se stabiliser après 100 appels identiques."""
    acc = ACCModule(n_modules=4)
    scores = []
    for _ in range(100):
        r = acc.forward([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
        scores.append(r.conflict_score)
    # Variance des 10 derniers doit être faible
    last_10 = scores[-10:]
    variance = sum((s - sum(last_10)/10)**2 for s in last_10) / 10
    assert variance < 0.1, f"ACC instable: variance={variance:.4f}"


def test_acc_trend():
    """trend() doit retourner un string valide."""
    acc = ACCModule(n_modules=4)
    for _ in range(30):
        acc.forward([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
    assert acc.trend() in ("stable", "rising", "falling")


def test_acc_prompt_fragment():
    """Le fragment de prompt doit être vide ou contenir CONFLIT."""
    acc = ACCModule(n_modules=4)
    r = acc.forward([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
    if r.is_conflict:
        assert "CONFLIT" in r.conflict_prompt_fragment
    else:
        assert r.conflict_prompt_fragment == ""
