from brain_hybrid.utils.step_scheduler import StepScheduler


def test_snn_every_step():
    """SNN doit s'activer à chaque step."""
    s = StepScheduler(snn_freq=1)
    for i in range(1, 20):
        s.step()
        assert s.should_update("snn"), f"SNN inactif au step {i}"


def test_qwen_rare():
    """Qwen ne s'active que tous les 100 steps."""
    s = StepScheduler(qwen_freq=100)
    for i in range(1, 100):
        s.step()
        assert not s.should_update("qwen"), f"Qwen actif trop tôt au step {i}"
    s.step()  # step 100
    assert s.should_update("qwen"), "Qwen inactif au step 100"


def test_arousal_boost():
    """Après arousal_boost, les fréquences doivent doubler (freqs divisées par 2)."""
    s = StepScheduler(cfc_freq=10, qwen_freq=100)
    s.arousal_boost()
    assert s.current_freqs["cfc"] == 5, f"cfc_freq={s.current_freqs['cfc']}, attendu 5"
    assert s.current_freqs["qwen"] == 50, f"qwen_freq={s.current_freqs['qwen']}, attendu 50"

    # Après 50 steps, les fréquences doivent revenir à la normale
    for _ in range(50):
        s.step()
    assert s.current_freqs["cfc"] == 10
    assert s.current_freqs["qwen"] == 100


def test_phase_detection():
    """get_phase() doit retourner le bon résultat."""
    s = StepScheduler()
    # Step 0 (initial)
    assert s.get_phase() in ("theta", "gamma", "idle")

    # Step 6 → theta
    for _ in range(6):
        s.step()
    assert s.get_phase() == "theta", f"Phase au step 6: {s.get_phase()}"

    # Step 8 → gamma (pair, pas multiple de 6)
    for _ in range(2):
        s.step()
    assert s.get_phase() == "gamma", f"Phase au step 8: {s.get_phase()}"

    # Step 9 → idle (impair, pas multiple de 6)
    s.step()
    assert s.get_phase() == "idle", f"Phase au step 9: {s.get_phase()}"


def test_save_load():
    """L'état du scheduler doit survivre au save/load."""
    s = StepScheduler()
    for _ in range(42):
        s.step()
    s.arousal_boost()
    state = s.save_state()

    s2 = StepScheduler()
    s2.load_state(state)
    assert s2.global_step == 42
    assert s2._arousal_remaining == s._arousal_remaining
