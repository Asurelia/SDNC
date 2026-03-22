import torch
from brain_hybrid.core.brain_module import BrainModule
from brain_hybrid.core.predictive_coding import PredictiveCodingLayer, HierarchicalPC
from brain_hybrid.core.stdp import STDPLearning


def test_error_computation():
    """L'erreur doit être actual - predicted avec les bonnes shapes."""
    module = BrainModule(input_dim=256, hidden_dim=32)
    pc = PredictiveCodingLayer(module)

    actual = torch.randn(1, 5, 256)
    predicted = torch.randn(1, 5, 256)
    error = pc.compute_error(actual, predicted)
    assert error.shape == (1, 5, 256)


def test_precision_positive():
    """La précision doit toujours être positive."""
    module = BrainModule(input_dim=256, hidden_dim=32)
    pc = PredictiveCodingLayer(module)

    actual = torch.randn(1, 5, 256)
    predicted = torch.randn(1, 5, 256)
    pc.compute_error(actual, predicted)
    assert pc.precision > 0, f"Précision négative: {pc.precision}"


def test_hierarchical_forward():
    """4 layers → 3 erreurs (entre couches adjacentes)."""
    modules = [BrainModule(input_dim=256, hidden_dim=32) for _ in range(4)]
    hpc = HierarchicalPC(modules, pc_lr=0.001)

    layer_reps = [torch.randn(1, 5, 256) for _ in range(4)]
    errors = hpc.forward(layer_reps)
    assert len(errors) == 3, f"Attendu 3 erreurs, reçu {len(errors)}"
    for e in errors:
        assert e.shape == (1, 5, 256)


def test_pc_precisions():
    """get_precisions() doit retourner 4 valeurs positives."""
    modules = [BrainModule(input_dim=256, hidden_dim=32) for _ in range(4)]
    hpc = HierarchicalPC(modules, pc_lr=0.001)

    layer_reps = [torch.randn(1, 5, 256) for _ in range(4)]
    hpc.forward(layer_reps)
    precisions = hpc.get_precisions()
    assert len(precisions) == 4
    # Les 3 premiers doivent avoir été calculés, le dernier reste à 1.0
    for p in precisions[:3]:
        assert p > 0


def test_pc_update_changes_weights():
    """Les poids CfC doivent changer après pc_update."""
    module = BrainModule(input_dim=256, hidden_dim=32)
    pc = PredictiveCodingLayer(module, pc_lr=0.01)
    stdp = STDPLearning()

    # Forward pour initialiser les traces
    x = torch.randn(1, 5, 256)
    module(x)

    # Compute error
    actual = torch.randn(1, 5, 256)
    predicted = torch.randn(1, 5, 256)
    pc.compute_error(actual, predicted)

    # Sauvegarder les poids avant
    weights_before = {}
    for name, p in module.cfc.named_parameters():
        if p.requires_grad and len(p.shape) == 2:
            weights_before[name] = p.data.clone()

    # PC update
    pc.pc_update(stdp, dopamine_signal=0.8)

    # Vérifier que les poids ont changé
    changed = False
    for name, p in module.cfc.named_parameters():
        if name in weights_before:
            if not torch.allclose(p.data, weights_before[name]):
                changed = True
                break
    assert changed, "Les poids CfC n'ont pas changé après pc_update"
