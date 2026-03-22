"""
Tests unitaires pour DistilledInputLayer et intégration bridge.
"""

import torch
import pytest
from unittest.mock import MagicMock

from brain_hybrid.core.distilled_input import DistilledInputLayer
from brain_hybrid.config import BrainConfig
from pipeline.distillation_engine import ProjectionBridge


class TestDistilledInputLayer:

    def test_alpha_init(self):
        """Vérifie que les alphas sont initialisés à 0.8."""
        layer = DistilledInputLayer(n_layers=4, hidden_dim=2560)
        for alpha in layer.alphas:
            assert abs(alpha.item() - 0.8) < 0.01

    def test_enriched_shape(self):
        """Vérifie les dimensions de sortie."""
        layer = DistilledInputLayer(n_layers=4, hidden_dim=2560)
        student = [torch.randn(1, 10, 2560) for _ in range(4)]
        teacher = [torch.randn(1, 10, 2560) for _ in range(4)]
        enriched, stats = layer(student, teacher)
        assert len(enriched) == 4
        assert enriched[0].shape == (1, 10, 2560)

    def test_alpha_clamp(self):
        """Vérifie que les alphas sont clampés [0, 1]."""
        layer = DistilledInputLayer(n_layers=4, hidden_dim=2560)
        layer.alphas[0].data = torch.tensor(1.5)
        layer.alphas[1].data = torch.tensor(-0.3)
        student = [torch.randn(1, 5, 2560) for _ in range(4)]
        teacher = [torch.randn(1, 5, 2560) for _ in range(4)]
        enriched, stats = layer(student, teacher)
        assert stats["alphas"][0] <= 1.0
        assert stats["alphas"][1] >= 0.0

    def test_teacher_influence(self):
        """Vérifie le calcul de l'influence teacher."""
        layer = DistilledInputLayer(n_layers=4, hidden_dim=2560, alpha_init=0.8)
        influence = layer.get_teacher_influence()
        assert abs(influence - 0.2) < 0.01  # 1 - 0.8 = 0.2

    def test_seq_len_mismatch(self):
        """Vérifie que des seq_len différentes sont gérées."""
        layer = DistilledInputLayer(n_layers=2, hidden_dim=256)
        student = [torch.randn(1, 10, 256), torch.randn(1, 10, 256)]
        teacher = [torch.randn(1, 7, 256), torch.randn(1, 12, 256)]
        enriched, stats = layer(student, teacher)
        assert enriched[0].shape[1] == 7   # min(10, 7)
        assert enriched[1].shape[1] == 10  # min(10, 12)

    def test_stats_structure(self):
        """Vérifie la structure des stats retournées."""
        layer = DistilledInputLayer(n_layers=4, hidden_dim=2560)
        student = [torch.randn(1, 5, 2560) for _ in range(4)]
        teacher = [torch.randn(1, 5, 2560) for _ in range(4)]
        _, stats = layer(student, teacher)
        assert "alphas" in stats
        assert "gates" in stats
        assert "cosine_per_layer" in stats
        assert len(stats["alphas"]) == 4
        assert len(stats["gates"]) == 4
        assert len(stats["cosine_per_layer"]) == 4


class TestProjectionBridgeIntegration:

    def test_bridge_frozen_after_load(self):
        """Vérifie que le bridge est gelé après chargement."""
        bridge = ProjectionBridge(5120, 1024, 2560)
        for p in bridge.parameters():
            p.requires_grad = False
        for p in bridge.parameters():
            assert not p.requires_grad

    def test_bridge_shape_5120_to_2560(self):
        """Vérifie les dimensions 5120→2560."""
        bridge = ProjectionBridge(5120, 1024, 2560)
        x = torch.randn(1, 10, 5120)
        out = bridge(x)
        assert out.shape == (1, 10, 2560)


class TestBrainConfigDistillation:

    def test_config_has_bridge_fields(self):
        """Vérifie les nouveaux champs de config."""
        config = BrainConfig()
        assert hasattr(config, 'bridge_checkpoint_path')
        assert hasattr(config, 'use_distilled_input')
        assert hasattr(config, 'distilled_input_alpha_init')
        assert config.bridge_checkpoint_path == ""
        assert config.use_distilled_input is True
        assert config.distilled_input_alpha_init == 0.8

    def test_config_without_bridge(self):
        """Vérifie que use_distilled_input=False désactive le module."""
        config = BrainConfig(use_distilled_input=False)
        assert config.use_distilled_input is False


class TestAblationStudy:

    def test_ablation_import(self):
        """Vérifie que AblationStudy est importable."""
        from brain_hybrid.eval.ablation import AblationStudy
        study = AblationStudy()
        assert study is not None
        assert len(study.test_prompts) == 100

    def test_ablation_prompts(self):
        """Vérifie que les prompts de test sont variés."""
        from brain_hybrid.eval.ablation import AblationStudy
        study = AblationStudy()
        unique = set(study.test_prompts)
        assert len(unique) >= 10  # au moins 10 prompts uniques
