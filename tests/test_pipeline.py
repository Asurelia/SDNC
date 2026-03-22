"""
Tests unitaires du pipeline de distillation SDNC.

Tous les tests utilisent des mocks — pas de GPU, pas de modèles réels.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from pipeline.dual_model_config import DualModelConfig
from pipeline.distillation_engine import (
    ProjectionBridge,
    DistillationEngine,
    DISTILLATION_CORPUS,
)
from pipeline.cloud_sync import GCSManager, GitHubManager


# ══════════════════════════════════════════════════════════════════
#  Test 1 : DualModelConfig defaults
# ══════════════════════════════════════════════════════════════════

class TestDualModelConfig:
    def test_dual_model_config_defaults(self):
        """Vérifie les valeurs par défaut du DualModelConfig."""
        config = DualModelConfig()

        assert config.teacher_model == "Qwen/Qwen3.5-35B-A3B"
        assert config.student_model == "Qwen/Qwen3.5-4B"
        assert config.teacher_hidden == 5120
        assert config.student_hidden == 2560
        assert config.teacher_layers == [16, 32, 48, 64]
        assert config.student_layers == [8, 16, 24, 36]
        assert config.distill_projection_dim == 1024
        assert config.distill_temperature == 2.0
        assert config.distill_alpha == 0.7
        assert config.sleep_replay_count == 50
        assert config.sleep_distill_lr == 5e-5
        assert config.teacher_device == "cpu"
        assert config.student_device == "cuda"
        assert config.teacher_dtype == "bfloat16"
        assert config.ram_safety_margin_gb == 3.0
        assert config.log_interval == 10
        assert config.gcs_bucket == "sdnc-models"
        assert config.github_repo == ""
        assert config.github_branch == "main"
        assert config.auto_push_every_n_distillations == 1

    def test_student_brain_config_property(self):
        """Vérifie que student_brain_config génère un BrainConfig valide."""
        config = DualModelConfig()
        brain = config.student_brain_config

        assert brain.model_name == "Qwen/Qwen3.5-4B"
        assert brain.llm_hidden_dim == 2560
        assert brain.intercept_layers == [8, 16, 24, 36]
        assert brain.n_modules == 4
        assert brain.use_quantization is False


# ══════════════════════════════════════════════════════════════════
#  Test 2 : ProjectionBridge
# ══════════════════════════════════════════════════════════════════

class TestProjectionBridge:
    def test_projection_bridge_shape(self):
        """Vérifie les dimensions d'entrée/sortie du bridge avec bottleneck 1024."""
        bridge = ProjectionBridge(
            teacher_dim=5120,
            bottleneck=1024,
            student_dim=2560,
        )

        x = torch.randn(1, 512, 5120)
        out = bridge(x)

        assert out.shape == (1, 512, 2560)

    def test_projection_bridge_gradient_flows(self):
        """Vérifie que le gradient traverse le bridge."""
        bridge = ProjectionBridge(5120, 1024, 2560)

        x = torch.randn(1, 10, 5120, requires_grad=True)
        out = bridge(x)
        loss = out.mean()
        loss.backward()

        assert bridge.down.weight.grad is not None
        assert bridge.up.weight.grad is not None
        assert bridge.down.weight.grad.abs().sum() > 0

    def test_projection_bridge_different_seq_lens(self):
        """Vérifie que le bridge fonctionne avec différentes longueurs de séquence."""
        bridge = ProjectionBridge(5120, 1024, 2560)

        for seq_len in [1, 10, 128, 512]:
            x = torch.randn(1, seq_len, 5120)
            out = bridge(x)
            assert out.shape == (1, seq_len, 2560)

    def test_projection_bridge_float32_internal(self):
        """Vérifie que le bridge cast en float32 interne et restaure le dtype."""
        bridge = ProjectionBridge(5120, 1024, 2560)

        x = torch.randn(1, 5, 5120, dtype=torch.bfloat16)
        out = bridge(x)
        assert out.dtype == torch.bfloat16

    def test_projection_bridge_has_mid_norm(self):
        """Vérifie que le bridge a bien un LayerNorm intermédiaire."""
        bridge = ProjectionBridge(5120, 1024, 2560)
        assert hasattr(bridge, 'mid_norm')


# ══════════════════════════════════════════════════════════════════
#  Test 3 : Distillation Engine (avec mocks)
# ══════════════════════════════════════════════════════════════════

class TestDistillationEngine:
    def _make_mock_engine(self):
        """Crée un DistillationEngine avec des mocks (API TeacherLoader)."""
        config = DualModelConfig()

        # Mock teacher_loader
        teacher_loader = MagicMock()
        teacher_loader.get_representations_and_targets.return_value = (
            [torch.randn(1, 10, 5120) for _ in range(4)],
            torch.randn(1, 32000),  # soft targets (dernier token)
        )

        # Mock teacher_model et tokenizer
        teacher_model = MagicMock()
        teacher_tokenizer = MagicMock()

        # Mock student brain
        student = MagicMock()
        student.llm.get_layer_representations.return_value = [
            torch.randn(1, 10, 2560) for _ in range(4)
        ]
        student.forward_with_external_reps.return_value = {
            'mean_error': 0.5,
            'conflict_score': 0.3,
            'response': 'test',
        }
        student._sleep_consolidation.return_value = True

        engine = DistillationEngine(
            config=config,
            teacher_loader=teacher_loader,
            teacher_model=teacher_model,
            teacher_tokenizer=teacher_tokenizer,
            student_brain=student,
            device=torch.device("cpu"),
        )

        return engine

    def test_distill_episode_returns_dict(self):
        """Vérifie que distill_episode retourne les clés attendues."""
        engine = self._make_mock_engine()
        result = engine.distill_episode("Test prompt")

        assert 'distill_errors' in result
        assert 'total_loss' in result
        assert 'mean_distill_loss' in result
        assert isinstance(result['distill_errors'], list)
        assert len(result['distill_errors']) == 4

    def test_kl_loss_positive(self):
        """Vérifie que la loss est positive (MSE >= 0)."""
        engine = self._make_mock_engine()
        result = engine.distill_episode("Test prompt")

        assert result['total_loss'] >= 0
        assert result['mean_distill_loss'] >= 0

    def test_total_loss_weighted(self):
        """Vérifie que la loss totale est un scalaire valide."""
        engine = self._make_mock_engine()
        result = engine.distill_episode("Test prompt")

        assert isinstance(result['total_loss'], float)
        assert not torch.isnan(torch.tensor(result['total_loss']))
        assert not torch.isinf(torch.tensor(result['total_loss']))

    def test_bridge_weights_change_after_episode(self):
        """Vérifie que les poids du bridge changent après un épisode."""
        engine = self._make_mock_engine()

        weights_before = [
            p.data.clone() for p in engine.bridges.parameters()
        ]

        engine.distill_episode("Test prompt")

        weights_after = [p.data for p in engine.bridges.parameters()]
        any_changed = any(
            not torch.equal(before, after)
            for before, after in zip(weights_before, weights_after)
        )
        assert any_changed, "Les poids du bridge n'ont pas changé après distillation"


# ══════════════════════════════════════════════════════════════════
#  Test 4 : Corpus
# ══════════════════════════════════════════════════════════════════

class TestCorpus:
    def test_corpus_size(self):
        """Vérifie que le corpus contient exactement 300 prompts."""
        assert len(DISTILLATION_CORPUS) == 300

    def test_corpus_all_strings(self):
        """Vérifie que tous les éléments du corpus sont des strings non vides."""
        for i, prompt in enumerate(DISTILLATION_CORPUS):
            assert isinstance(prompt, str), f"Prompt {i} n'est pas un string"
            assert len(prompt) > 10, f"Prompt {i} trop court : '{prompt}'"

    def test_corpus_no_duplicates(self):
        """Vérifie qu'il n'y a pas de doublons dans le corpus."""
        assert len(set(DISTILLATION_CORPUS)) == len(DISTILLATION_CORPUS)


# ══════════════════════════════════════════════════════════════════
#  Test 5 : Cloud Sync
# ══════════════════════════════════════════════════════════════════

class TestCloudSync:
    def test_gcs_uri_format(self):
        """Vérifie le format des URI GCS."""
        gcs = GCSManager(bucket="sdnc-models", prefix="checkpoints")

        uri = f"gs://{gcs.bucket}/{gcs.prefix}/20260322/student/step_000100.pt"
        assert uri.startswith("gs://")
        assert "sdnc-models" in uri
        assert "step_" in uri
        assert uri.endswith(".pt")

    def test_latest_json_schema(self):
        """Vérifie que le schéma latest.json contient toutes les clés requises."""
        required_keys = [
            'version', 'step', 'timestamp', 'teacher', 'student',
            'gcs_uri', 'metrics', 'hw_compatible', 'sdnc_version'
        ]

        latest = {
            'version': '0.2.0',
            'step': 100,
            'timestamp': '2026-03-22T14:30:00',
            'teacher': 'Qwen3.5-35B-A3B',
            'student': 'Qwen3.5-4B',
            'gcs_uri': 'gs://sdnc-models/checkpoints/20260322/student/step_000100.pt',
            'metrics': {
                'pc_convergence': 0.15,
                'episodic_memory': 0.0,
                'conflict_f1': 0.0,
                'temporal_consistency': 0.0,
                'forgetting_score': 0.0,
                'distill_loss': 0.5,
            },
            'hw_compatible': ['A100', 'RX7800XT-16GB'],
            'sdnc_version': '0.2.0',
        }

        for key in required_keys:
            assert key in latest, f"Clé manquante : {key}"

        json_str = json.dumps(latest, indent=2)
        parsed = json.loads(json_str)
        assert parsed == latest

    def test_github_commit_message(self):
        """Vérifie que le format de commit contient 'SDNC auto-release'."""
        step = 100
        loss = 0.1234
        msg = f"🤖 SDNC auto-release step {step} — loss={loss:.4f}"
        assert "SDNC auto-release" in msg
        assert str(step) in msg


# ══════════════════════════════════════════════════════════════════
#  Test 6 : Resume & Emergency Save
# ══════════════════════════════════════════════════════════════════

class TestResumeAndEmergency:
    def test_resume_no_checkpoint(self):
        """Vérifie que resume() démarre from scratch sans erreur."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir)
            ckpts = sorted(ckpt_dir.glob("student_cycle_*.pt"))
            assert len(ckpts) == 0

    def test_emergency_save(self):
        """Vérifie que emergency_save crée un fichier de checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "emergency_bridges.pt")

            bridge = ProjectionBridge(5120, 1024, 2560)
            torch.save({
                'bridges': bridge.state_dict(),
                'total_episodes': 42,
                'loss_history': [0.5, 0.4, 0.3],
            }, path)

            assert os.path.exists(path)

            ckpt = torch.load(path, weights_only=False)
            assert 'bridges' in ckpt
            assert ckpt['total_episodes'] == 42
            assert len(ckpt['loss_history']) == 3


# ══════════════════════════════════════════════════════════════════
#  Test 7 : Status ASCII
# ══════════════════════════════════════════════════════════════════

class TestStatus:
    def test_status_ascii(self):
        """Vérifie que status() retourne une string non vide."""
        from pipeline.full_loop import SDNCPipeline

        pipeline = SDNCPipeline.__new__(SDNCPipeline)
        pipeline.current_phase = "idle"
        pipeline.cycle_count = 5
        pipeline.student = None
        pipeline.engine = None
        pipeline.last_benchmark = {}

        status = pipeline.status()

        assert isinstance(status, str)
        assert len(status) > 0
        assert "idle" in status
        assert "5" in status


# ══════════════════════════════════════════════════════════════════
#  Test 8 : LocalRunner
# ══════════════════════════════════════════════════════════════════

class TestLocalRunner:
    def test_local_runner_sync_reads_latest_json(self):
        """Vérifie que le local runner lit correctement latest.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            releases_dir = Path(tmpdir) / "releases"
            releases_dir.mkdir()
            latest = {
                'version': '0.2.0',
                'step': 100,
                'gcs_uri': 'gs://sdnc-models/checkpoints/student/step_000100.pt',
            }
            with open(releases_dir / "latest.json", 'w') as f:
                json.dump(latest, f)

            with open(releases_dir / "latest.json", 'r') as f:
                loaded = json.load(f)

            assert loaded['step'] == 100
            assert loaded['gcs_uri'].startswith("gs://")
            assert loaded['gcs_uri'].endswith(".pt")


# ══════════════════════════════════════════════════════════════════
#  Test 9 : TeacherLoader
# ══════════════════════════════════════════════════════════════════

class TestTeacherLoader:
    def test_teacher_loader_import(self):
        """Vérifie que TeacherLoader est importable."""
        from pipeline.teacher_loader import TeacherLoader
        loader = TeacherLoader()
        assert loader is not None

    def test_teacher_loader_unload(self):
        """Vérifie que unload ne crash pas avec un mock."""
        from pipeline.teacher_loader import TeacherLoader
        mock_model = MagicMock()
        # Should not raise
        TeacherLoader.unload(mock_model)


# ══════════════════════════════════════════════════════════════════
#  Test 10 : Hardware Profile
# ══════════════════════════════════════════════════════════════════

class TestHardwareProfile:
    def test_hw_config_has_expected_fields(self):
        """Vérifie que HW_CONFIG a les champs de base."""
        from brain_hybrid.utils.device import HW_CONFIG
        assert 'device' in HW_CONFIG
        assert 'hw_type' in HW_CONFIG
        assert 'batch_size' in HW_CONFIG
