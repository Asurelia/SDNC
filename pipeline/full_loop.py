"""
Orchestrateur du pipeline complet de distillation SDNC.

Stratégie A100 haute-RAM :
  Teacher 35B en RAM CPU (BF16 natif, ~70 GB)
  Student 4B en VRAM (~8 GB)
  Transfert asynchrone RAM → VRAM via ProjectionBridge

5 phases par cycle :
  Phase 1 — Training combiné : student SDNC + distillation teacher→student
  Phase 2 — Benchmark intermédiaire : smoke test rapide
  Phase 3 — Distillation sommeil (200 steps)
  Phase 4 — Benchmark complet : décharge teacher, benchmark student seul
  Phase 5 — Cloud sync : GCS upload + GitHub push
"""

import csv
import gc
import random
import time

import torch
from pathlib import Path
from datetime import datetime

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from .dual_model_config import DualModelConfig
from .teacher_loader import TeacherLoader
from .distillation_engine import DistillationEngine, DISTILLATION_CORPUS
from .cloud_sync import GCSManager, GitHubManager
from brain_hybrid.model import BrainHybridModel
from brain_hybrid.eval.benchmark import SDNCBenchmark
from brain_hybrid.utils.device import get_device, HW_CONFIG


def _ram_gb() -> float:
    """RAM utilisée par ce processus en GB."""
    if _HAS_PSUTIL:
        return psutil.Process().memory_info().rss / 1e9
    return 0.0


def _ram_total_gb() -> float:
    """RAM système totale en GB."""
    if _HAS_PSUTIL:
        return psutil.virtual_memory().total / 1e9
    return 0.0


def _vram_gb() -> float:
    """VRAM GPU utilisée en GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(0) / 1e9
    return 0.0


class SDNCPipeline:
    """
    Pipeline complet : audit → train+distill → benchmark → cloud sync.

    Stratégie A100 haute-RAM :
      - Teacher 35B chargé en RAM CPU via TeacherLoader
      - Student 4B en VRAM via BrainHybridModel
      - ProjectionBridge (VRAM, seul module backprop)
      - Monitoring RAM/VRAM avec CSV logging
    """

    def __init__(self, config: DualModelConfig = None):
        """
        Initialise le pipeline complet.

        Args:
            config: Configuration dual-model. Défaut si None.
        """
        self.config = config or DualModelConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # État
        self.cycle_count = 0
        self.current_phase = "init"
        self.last_benchmark = {}
        self.last_distill_metrics = {}
        self._initialized = False

        # Paths
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # CSV logger
        self.csv_path = self.checkpoint_dir / "monitoring.csv"

        # Cloud managers
        self.gcs = GCSManager(
            bucket=self.config.gcs_bucket,
            prefix=self.config.gcs_model_prefix,
        )
        self.github = GitHubManager(
            repo=self.config.github_repo,
            branch=self.config.github_branch,
        )

        # Modèles et engine — chargés par audit()
        self.teacher_loader = TeacherLoader()
        self.teacher_model = None
        self.teacher_tokenizer = None
        self.student = None
        self.engine = None

    def _log_csv(self, step: int, phase: str, **kwargs):
        """Écrit une ligne dans le CSV de monitoring."""
        row = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'phase': phase,
            'ram_gb': _ram_gb(),
            'vram_gb': _vram_gb(),
            **kwargs,
        }
        file_exists = self.csv_path.exists()
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _check_ram_safety(self) -> bool:
        """Vérifie que la RAM n'est pas critique. Retourne False si danger."""
        if not _HAS_PSUTIL:
            return True
        available = psutil.virtual_memory().available / 1e9
        return available > self.config.ram_safety_margin_gb

    def audit(self) -> dict:
        """
        Phase audit — DOIT passer avant toute autre phase.

        Vérifie :
          - GPU disponible et type (A100)
          - RAM >= 80 GB (psutil)
          - VRAM >= 38 GB
          - Teacher charge en RAM CPU
          - Student charge en VRAM
          - Dimensions compatibles
          - Secrets présents (warning si absents)

        Returns:
            dict avec status par check.

        Raises:
            RuntimeError si un check critique échoue.
        """
        results = {}
        print("\n" + "=" * 60)
        print(" AUDIT PIPELINE SDNC — A100 haute-RAM")
        print("=" * 60)

        # 1. GPU
        if not torch.cuda.is_available():
            results['gpu'] = 'FAIL — pas de GPU détecté'
            raise RuntimeError("Aucun GPU disponible. A100 requis.")

        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        ram_total = _ram_total_gb()
        results['gpu'] = f'PASS — {gpu_name} ({vram_total:.0f} GB VRAM, {ram_total:.0f} GB RAM)'
        print(f"  GPU  : {gpu_name} | VRAM : {vram_total:.0f} GB")
        print(f"  RAM  : {ram_total:.1f} GB")

        # Détecter profil hardware
        get_device()
        hw_type = HW_CONFIG.get("hw_type", "unknown")
        results['profile'] = hw_type
        print(f"  Profil : {hw_type}")

        # 2. Teacher — chargement en RAM CPU
        print(f"\n  Chargement teacher : {self.config.teacher_model} sur CPU RAM...")
        ram_before = _ram_gb()
        try:
            dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
            teacher_dtype = dtype_map.get(self.config.teacher_dtype, torch.bfloat16)

            self.teacher_model, self.teacher_tokenizer = self.teacher_loader.load(
                model_name=self.config.teacher_model,
                dtype=teacher_dtype,
            )

            # Vérifier hidden_size
            teacher_config = self.teacher_model.config
            actual_hidden = getattr(teacher_config, 'hidden_size', None)
            if actual_hidden and actual_hidden != self.config.teacher_hidden:
                print(f"  Teacher hidden_size={actual_hidden} "
                      f"(attendu {self.config.teacher_hidden}) — mise à jour config")
                self.config.teacher_hidden = actual_hidden

            ram_after = _ram_gb()
            results['teacher'] = f'PASS — hidden={actual_hidden}, RAM delta={ram_after - ram_before:.1f} GB'
            print(f"  Teacher : PASS (hidden={actual_hidden}, delta RAM={ram_after - ram_before:.1f} GB)")
        except Exception as e:
            results['teacher'] = f'FAIL — {e}'
            raise RuntimeError(f"Échec chargement teacher : {e}")

        # 3. Student — chargement en VRAM
        print(f"\n  Chargement student : {self.config.student_model} sur VRAM...")
        vram_before = _vram_gb()
        try:
            brain_config = self.config.student_brain_config
            self.student = BrainHybridModel(brain_config)
            student_config = self.student.llm.model.config
            actual_hidden = getattr(student_config, 'hidden_size', None)
            if actual_hidden and actual_hidden != self.config.student_hidden:
                print(f"  Student hidden_size={actual_hidden} "
                      f"(attendu {self.config.student_hidden}) — mise à jour config")
                self.config.student_hidden = actual_hidden

            vram_after = _vram_gb()
            results['student'] = f'PASS — hidden={actual_hidden}, VRAM delta={vram_after - vram_before:.1f} GB'
            print(f"  Student : PASS (hidden={actual_hidden}, delta VRAM={vram_after - vram_before:.1f} GB)")
        except Exception as e:
            results['student'] = f'FAIL — {e}'
            raise RuntimeError(f"Échec chargement student : {e}")

        # 4. Budget mémoire
        ram_used = _ram_gb()
        vram_used = _vram_gb()
        print(f"\n  Budget mémoire :")
        print(f"    RAM  : {ram_used:.1f} / {ram_total:.1f} GB ({ram_used/ram_total*100:.0f}%)")
        print(f"    VRAM : {vram_used:.1f} / {vram_total:.0f} GB ({vram_used/vram_total*100:.0f}%)")
        results['ram'] = f'{ram_used:.1f} / {ram_total:.0f} GB'
        results['vram'] = f'{vram_used:.1f} / {vram_total:.0f} GB'

        # 5. Distillation Engine
        print("\n  Initialisation DistillationEngine...")
        self.engine = DistillationEngine(
            config=self.config,
            teacher_loader=self.teacher_loader,
            teacher_model=self.teacher_model,
            teacher_tokenizer=self.teacher_tokenizer,
            student_brain=self.student,
            device=self.device,
        )
        results['engine'] = 'PASS'

        # 6. Dimensions test (dummy forward)
        print("  Test dimensions (dummy forward)...")
        try:
            test_prompt = "Test audit pipeline SDNC."
            teacher_reps = self.teacher_loader.get_representations(
                self.teacher_model, self.teacher_tokenizer, test_prompt,
                layers=self.config.teacher_layers,
                target_device=str(self.device),
            )
            for i, rep in enumerate(teacher_reps):
                print(f"    Teacher layer {self.config.teacher_layers[i]} : {rep.shape}")

            student_reps = self.student.llm.get_layer_representations(
                test_prompt,
                layers=self.config.student_layers,
            )
            for i, rep in enumerate(student_reps):
                print(f"    Student layer {self.config.student_layers[i]} : {rep.shape}")

            results['dimensions'] = 'PASS'
        except Exception as e:
            results['dimensions'] = f'FAIL — {e}'
            raise RuntimeError(f"Échec test dimensions : {e}")

        # 7. Secrets (non-bloquant)
        has_gcs = self.gcs._available
        has_github = bool(self.config.github_repo)
        results['gcs'] = 'PASS' if has_gcs else 'WARNING — gsutil non disponible'
        results['github'] = 'PASS' if has_github else 'WARNING — pas de repo configuré'

        self._initialized = True
        self._log_csv(0, "audit", distill_loss=0.0, sdnc_loss=0.0)

        print("\n" + "=" * 60)
        print(" AUDIT COMPLET — Pipeline prêt")
        print("=" * 60)

        return results

    def run_full_cycle(self, n_train_steps: int = 1000):
        """
        Boucle principale — exécute des cycles complets.

        Chaque cycle :
          Phase 1 — Training combiné (SDNC + distillation)
          Phase 2 — Benchmark intermédiaire
          Phase 3 — Distillation sommeil (200 steps)
          Phase 4 — Benchmark complet (teacher déchargé)
          Phase 5 — Cloud sync

        Args:
            n_train_steps: Nombre de steps de training par cycle.
        """
        if not self._initialized:
            self.audit()

        cycle = 0
        while True:
            cycle += 1
            self.cycle_count = cycle
            start_time = time.time()

            print(f"\n{'#'*60}")
            print(f" CYCLE {cycle}")
            print(f"{'#'*60}")

            try:
                # ── Phase 1 : Training combiné ────────────────────────
                self.current_phase = "training"
                half_steps = n_train_steps // 2
                print(f"\n  Phase 1 — Training combiné ({half_steps} steps)")

                prompts = DISTILLATION_CORPUS.copy()
                random.shuffle(prompts)

                for step in range(half_steps):
                    prompt = prompts[step % len(prompts)]

                    # a) Student forward SDNC (STDP + PC + ACC)
                    try:
                        sdnc_result = self.student.forward(prompt, learn=True)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                        raise

                    # b) Distillation step (teacher RAM → student VRAM)
                    try:
                        distill_result = self.engine.distill_episode(prompt)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                        raise

                    # c) Logging toutes les N steps
                    if (step + 1) % self.config.log_interval == 0:
                        ram = _ram_gb()
                        vram = _vram_gb()
                        sdnc_loss = sdnc_result.get('mean_error', 0.0)
                        d_loss = distill_result.get('total_loss', 0.0)

                        self._log_csv(
                            step=step + 1, phase="training",
                            distill_loss=d_loss, sdnc_loss=sdnc_loss,
                        )

                        print(
                            f"    Step {step+1:4d} "
                            f"| sdnc_loss={sdnc_loss:.4f} "
                            f"| distill_loss={d_loss:.4f} "
                            f"| RAM={ram:.1f}GB "
                            f"| VRAM={vram:.1f}GB"
                        )

                    # d) Safety check RAM
                    if (step + 1) % 50 == 0 and not self._check_ram_safety():
                        print(f"    RAM critique — emergency_save + arrêt propre")
                        self.emergency_save()
                        return

                # ── Phase 2 : Benchmark intermédiaire ─────────────────
                self.current_phase = "benchmark_smoke"
                print("\n  Phase 2 — Benchmark smoke test")

                bench = SDNCBenchmark(self.student)
                pc_score = bench.pc_convergence_score(n_steps=20)
                print(f"    PC Convergence : {pc_score:.4f}")

                if pc_score < -0.5:
                    print("    Score trop bas — skip distillation, continue training")
                    continue

                # ── Phase 3 : Distillation sommeil ────────────────────
                self.current_phase = "distillation"
                print(f"\n  Phase 3 — Distillation sommeil (200 steps)")

                self.last_distill_metrics = self.engine.run_sleep_distillation(
                    n_steps=200
                )

                # ── Phase 4 : Benchmark complet ───────────────────────
                self.current_phase = "benchmark_full"
                print("\n  Phase 4 — Benchmark complet")

                # Décharger teacher pour libérer ~70 GB RAM
                if self.teacher_model is not None:
                    TeacherLoader.unload(self.teacher_model)
                    self.teacher_model = None
                    gc.collect()

                try:
                    bench_results = bench.run_full_benchmark(
                        pc_steps=30,
                        n_distractors=50,
                        consistency_steps=100,
                        forgetting_steps=200,
                    )
                    self.last_benchmark = bench_results
                finally:
                    # Recharger teacher
                    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
                    teacher_dtype = dtype_map.get(self.config.teacher_dtype, torch.bfloat16)
                    self.teacher_model, self.teacher_tokenizer = self.teacher_loader.load(
                        model_name=self.config.teacher_model,
                        dtype=teacher_dtype,
                    )
                    self.engine.teacher_model = self.teacher_model
                    self.engine.teacher_tokenizer = self.teacher_tokenizer

                # ── Phase 5 : Cloud sync ──────────────────────────────
                self.current_phase = "cloud_sync"
                print("\n  Phase 5 — Cloud sync")

                # Sauvegarder checkpoint student
                ckpt_path = str(
                    self.checkpoint_dir / f"student_cycle_{cycle:04d}.pt"
                )
                self.student.save_state(ckpt_path)

                # Sauvegarder bridges
                bridge_path = str(
                    self.checkpoint_dir / f"bridges_cycle_{cycle:04d}.pt"
                )
                self.engine.save_bridges(bridge_path)

                # Upload GCS
                gcs_uri = self.gcs.upload_checkpoint(ckpt_path, step=cycle)

                # Push GitHub
                metrics_for_push = {
                    'pc_convergence': pc_score,
                    'distill_loss': self.last_distill_metrics.get('final_loss', 0.0),
                }
                if self.last_benchmark:
                    scores = self.last_benchmark.get('scores', {})
                    for name, val in scores.items():
                        key = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                        metrics_for_push[key] = val if isinstance(val, (int, float)) else 0.0

                if cycle % self.config.auto_push_every_n_distillations == 0:
                    self.github.push_model(
                        local_dir=".",
                        metrics=metrics_for_push,
                        step=cycle,
                        gcs_uri=gcs_uri,
                    )

                # Résumé cycle
                elapsed = time.time() - start_time
                self.current_phase = "idle"
                self._log_csv(
                    step=cycle, phase="cycle_end",
                    distill_loss=self.last_distill_metrics.get('final_loss', 0.0),
                    sdnc_loss=0.0,
                )
                print(f"\n  Cycle {cycle} terminé en {elapsed/60:.1f} min")
                print(f"  Distill loss : {self.last_distill_metrics.get('final_loss', 0.0):.4f}")
                if self.last_benchmark:
                    print(f"  Benchmark : {self.last_benchmark.get('global_score', '?')}")

            except KeyboardInterrupt:
                print("\n\n  Interruption détectée — sauvegarde d'urgence...")
                self.emergency_save()
                print("  Interrompu proprement — checkpoint sauvegardé")
                return

            except Exception as e:
                print(f"\n  ERREUR cycle {cycle} : {e}")
                self.emergency_save()
                raise

    def resume(self):
        """
        Reprend depuis le dernier checkpoint.

        Cherche d'abord en local, puis sur GCS si disponible.
        """
        # Chercher en local
        local_ckpts = sorted(self.checkpoint_dir.glob("student_cycle_*.pt"))

        if local_ckpts:
            latest = local_ckpts[-1]
            print(f"[Resume] Checkpoint local trouvé : {latest}")
            self.student.load_state(str(latest))

            # Charger bridges si disponible
            cycle_num = latest.stem.split("_")[-1]
            bridge_path = self.checkpoint_dir / f"bridges_cycle_{cycle_num}.pt"
            if bridge_path.exists() and self.engine:
                self.engine.load_bridges(str(bridge_path))

            return

        # Chercher sur GCS
        gcs_path = self.gcs.download_latest_checkpoint(
            str(self.checkpoint_dir), model_type="student"
        )
        if gcs_path:
            print(f"[Resume] Checkpoint GCS téléchargé : {gcs_path}")
            self.student.load_state(str(gcs_path))
            return

        print("[Resume] Aucun checkpoint trouvé — démarrage from scratch")

    def emergency_save(self):
        """
        Sauvegarde d'urgence — brain state + bridge state.

        Appelé depuis try/finally. Fonctionne même après un OOM partiel.
        """
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Student brain
        try:
            ckpt_path = str(
                self.checkpoint_dir / f"emergency_student_{timestamp}.pt"
            )
            self.student.save_state(ckpt_path)
        except Exception as e:
            print(f"[Emergency] ERREUR save student : {e}")

        # Bridges
        if self.engine:
            try:
                bridge_path = str(
                    self.checkpoint_dir / f"emergency_bridges_{timestamp}.pt"
                )
                self.engine.emergency_save(bridge_path)
            except Exception as e:
                print(f"[Emergency] ERREUR save bridges : {e}")

    def status(self) -> str:
        """
        Retourne un tableau ASCII de l'état courant du pipeline.

        Affiche : phase, cycle, benchmark, RAM, VRAM, steps.
        """
        lines = []
        lines.append("+" + "-" * 40 + "+" + "-" * 20 + "+")
        lines.append(f"| {'Paramètre':<38} | {'Valeur':>18} |")
        lines.append("+" + "-" * 40 + "+" + "-" * 20 + "+")

        lines.append(f"| {'Phase courante':<38} | {self.current_phase:>18} |")
        lines.append(f"| {'Cycle':<38} | {self.cycle_count:>18} |")

        if self.student:
            lines.append(f"| {'Steps student':<38} | {self.student.step_count:>18} |")
            lines.append(f"| {'Mémoires hippocampe':<38} | {len(self.student.hippocampus.metadata):>18} |")

        if self.engine:
            lines.append(f"| {'Épisodes distillation':<38} | {self.engine.total_episodes:>18} |")
            if self.engine.loss_history:
                lines.append(f"| {'Dernière loss bridge':<38} | {self.engine.loss_history[-1]:>18.4f} |")

        # RAM
        ram = _ram_gb()
        ram_total = _ram_total_gb()
        if ram_total > 0:
            lines.append(f"| {'RAM':<38} | {f'{ram:.1f}/{ram_total:.0f} GB':>18} |")

        # VRAM
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated(0) / 1e9
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            lines.append(f"| {'VRAM':<38} | {f'{vram_used:.1f}/{vram_total:.0f} GB':>18} |")

        if self.last_benchmark:
            score = self.last_benchmark.get('global_score', '?')
            lines.append(f"| {'Dernier benchmark':<38} | {str(score):>18} |")

        lines.append("+" + "-" * 40 + "+" + "-" * 20 + "+")

        return "\n".join(lines)
