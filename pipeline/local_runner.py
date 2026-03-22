"""
Runner local pour le PC — inférence offline avec le brain synced.

Charge uniquement le student (Qwen3.5-4B + brain modules CfC).
N'a PAS besoin du teacher ni du bridge.
Compatible RX 7800 XT (ROCm/DirectML), fonctionne offline.

Commandes interactives :
  /status  — état du brain (step, erreurs, mémoires)
  /memory  — statistiques hippocampe
  /sync    — télécharger le dernier checkpoint depuis GCS
  /bench   — lancer le benchmark SDNC (smoke test)
  /quit    — quitter
  (texte)  — chat normal avec le brain
"""

import json
from pathlib import Path

import torch

from .dual_model_config import DualModelConfig
from .cloud_sync import GCSManager
from brain_hybrid.model import BrainHybridModel
from brain_hybrid.eval.benchmark import SDNCBenchmark


class LocalRunner:
    """
    Runner offline pour le PC local (RX 7800 XT).

    Charge le student brain Qwen3.5-4B + CfC/SNN depuis un checkpoint.
    Fonctionne sans internet (mode offline avec checkpoint local).
    """

    def __init__(
        self,
        config: DualModelConfig = None,
        checkpoint_dir: str = "checkpoints",
    ):
        """
        Initialise le runner local.

        Args:
            config: Configuration (utilise student_brain_config).
            checkpoint_dir: Répertoire des checkpoints locaux.
        """
        self.config = config or DualModelConfig()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

    def sync_from_github(self, repo_url: str, local_dir: str):
        """
        Synchronise depuis GitHub + télécharge le checkpoint GCS si nécessaire.

        Args:
            repo_url: URL du repo GitHub.
            local_dir: Répertoire local de destination.
        """
        from .cloud_sync import GitHubManager

        github = GitHubManager(repo=repo_url, branch=self.config.github_branch)
        github.clone_or_pull(local_dir)

        # Lire releases/latest.json
        latest_path = Path(local_dir) / "releases" / "latest.json"
        if not latest_path.exists():
            print("[Sync] Pas de releases/latest.json — rien à télécharger.")
            return

        with open(latest_path, 'r', encoding='utf-8') as f:
            latest = json.load(f)

        gcs_uri = latest.get('gcs_uri', '')
        if not gcs_uri:
            print("[Sync] Pas de gcs_uri dans latest.json.")
            return

        # Vérifier si le checkpoint est déjà présent localement
        filename = Path(gcs_uri).name
        local_ckpt = self.checkpoint_dir / filename

        if local_ckpt.exists():
            print(f"[Sync] Checkpoint déjà présent : {local_ckpt}")
            return

        # Télécharger depuis GCS
        gcs = GCSManager(bucket=self.config.gcs_bucket)
        downloaded = gcs.download_latest_checkpoint(
            str(self.checkpoint_dir), model_type="student"
        )
        if downloaded:
            print(f"[Sync] Checkpoint téléchargé : {downloaded}")
        else:
            print("[Sync] Impossible de télécharger — utiliser le checkpoint local existant.")

    def load_student(self, checkpoint_path: str = None) -> BrainHybridModel:
        """
        Charge le modèle student (Qwen3.5-4B + brain modules CfC).

        Args:
            checkpoint_path: Chemin vers le checkpoint .pt.
                            Si None, cherche le dernier dans checkpoint_dir.

        Returns:
            BrainHybridModel chargé et prêt.
        """
        brain_config = self.config.student_brain_config

        print(f"Chargement student : {self.config.student_model}")
        self.model = BrainHybridModel(brain_config)

        if checkpoint_path is None:
            # Chercher le dernier checkpoint
            ckpts = sorted(self.checkpoint_dir.glob("*.pt"))
            if ckpts:
                checkpoint_path = str(ckpts[-1])
                print(f"Dernier checkpoint : {checkpoint_path}")

        if checkpoint_path and Path(checkpoint_path).exists():
            self.model.load_state(checkpoint_path)
        else:
            print("Aucun checkpoint trouvé — démarrage from scratch.")

        return self.model

    def interactive_chat(self, model: BrainHybridModel = None):
        """
        Boucle de chat interactive avec le brain.

        Commandes spéciales :
          /status — affiche métriques courantes
          /memory — statistiques hippocampe
          /sync   — force git pull + téléchargement GCS
          /bench  — lance benchmark smoke test local
          /quit   — quitter

        Args:
            model: BrainHybridModel. Si None, utilise self.model.
        """
        if model is None:
            model = self.model
        if model is None:
            print("Aucun modèle chargé. Appelez load_student() d'abord.")
            return

        print("\n" + "=" * 50)
        print(" SDNC Brain Chat — mode local")
        print(" Commandes : /status /memory /sync /bench /quit")
        print("=" * 50 + "\n")

        while True:
            try:
                user_input = input("Vous > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nAu revoir.")
                break

            if not user_input:
                continue

            # Commandes spéciales
            if user_input == "/quit":
                print("Au revoir.")
                break

            elif user_input == "/status":
                self._show_status(model)
                continue

            elif user_input == "/memory":
                self._show_memory(model)
                continue

            elif user_input == "/sync":
                if self.config.github_repo:
                    self.sync_from_github(self.config.github_repo, ".")
                else:
                    print("Pas de repo GitHub configuré.")
                continue

            elif user_input == "/bench":
                self._run_smoke_bench(model)
                continue

            # Chat normal
            try:
                result = model.forward(user_input, learn=True)
                response = result.get('response', '(pas de réponse)')
                print(f"\nSDNC > {response}\n")

                # Indicateurs rapides
                if result.get('is_conflict'):
                    print(f"  [CONFLIT détecté — score={result['conflict_score']:.2f}]")
                if result.get('sleep_triggered'):
                    print("  [Sleep consolidation déclenchée]")

            except Exception as e:
                print(f"\nErreur : {e}\n")

    def _show_status(self, model: BrainHybridModel):
        """Affiche l'état courant du brain."""
        print("\n--- Status Brain ---")
        print(f"  Steps          : {model.step_count}")
        print(f"  Mémoires       : {len(model.hippocampus.metadata)}")
        print(f"  Phase scheduler: {model.scheduler.get_phase()}")
        print(f"  ACC trend      : {model.acc.trend()}")

        if model.error_history:
            recent = model.error_history[-10:]
            mean_err = sum(recent) / len(recent)
            print(f"  Erreur moyenne : {mean_err:.4f} (10 derniers)")

        gate_alphas = [g.alpha.item() for g in model.injection_gates]
        print(f"  Gate alphas    : {[f'{a:.4f}' for a in gate_alphas]}")

        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated(0) / 1e9
            print(f"  VRAM utilisée  : {vram:.1f} GB")
        print()

    def _show_memory(self, model: BrainHybridModel):
        """Affiche les statistiques de l'hippocampe."""
        stats = model.hippocampus.stats()
        print("\n--- Hippocampe (SDM) ---")
        print(f"  Souvenirs stockés    : {stats['memories_stored']}")
        print(f"  Locations actives    : {stats['active_locations']}")
        print(f"  Locations totales    : {stats['total_locations']}")
        print(f"  Utilisation          : {stats['usage_pct']:.1f}%")
        print()

    def _run_smoke_bench(self, model: BrainHybridModel):
        """Lance un benchmark rapide."""
        print("\nLancement benchmark smoke test...")
        bench = SDNCBenchmark(model)
        score = bench.pc_convergence_score(n_steps=20)
        print(f"  PC Convergence : {score:.4f}")
        print()
