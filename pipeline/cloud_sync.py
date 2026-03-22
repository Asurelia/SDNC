"""
Synchronisation cloud pour le pipeline SDNC.

GCSManager  : upload/download checkpoints vers Google Cloud Storage.
GitHubManager : push JSON metrics + scripts vers GitHub.

Secrets via variables d'environnement ou Colab Secrets — jamais hardcodés.
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def _get_secret(key: str) -> Optional[str]:
    """
    Récupère un secret depuis l'environnement ou les Colab Secrets.

    Priorité : os.environ > google.colab.userdata
    """
    val = os.environ.get(key)
    if val:
        return val
    try:
        from google.colab import userdata
        return userdata.get(key)
    except Exception:
        return None


class GCSManager:
    """
    Gestionnaire de sync avec Google Cloud Storage.

    Upload/download checkpoints et model cards.
    Secrets via GOOGLE_APPLICATION_CREDENTIALS ou auth Colab.
    """

    def __init__(self, bucket: str = "sdnc-models", prefix: str = "checkpoints"):
        """
        Initialise le gestionnaire GCS.

        Args:
            bucket: Nom du bucket GCS.
            prefix: Préfixe pour les chemins de checkpoints.
        """
        self.bucket = bucket
        self.prefix = prefix
        self._available = self._check_available()

        if not self._available:
            print("[GCS] gsutil non disponible — mode offline.")

    def _check_available(self) -> bool:
        """Vérifie si gsutil est disponible."""
        try:
            result = subprocess.run(
                ["gsutil", "version"],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def upload_checkpoint(
        self,
        local_path: str,
        step: int,
        model_type: str = "student",
    ) -> str:
        """
        Upload un checkpoint vers GCS.

        Args:
            local_path: Chemin local du fichier .pt.
            step: Numéro du step.
            model_type: 'student' ou 'teacher'.

        Returns:
            URI GCS du fichier uploadé, ou chaîne vide si offline.
        """
        if not self._available:
            print("[GCS] Offline — skip upload.")
            return ""

        date_str = datetime.now().strftime("%Y%m%d")
        gcs_path = f"gs://{self.bucket}/{self.prefix}/{date_str}/{model_type}/step_{step:06d}.pt"

        try:
            subprocess.run(
                ["gsutil", "cp", local_path, gcs_path],
                check=True, capture_output=True, text=True, timeout=600
            )
            print(f"[GCS] Upload : {gcs_path}")
            return gcs_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"[GCS] ERREUR upload : {e}")
            return ""

    def upload_model_card(self, metrics: dict) -> str:
        """
        Upload un model card JSON vers GCS.

        Args:
            metrics: Dictionnaire de métriques.

        Returns:
            URI GCS du model card.
        """
        if not self._available:
            return ""

        import tempfile
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        gcs_path = f"gs://{self.bucket}/{self.prefix}/model_cards/card_{date_str}.json"

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
            tmp_path = f.name

        try:
            subprocess.run(
                ["gsutil", "cp", tmp_path, gcs_path],
                check=True, capture_output=True, text=True, timeout=120
            )
            print(f"[GCS] Model card : {gcs_path}")
            return gcs_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"[GCS] ERREUR model card : {e}")
            return ""
        finally:
            os.unlink(tmp_path)

    def download_latest_checkpoint(
        self,
        local_dir: str,
        model_type: str = "student",
    ) -> Optional[Path]:
        """
        Télécharge le dernier checkpoint depuis GCS.

        Args:
            local_dir: Répertoire local de destination.
            model_type: 'student' ou 'teacher'.

        Returns:
            Path du fichier téléchargé, ou None si indisponible.
        """
        if not self._available:
            print("[GCS] Offline — skip download.")
            return None

        gcs_prefix = f"gs://{self.bucket}/{self.prefix}/"
        try:
            result = subprocess.run(
                ["gsutil", "ls", "-l", f"{gcs_prefix}**/{model_type}/*.pt"],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                return None

            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip() and "TOTAL" not in l]
            if not lines:
                return None

            # Dernier fichier (trié par date)
            latest_line = lines[-1]
            parts = latest_line.split()
            gcs_uri = parts[-1] if parts else None

            if not gcs_uri:
                return None

            local_path = Path(local_dir) / Path(gcs_uri).name
            local_path.parent.mkdir(parents=True, exist_ok=True)

            subprocess.run(
                ["gsutil", "cp", gcs_uri, str(local_path)],
                check=True, capture_output=True, text=True, timeout=600
            )
            print(f"[GCS] Téléchargé : {local_path}")
            return local_path

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"[GCS] ERREUR download : {e}")
            return None

    def list_checkpoints(self) -> List[dict]:
        """Liste tous les checkpoints disponibles sur GCS."""
        if not self._available:
            return []

        gcs_prefix = f"gs://{self.bucket}/{self.prefix}/"
        try:
            result = subprocess.run(
                ["gsutil", "ls", "-l", f"{gcs_prefix}**/*.pt"],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                return []

            checkpoints = []
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line and "TOTAL" not in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        checkpoints.append({
                            'size': parts[0],
                            'date': parts[1],
                            'uri': parts[2] if len(parts) > 2 else parts[-1],
                        })
            return checkpoints

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return []


class GitHubManager:
    """
    Gestionnaire de push/pull vers GitHub.

    Ne push JAMAIS de fichiers .pt (trop gros, dans .gitignore).
    Push uniquement : JSON metrics, scripts, configs.
    """

    def __init__(self, repo: str = "", branch: str = "main"):
        """
        Initialise le gestionnaire GitHub.

        Args:
            repo: URL du repo GitHub (ex: https://github.com/user/sdnc).
            branch: Branche cible.
        """
        self.repo = repo
        self.branch = branch
        self._token = _get_secret("GITHUB_TOKEN")

        if not self.repo:
            print("[GitHub] Aucun repo configuré — mode offline.")

    def clone_or_pull(self, local_dir: str):
        """
        Clone le repo ou fait un git pull si déjà présent.

        Args:
            local_dir: Répertoire local du repo.
        """
        if not self.repo:
            return

        local = Path(local_dir)
        if (local / ".git").exists():
            subprocess.run(
                ["git", "pull", "origin", self.branch],
                cwd=str(local), capture_output=True, text=True, timeout=120
            )
        else:
            repo_url = self.repo
            if self._token and "github.com" in repo_url:
                repo_url = repo_url.replace(
                    "https://",
                    f"https://{self._token}@"
                )
            local.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "-b", self.branch, repo_url, str(local)],
                capture_output=True, text=True, timeout=300
            )

    def push_model(
        self,
        local_dir: str,
        metrics: dict,
        step: int,
        gcs_uri: str = "",
    ):
        """
        Met à jour releases/latest.json et push vers GitHub.

        Args:
            local_dir: Répertoire local du repo.
            metrics: Métriques du benchmark.
            step: Numéro du step.
            gcs_uri: URI GCS du checkpoint (.pt en .gitignore).
        """
        if not self.repo:
            print("[GitHub] Offline — skip push.")
            return

        local = Path(local_dir)
        releases_dir = local / "releases"
        releases_dir.mkdir(parents=True, exist_ok=True)

        # Créer latest.json
        latest = {
            'version': '0.2.0',
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'teacher': 'Qwen3.5-35B-A3B',
            'student': 'Qwen3.5-4B',
            'gcs_uri': gcs_uri,
            'metrics': {
                'pc_convergence': metrics.get('pc_convergence', 0.0),
                'episodic_memory': metrics.get('episodic_memory', 0.0),
                'conflict_f1': metrics.get('conflict_f1', 0.0),
                'temporal_consistency': metrics.get('temporal_consistency', 0.0),
                'forgetting_score': metrics.get('forgetting_score', 0.0),
                'distill_loss': metrics.get('distill_loss', 0.0),
            },
            'hw_compatible': ['A100', 'RX7800XT-16GB'],
            'sdnc_version': '0.2.0',
        }

        latest_path = releases_dir / "latest.json"
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(latest, f, indent=2, ensure_ascii=False)

        # Git add + commit + push
        try:
            subprocess.run(
                ["git", "add", "releases/latest.json"],
                cwd=str(local), check=True, capture_output=True, timeout=30
            )
            commit_msg = f"🤖 SDNC auto-release step {step} — loss={metrics.get('distill_loss', 0.0):.4f}"
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=str(local), capture_output=True, text=True, timeout=30
            )

            push_cmd = ["git", "push", "origin", self.branch]
            subprocess.run(
                push_cmd,
                cwd=str(local), check=True, capture_output=True, timeout=120
            )
            print(f"[GitHub] Push : {commit_msg}")

        except subprocess.CalledProcessError as e:
            print(f"[GitHub] ERREUR push : {e}")
        except subprocess.TimeoutExpired:
            print("[GitHub] Timeout push.")
