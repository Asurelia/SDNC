"""
Étude d'ablation — compare SDNC avec et sans enrichissement distillé.

Mesure si le ProjectionBridge (cosine=0.61) aide réellement
les modules CfC+SNN dans leur apprentissage local.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List


class AblationStudy:
    """
    Compare les performances SDNC avec et sans bridge distillé.

    Lance N steps identiques sur deux versions du modèle
    et compare les métriques clés.
    """

    def __init__(self):
        self.test_prompts = self._get_test_prompts()

    def _get_test_prompts(self, n: int = 100) -> List[str]:
        """Génère N prompts de test fixes (reproductibles)."""
        base = [
            "Explique le fonctionnement d'un processeur quantique.",
            "Décris la structure interne d'une étoile à neutrons.",
            "Qu'est-ce que la mémoire de travail selon Baddeley ?",
            "Comment fonctionne l'algorithme PageRank de Google ?",
            "Explique la différence entre ARN messager et ARN de transfert.",
            "Qu'est-ce que la conjecture de Poincaré et pourquoi est-elle importante ?",
            "Décris le mécanisme de la vision binoculaire.",
            "Comment fonctionne la compression avec les transformées en ondelettes ?",
            "Explique le paradoxe EPR en mécanique quantique.",
            "Qu'est-ce que l'architecture von Neumann et ses limitations ?",
            "Décris le rôle du cortex visuel V1 dans le traitement de l'image.",
            "Comment fonctionne un laser à fibre optique ?",
            "Explique la théorie de la relativité restreinte.",
            "Qu'est-ce que le deep learning et en quoi diffère-t-il du machine learning classique ?",
            "Décris le cycle de Calvin dans la photosynthèse.",
            "Comment fonctionne la mémoire cache dans un processeur ?",
            "Explique le principe de la tomographie par émission de positrons (TEP).",
            "Qu'est-ce que la théorie des jeux et l'équilibre de Nash ?",
            "Décris le fonctionnement des cellules souches pluripotentes.",
            "Comment fonctionne un réseau de neurones récurrent (RNN) ?",
        ]
        # Répéter pour atteindre n prompts
        prompts = []
        while len(prompts) < n:
            prompts.extend(base)
        return prompts[:n]

    def run(
        self,
        model_with_bridge,
        model_without_bridge,
        n_steps: int = 100,
    ) -> dict:
        """
        Lance n_steps identiques sur les deux versions du modèle.

        Args:
            model_with_bridge: BrainHybridModel avec bridge chargé.
            model_without_bridge: BrainHybridModel sans bridge.
            n_steps: Nombre de steps à comparer.

        Returns:
            dict avec métriques comparatives et verdict.
        """
        results = {"with_bridge": {}, "without_bridge": {}}
        prompts = self.test_prompts[:n_steps]

        for label, model in [
            ("with_bridge", model_with_bridge),
            ("without_bridge", model_without_bridge),
        ]:
            errors_history = []
            pc_history = []

            print(f"\n  [{label}] {n_steps} steps...")
            for i, prompt in enumerate(prompts):
                try:
                    result = model.forward(prompt, learn=True)
                    errors_history.append(result.get("mean_error", 0.0))
                    pc_errs = result.get("pc_errors", [0.0])
                    pc_history.append(pc_errs[0] if pc_errs else 0.0)
                except Exception:
                    continue

                if (i + 1) % 25 == 0:
                    recent = errors_history[-25:]
                    print(f"    step {i+1:3d} | err={sum(recent)/len(recent):.4f}")

            results[label] = {
                "mean_prediction_error": (
                    sum(errors_history) / len(errors_history)
                    if errors_history else 0.0
                ),
                "prediction_error_trend": (
                    errors_history[-1] - errors_history[0]
                    if len(errors_history) > 1 else 0.0
                ),
                "mean_pc_error": (
                    sum(pc_history) / len(pc_history)
                    if pc_history else 0.0
                ),
                "pc_trend": (
                    pc_history[-1] - pc_history[0]
                    if len(pc_history) > 1 else 0.0
                ),
                "n_steps_completed": len(errors_history),
            }

        # Delta
        results["delta"] = {
            k: results["with_bridge"][k] - results["without_bridge"][k]
            for k in results["with_bridge"]
            if isinstance(results["with_bridge"][k], (int, float))
        }
        results["bridge_helps"] = results["delta"].get("prediction_error_trend", 0) < 0
        results["timestamp"] = datetime.now().isoformat()

        return results

    def print_report(self, results: dict):
        """Affiche le tableau comparatif ASCII."""
        print()
        print("+" + "-" * 29 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 10 + "+")
        print(f"| {'Métrique':<27} | {'Avec bridge':>12} | {'Sans bridge':>12} | {'Delta':>8} |")
        print("+" + "-" * 29 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 10 + "+")

        for key in results.get("with_bridge", {}):
            wb = results["with_bridge"][key]
            nb = results["without_bridge"][key]
            if isinstance(wb, (int, float)) and isinstance(nb, (int, float)):
                d = results["delta"].get(key, 0)
                sign = "+" if d > 0 else ""
                print(f"| {key:<27} | {wb:>12.4f} | {nb:>12.4f} | {sign}{d:>7.4f} |")

        print("+" + "-" * 29 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 10 + "+")

        verdict = "BRIDGE UTILE" if results.get("bridge_helps") else "BRIDGE NEUTRE"
        print(f"| Verdict : {verdict:<55} |")
        print("+" + "-" * 69 + "+")

    def save_results(self, results: dict, output_dir: str = "eval/results"):
        """Sauvegarde les résultats en JSON."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out / f"ablation_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"Résultats sauvegardés : {path}")
        return path
