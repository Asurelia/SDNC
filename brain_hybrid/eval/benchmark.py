"""
Benchmark maison SDNC — mesure ce que SDNC optimise vraiment.

5 métriques :
  1. PC Convergence — l'erreur de prédiction diminue-t-elle ?
  2. Mémoire épisodique — les faits encodés sont-ils récupérables ?
  3. Détection de conflit — l'ACC détecte-t-il les contradictions ?
  4. Cohérence temporelle — les réponses restent-elles stables ?
  5. Anti catastrophic forgetting — les anciens souvenirs survivent-ils ?

Usage :
    from brain_hybrid.eval.benchmark import SDNCBenchmark
    bench = SDNCBenchmark(model)
    results = bench.run_full_benchmark()
"""

import json
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime


# ── Données de test ──────────────────────────────────────────────

FACTS = [
    "Le projet s'appelle SDNC et utilise Qwen3-4B",
    "Le GPU cible est AMD RX 7800 XT avec 16 GB VRAM",
    "L hippocampe utilise Sparse Distributed Memory de Kanerva",
    "Le cortex préfrontal est modélisé par Qwen3-4B gelé",
    "STDP signifie Spike-Timing-Dependent Plasticity",
    "Le StepScheduler simule les rythmes gamma et thêta",
    "L ACC détecte les conflits et alerte Qwen",
    "Le predictive coding minimise l erreur de prédiction",
    "La consolidation sommeil transfère SDM vers les CfC",
    "La dopamine module l apprentissage local via STDP",
]

FACT_KEYWORDS = [
    "SDNC", "7800 XT", "Kanerva", "préfrontal", "STDP",
    "StepScheduler", "ACC", "predictive", "sommeil", "dopamine",
]

CONFLICT_LOGIQUE = [
    "2+2=5 donc la Terre est plate",
    "Si A>B et B>A alors A=B est faux",
    "Tous les chats sont des chiens donc mon chat est un chien",
    "L eau bout à 50 degrés à pression normale",
    "Si demain c est hier alors aujourd hui n existe pas",
]

CONFLICT_FACTUEL = [
    "La tour Eiffel est à Berlin",
    "Python est un langage compilé comme le C",
    "Qwen3-4B a 175 milliards de paramètres",
    "Le cervelet contrôle la mémoire épisodique",
    "STDP signifie Static Training Data Protocol",
]

NEUTRAL = [
    "Explique comment fonctionne l attention dans les transformers",
    "Qu est-ce que le predictive coding ?",
    "Comment fonctionne la mémoire épisodique ?",
    "Décris l architecture CfC",
    "Qu est-ce que la dopamine dans un réseau de neurones artificiel ?",
]

CONSISTENCY_PROMPTS = [
    "Quel est ton objectif principal ?",
    "Décris ton architecture en une phrase",
    "Qu est-ce que tu mémorises ?",
    "Comment détectes-tu les erreurs ?",
    "Qu apprends-tu au fil du temps ?",
]

PHYSICS_FACTS = [
    "La superposition quantique permet à un qubit d être 0 et 1 simultanément",
    "L intrication quantique relie deux particules instantanément",
    "Le principe d incertitude d Heisenberg limite la précision de mesure",
    "La décohérence quantique détruit la superposition par interaction",
    "Les portes quantiques sont des matrices unitaires",
]

COOKING_PROMPTS = [
    "La béchamel est une sauce à base de lait et farine",
    "Le soufflé doit cuire sans ouvrir le four",
    "Le roux blond est la base de nombreuses sauces",
    "Le beurre clarifié résiste mieux à la chaleur",
    "La pâte feuilletée nécessite de nombreux pliages",
    "Le fond de veau est la base de la cuisine française",
    "La meringue française utilise des blancs d oeufs battus",
    "Le confit de canard cuit lentement dans sa graisse",
    "La crème anglaise est une base pour les glaces",
    "Le caramel se forme à 170 degrés Celsius",
]

PHYSICS_KEYWORDS = [
    "superposition", "intrication", "Heisenberg", "décohérence", "unitaires",
]


class SDNCBenchmark:
    """Benchmark maison mesurant les capacités propres à SDNC."""

    def __init__(self, model):
        self.model = model
        self.results = {}

    # ── Métrique 1 : PC Convergence ─────────────────────────────

    def pc_convergence_score(self, prompt="Explique le fonctionnement du cerveau humain", n_steps=50):
        """
        Mesure si l'erreur de prédiction converge sur un prompt répété.

        Retourne : float dans [-1, 1] (positif = amélioration)
        Seuil succès : > 0.1
        """
        try:
            errors = []
            for _ in range(n_steps):
                r = self.model.forward(prompt, learn=True)
                if r["pc_errors"]:
                    errors.append(r["pc_errors"][0])

            if len(errors) < 20:
                return 0.0

            first_10 = errors[:10]
            last_10 = errors[-10:]
            mean_first = sum(first_10) / len(first_10)
            mean_last = sum(last_10) / len(last_10)

            if mean_first < 1e-8:
                return 0.0

            return 1.0 - (mean_last / mean_first)
        except Exception as e:
            print(f"  [PC Convergence ERREUR] {e}")
            return 0.0

    # ── Métrique 2 : Mémoire épisodique ─────────────────────────

    def episodic_memory_score(self, facts=None, keywords=None, n_distractors=100):
        """
        Mesure la rétention de faits après distraction.

        Retourne : float dans [0, 1]
        Seuil succès : > 0.5
        """
        if facts is None:
            facts = FACTS
        if keywords is None:
            keywords = FACT_KEYWORDS

        try:
            # Phase encode
            embeddings = {}
            for i, fact in enumerate(facts):
                r = self.model.forward(fact, learn=True)
                layer_reps = self.model.llm.get_layer_representations(
                    fact, layers=[self.model.config.intercept_layers[0]]
                )
                embeddings[keywords[i]] = layer_reps[0].mean(dim=1).squeeze().detach().cpu()

            # Phase distraction
            distractor_prompts = [
                f"Parle moi du sujet numéro {i} en détail" for i in range(n_distractors)
            ]
            for prompt in distractor_prompts:
                self.model.forward(prompt, learn=True)

            # Phase recall
            sims = []
            for keyword, original_emb in embeddings.items():
                recalled = self.model.remember(keyword)
                if recalled.norm().item() < 1e-6:
                    sims.append(0.0)
                    continue
                sim = F.cosine_similarity(
                    original_emb.unsqueeze(0).float(),
                    recalled.unsqueeze(0).float()
                ).item()
                sims.append(max(0.0, sim))

            return sum(sims) / len(sims) if sims else 0.0
        except Exception as e:
            print(f"  [Mémoire épisodique ERREUR] {e}")
            return 0.0

    # ── Métrique 3 : Détection de conflit ───────────────────────

    def conflict_detection_score(self):
        """
        Mesure la capacité de l'ACC à détecter les conflits.

        Retourne : dict avec detection_rates + F1
        Seuil succès : F1 > 0.6
        """
        try:
            results = {"logique": [], "factuel": [], "neutre": []}

            for prompt in CONFLICT_LOGIQUE:
                r = self.model.forward(prompt, learn=False)
                results["logique"].append(r["is_conflict"])

            for prompt in CONFLICT_FACTUEL:
                r = self.model.forward(prompt, learn=False)
                results["factuel"].append(r["is_conflict"])

            for prompt in NEUTRAL:
                r = self.model.forward(prompt, learn=False)
                results["neutre"].append(r["is_conflict"])

            # Taux de détection
            logique_rate = sum(results["logique"]) / len(results["logique"])
            factuel_rate = sum(results["factuel"]) / len(results["factuel"])
            fp_rate = sum(results["neutre"]) / len(results["neutre"])

            # F1
            true_positives = sum(results["logique"]) + sum(results["factuel"])
            total_detected = true_positives + sum(results["neutre"])
            total_real = len(CONFLICT_LOGIQUE) + len(CONFLICT_FACTUEL)

            precision = true_positives / (total_detected + 1e-8)
            recall = true_positives / (total_real + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            return {
                "conflict_logique_detection_rate": logique_rate,
                "conflict_factuel_detection_rate": factuel_rate,
                "neutral_false_positive_rate": fp_rate,
                "global_f1": f1,
            }
        except Exception as e:
            print(f"  [Détection conflit ERREUR] {e}")
            return {
                "conflict_logique_detection_rate": 0.0,
                "conflict_factuel_detection_rate": 0.0,
                "neutral_false_positive_rate": 0.0,
                "global_f1": 0.0,
            }

    # ── Métrique 4 : Cohérence temporelle ───────────────────────

    def temporal_consistency_score(self, n_steps=200):
        """
        Mesure si les réponses restent cohérentes dans le temps.

        Retourne : float dans [0, 1]
        Seuil succès : > 0.7
        """
        try:
            # Enregistrer les réponses initiales
            initial_embeddings = {}
            for prompt in CONSISTENCY_PROMPTS:
                r = self.model.forward(prompt, learn=True)
                reps = self.model.llm.get_layer_representations(
                    r["response"][:200], layers=[self.model.config.intercept_layers[-1]]
                )
                initial_embeddings[prompt] = reps[0].mean(dim=1).squeeze().detach().cpu()

            # N steps intermédiaires avec prompts variés
            filler_prompts = [
                f"Question numéro {i} sur un sujet quelconque" for i in range(n_steps)
            ]
            for fp in filler_prompts:
                self.model.forward(fp, learn=True)

            # Mesurer les réponses finales
            sims = []
            for prompt, init_emb in initial_embeddings.items():
                r = self.model.forward(prompt, learn=False)
                reps = self.model.llm.get_layer_representations(
                    r["response"][:200], layers=[self.model.config.intercept_layers[-1]]
                )
                final_emb = reps[0].mean(dim=1).squeeze().detach().cpu()
                sim = F.cosine_similarity(
                    init_emb.unsqueeze(0).float(),
                    final_emb.unsqueeze(0).float()
                ).item()
                sims.append(max(0.0, sim))

            return sum(sims) / len(sims) if sims else 0.0
        except Exception as e:
            print(f"  [Cohérence temporelle ERREUR] {e}")
            return 0.0

    # ── Métrique 5 : Anti catastrophic forgetting ───────────────

    def forgetting_score(self, n_steps_between=500):
        """
        Mesure la rétention de souvenirs après distraction massive.

        Retourne : float dans [0, 1] (1.0 = aucun oubli)
        Seuil succès : > 0.7
        """
        try:
            # Phase A : encoder les faits physique
            recall_before = self.episodic_memory_score(
                facts=PHYSICS_FACTS, keywords=PHYSICS_KEYWORDS, n_distractors=0
            )

            # Phase distraction : cuisine française
            for i in range(n_steps_between):
                prompt = COOKING_PROMPTS[i % len(COOKING_PROMPTS)]
                self.model.forward(prompt, learn=True)

            # Phase recall
            recall_after = self.episodic_memory_score(
                facts=PHYSICS_FACTS, keywords=PHYSICS_KEYWORDS, n_distractors=0
            )

            if recall_before < 1e-8:
                return 1.0

            return min(1.0, recall_after / recall_before)
        except Exception as e:
            print(f"  [Anti-forgetting ERREUR] {e}")
            return 0.0

    # ── Rapport complet ─────────────────────────────────────────

    def run_full_benchmark(self, output_dir=None, **kwargs):
        """
        Exécute les 5 métriques et génère un rapport.

        kwargs peuvent surcharger : n_steps, n_facts, n_distractors, etc.
        """
        if output_dir is None:
            output_dir = Path(__file__).parent / "results"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {"timestamp": timestamp, "sdnc_version": "0.2.0"}

        # Hardware info
        if torch.cuda.is_available():
            results["hw_type"] = torch.cuda.get_device_name(0)
            results["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            results["hw_type"] = "cpu"
            results["vram_gb"] = 0

        metrics = [
            ("PC Convergence", "> 0.10", lambda: self.pc_convergence_score(
                n_steps=kwargs.get("pc_steps", 50))),
            ("Mémoire épisodique", "> 0.50", lambda: self.episodic_memory_score(
                n_distractors=kwargs.get("n_distractors", 100))),
            ("Détection conflit (F1)", "> 0.60", lambda: self.conflict_detection_score()),
            ("Cohérence temporelle", "> 0.70", lambda: self.temporal_consistency_score(
                n_steps=kwargs.get("consistency_steps", 200))),
            ("Anti-forgetting", "> 0.70", lambda: self.forgetting_score(
                n_steps_between=kwargs.get("forgetting_steps", 500))),
        ]

        thresholds = [0.10, 0.50, 0.60, 0.70, 0.70]
        scores = []
        passed = 0

        print()
        print("+" + "-" * 27 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 8 + "+")
        print(f"| {'Métrique':<25} | {'Score':>8} | {'Seuil':>8} | {'Status':>6} |")
        print("+" + "-" * 27 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 8 + "+")

        for i, (name, seuil_str, fn) in enumerate(metrics):
            print(f"| {name:<25} | {'...':>8} | {seuil_str:>8} | {'...':>6} |", end="\r")
            result = fn()

            if isinstance(result, dict):
                score = result.get("global_f1", 0.0)
                results[name.lower().replace(" ", "_").replace("(", "").replace(")", "")] = result
            else:
                score = result
                results[name.lower().replace(" ", "_").replace("(", "").replace(")", "")] = score

            scores.append(score)
            status = "PASS" if score > thresholds[i] else "FAIL"
            if score > thresholds[i]:
                passed += 1

            print(f"| {name:<25} | {score:>8.4f} | {seuil_str:>8} | {status:>6} |")

        print("+" + "-" * 27 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 8 + "+")
        print(f"| {'SCORE GLOBAL':<25} | {passed}/{len(metrics):>6} |          |        |")
        print("+" + "-" * 27 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 8 + "+")

        results["global_score"] = f"{passed}/{len(metrics)}"
        results["scores"] = {name: s for (name, _, _), s in zip(metrics, scores)}

        # Sauvegarder JSON
        json_path = output_dir / f"benchmark_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        latest_path = output_dir / "latest.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        print(f"\nRésultats sauvegardés : {json_path}")
        return results
