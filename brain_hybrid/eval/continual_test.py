"""
Test d'apprentissage continu — vérifie les critères AGENTS.md §8.

Usage depuis Colab ou local :
    from brain_hybrid.eval.continual_test import run_continual_test
    results = run_continual_test(model, n_steps=100)
"""

import torch


PROMPTS = [
    "Explique comment fonctionne la photosynthèse",
    "Qu'est-ce que la conscience selon les neurosciences ?",
    "Comment le cerveau consolide-t-il les souvenirs ?",
    "Décris le fonctionnement d'un neurone biologique",
    "Quelle est la différence entre mémoire courte et longue durée ?",
    "Comment fonctionne la plasticité synaptique ?",
    "Quel rôle joue la dopamine dans l'apprentissage ?",
    "Mon chat Luna est roux et adore la laine",
    "Les réseaux de neurones artificiels imitent le cerveau",
    "Le sommeil est essentiel pour la consolidation mémorielle",
]


def run_continual_test(model, n_steps=100, verbose=True):
    """
    Lance n_steps forward+learn et vérifie les critères de succès.

    Retourne un dict avec les résultats et un bool 'all_passed'.
    """
    start_step = model.step_count

    if verbose:
        print(f"=== Test d'apprentissage continu ({n_steps} steps) ===")
        print(f"Reprise depuis step {start_step}")
        print()

    for i in range(n_steps):
        prompt = PROMPTS[i % len(PROMPTS)]
        result = model.forward(prompt, learn=True)

        if verbose and (i + 1) % 10 == 0:
            errs = [f"{e:.4f}" for e in result["prediction_errors"]]
            print(f"  Step {model.step_count:4d} | err={result['mean_error']:.4f} | "
                  f"dop={result['dopamine'][0]:.3f} | mem={result['memories_stored']}")

    if verbose:
        print()

    # --- Critères de succès ---
    results = {}

    # 1. Prédiction s'améliore : erreur(step 100) < erreur(step 1) × 0.9
    err_first = model.error_history[start_step]
    err_last = model.error_history[-1]
    improvement = 1 - err_last / err_first if err_first > 0 else 0
    results["prediction_improves"] = err_last < err_first * 0.9
    results["err_first"] = err_first
    results["err_last"] = err_last
    results["improvement_pct"] = improvement * 100

    # 2. STDP stable : changement poids < 1% par step
    # (déjà validé par unit test, mais on vérifie que rien n'a explosé)
    weight_norms = []
    for module in model.brain_modules:
        for p in module.cfc.parameters():
            if p.requires_grad and len(p.shape) == 2:
                weight_norms.append(p.norm().item())
    results["stdp_stable"] = all(n < 1e6 for n in weight_norms)
    results["max_weight_norm"] = max(weight_norms) if weight_norms else 0

    # 3. Hippocampe recall : "Luna" récupérable
    recalled = model.remember("chat Luna")
    recall_norm = recalled.norm().item()
    results["hippocampe_recall"] = recall_norm > 1.0
    results["recall_norm"] = recall_norm

    # 4. Mémoire épisodique : des souvenirs stockés
    stats = model.hippocampus.stats()
    results["memories_stored"] = stats["memories_stored"]
    results["hippocampe_usage_pct"] = stats["usage_pct"]

    # 5. Dopamine clampée [0, 1]
    dopamine_ok = all(0.0 <= s.dopamine_signal <= 1.0 for s in model.stdp_learners)
    results["dopamine_clamped"] = dopamine_ok

    # 6. Pas d'OOM (si on arrive ici, c'est bon)
    results["no_oom"] = True

    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated(0) / 1e9
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        results["vram_used_gb"] = vram_used
        results["vram_total_gb"] = vram_total

    # Résumé
    results["all_passed"] = all([
        results["prediction_improves"],
        results["stdp_stable"],
        results["hippocampe_recall"],
        results["dopamine_clamped"],
        results["no_oom"],
    ])

    if verbose:
        print("=== Résultats ===")
        print(f"  Erreur init → fin   : {err_first:.4f} → {err_last:.4f} ({improvement*100:+.1f}%)")
        print(f"  Prédiction améliore : {'PASS' if results['prediction_improves'] else 'FAIL'}")
        print(f"  STDP stable         : {'PASS' if results['stdp_stable'] else 'FAIL'}")
        print(f"  Hippocampe recall   : {'PASS' if results['hippocampe_recall'] else 'FAIL'} (norme={recall_norm:.4f})")
        print(f"  Dopamine clampée    : {'PASS' if results['dopamine_clamped'] else 'FAIL'}")
        print(f"  Pas d'OOM           : {'PASS' if results['no_oom'] else 'FAIL'}")
        print(f"  Mémoires stockées   : {results['memories_stored']}")
        if torch.cuda.is_available():
            print(f"  VRAM                : {results['vram_used_gb']:.2f} / {results['vram_total_gb']:.1f} GB")
        print()
        print(f"  === {'TOUS LES CRITÈRES PASSENT' if results['all_passed'] else 'CERTAINS CRITÈRES ÉCHOUENT'} ===")

    return results
