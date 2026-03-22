"""
Benchmark BRAIN-HYBRID vs Qwen pur via lm-evaluation-harness.

Usage sur Colab :
    from brain_hybrid.eval.benchmark import run_benchmark
    results = run_benchmark(model)
"""

import torch
import torch.nn.functional as F
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
import lm_eval


class BrainHybridLM(LM):
    """
    Wrapper lm-eval pour BrainHybridModel.

    Mode A : cfc_enabled=False → Qwen pur (baseline)
    Mode B : cfc_enabled=True  → Qwen + CfC injection via hooks
    """

    def __init__(self, brain_model, cfc_enabled: bool = True):
        super().__init__()
        self.brain_model = brain_model
        self.cfc_enabled = cfc_enabled
        self.qwen = brain_model.llm
        self.tokenizer = self.qwen.processor.tokenizer
        self._device = self.qwen.device

        # S'assurer que le tokenizer a un pad_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return 4096

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return self._device

    def tok_encode(self, string, **kwargs):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _get_logits(self, input_ids):
        """Forward pass avec ou sans hooks CfC."""
        if self.cfc_enabled:
            # Enregistrer les hooks
            layer_indices = [l - 1 for l in self.brain_model.config.intercept_layers]
            self.brain_model.hook_manager.register_hooks(
                self.qwen.model,
                self.brain_model.brain_modules,
                self.brain_model.injection_gates,
                layer_indices,
            )

        try:
            with torch.no_grad():
                outputs = self.qwen.model(input_ids=input_ids)
        finally:
            if self.cfc_enabled:
                self.brain_model.hook_manager.remove_hooks()

        return outputs.logits

    def loglikelihood(self, requests: list) -> list:
        results = []
        for request in requests:
            context, continuation = request.args

            # Tokenizer
            if context:
                ctx_ids = self.tokenizer.encode(context, add_special_tokens=True)
            else:
                ctx_ids = [self.tokenizer.bos_token_id or self.tokenizer.eos_token_id]
            cont_ids = self.tokenizer.encode(continuation, add_special_tokens=False)

            full_ids = ctx_ids + cont_ids
            input_ids = torch.tensor([full_ids], device=self._device)

            logits = self._get_logits(input_ids)

            # Score les tokens de continuation
            ctx_len = len(ctx_ids)
            cont_logits = logits[0, ctx_len - 1: -1, :]
            cont_targets = input_ids[0, ctx_len:]

            log_probs = F.log_softmax(cont_logits.float(), dim=-1)
            token_log_probs = log_probs[range(len(cont_targets)), cont_targets]
            total_log_prob = token_log_probs.sum().item()

            greedy_ids = cont_logits.argmax(dim=-1)
            is_greedy = (greedy_ids == cont_targets).all().item()

            results.append((total_log_prob, bool(is_greedy)))
        return results

    def loglikelihood_rolling(self, requests: list) -> list:
        results = []
        for request in requests:
            (string,) = request.args
            input_ids = self.tokenizer.encode(string, return_tensors="pt",
                                               add_special_tokens=True).to(self._device)

            logits = self._get_logits(input_ids)

            shift_logits = logits[0, :-1, :]
            shift_labels = input_ids[0, 1:]
            log_probs = F.log_softmax(shift_logits.float(), dim=-1)
            token_log_probs = log_probs[range(len(shift_labels)), shift_labels]
            results.append(token_log_probs.sum().item())
        return results

    def generate_until(self, requests: list) -> list:
        results = []
        for request in requests:
            context, gen_kwargs = request.args
            until = gen_kwargs.get("until", [])
            max_new_tokens = gen_kwargs.get("max_gen_toks", 128)

            if self.cfc_enabled:
                text = self.brain_model.llm.generate_with_brain(
                    context,
                    brain_modules=self.brain_model.brain_modules,
                    injection_gates=self.brain_model.injection_gates,
                    hook_manager=self.brain_model.hook_manager,
                    max_new_tokens=max_new_tokens,
                )
            else:
                input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self._device)
                with torch.no_grad():
                    out = self.qwen.model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                text = self.tokenizer.decode(out[0][input_ids.shape[1]:],
                                              skip_special_tokens=True)

            for stop in until:
                if stop in text:
                    text = text[:text.index(stop)]

            results.append(text)
        return results


def run_benchmark(brain_model, tasks=None, num_fewshot=0, limit=100):
    """
    Lance le benchmark Mode A (Qwen pur) vs Mode B (Qwen + CfC).

    tasks   : liste de tâches lm-eval (défaut: arc_challenge, hellaswag)
    limit   : nombre d'exemples par tâche (défaut: 100 pour aller vite)

    Retourne un dict avec les résultats comparatifs.
    """
    if tasks is None:
        tasks = ["arc_challenge", "hellaswag"]

    print("=" * 60)
    print("  BENCHMARK : Qwen pur vs Qwen + CfC")
    print("=" * 60)

    # Mode A — Qwen seul
    print("\n--- Mode A : Qwen seul (CfC désactivé) ---")
    lm_off = BrainHybridLM(brain_model, cfc_enabled=False)
    results_off = lm_eval.simple_evaluate(
        model=lm_off,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=1,
        limit=limit,
        log_samples=False,
    )

    # Mode B — Qwen + CfC
    print("\n--- Mode B : Qwen + CfC (injection active) ---")
    lm_on = BrainHybridLM(brain_model, cfc_enabled=True)
    results_on = lm_eval.simple_evaluate(
        model=lm_on,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=1,
        limit=limit,
        log_samples=False,
    )

    # Comparaison
    print("\n" + "=" * 60)
    print("  RÉSULTATS COMPARATIFS")
    print("=" * 60)
    print(f"{'Tâche':<20} {'Qwen seul':>10} {'Qwen+CfC':>10} {'Delta':>10}")
    print("-" * 50)

    comparison = {}
    for task in tasks:
        # Chercher la métrique acc ou acc_norm
        r_off = results_off["results"].get(task, {})
        r_on = results_on["results"].get(task, {})

        metric = "acc_norm,none" if "acc_norm,none" in r_off else "acc,none"
        acc_off = r_off.get(metric, 0)
        acc_on = r_on.get(metric, 0)
        delta = acc_on - acc_off

        print(f"{task:<20} {acc_off:>10.4f} {acc_on:>10.4f} {delta:>+10.4f}")
        comparison[task] = {
            "qwen_only": acc_off,
            "qwen_cfc": acc_on,
            "delta": delta,
        }

    print("-" * 50)
    avg_off = sum(c["qwen_only"] for c in comparison.values()) / len(comparison)
    avg_on = sum(c["qwen_cfc"] for c in comparison.values()) / len(comparison)
    avg_delta = avg_on - avg_off
    print(f"{'MOYENNE':<20} {avg_off:>10.4f} {avg_on:>10.4f} {avg_delta:>+10.4f}")

    verdict = "CfC AIDE" if avg_delta > 0 else "CfC NEUTRE" if avg_delta == 0 else "CfC DÉGRADE"
    print(f"\nVerdict : {verdict} ({avg_delta:+.2%})")

    return {
        "comparison": comparison,
        "results_off": results_off,
        "results_on": results_on,
        "verdict": verdict,
    }
