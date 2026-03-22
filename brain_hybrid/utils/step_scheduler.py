"""
Planificateur de steps inspiré des rythmes cérébraux.

Simule les oscillations gamma (rapide) et thêta (lent) du cerveau :
- gamma : activité rapide, SNN à chaque step
- thêta : rythme hippocampique, consolidation mémoire
- sleep : consolidation offline, replay épisodique
"""

import torch


class StepScheduler:
    """
    Compteur global de steps avec fréquences par module.

    Chaque module du cerveau a sa propre fréquence d'activation,
    comme les différentes régions cérébrales qui oscillent à des
    fréquences différentes.
    """

    def __init__(
        self,
        snn_freq: int = 1,
        cfc_freq: int = 10,
        hippocampus_freq: int = 6,
        qwen_freq: int = 100,
        sleep_freq: int = 5000,
        arousal_boost_steps: int = 50,
    ):
        self.base_freqs = {
            "snn": snn_freq,
            "cfc": cfc_freq,
            "hippocampus": hippocampus_freq,
            "qwen": qwen_freq,
            "sleep": sleep_freq,
        }
        self.current_freqs = dict(self.base_freqs)
        self.arousal_boost_steps = arousal_boost_steps

        self.global_step = 0
        self._arousal_remaining = 0

    def step(self):
        """Incrémente le compteur global et décrémente l'arousal."""
        self.global_step += 1
        if self._arousal_remaining > 0:
            self._arousal_remaining -= 1
            if self._arousal_remaining == 0:
                self.current_freqs = dict(self.base_freqs)

    def should_update(self, module_name: str, step: int = None) -> bool:
        """Vérifie si le module doit être mis à jour à ce step."""
        if step is None:
            step = self.global_step
        freq = self.current_freqs.get(module_name, 1)
        if freq <= 0:
            return False
        return step % freq == 0

    def get_phase(self) -> str:
        """
        Retourne la phase cérébrale simulée.

        gamma : activité rapide (steps pairs)
        theta : rythme hippocampique (tous les 6 steps)
        idle  : repos
        """
        if self.global_step % 6 == 0:
            return "theta"
        if self.global_step % 2 == 0:
            return "gamma"
        return "idle"

    def arousal_boost(self):
        """
        Simule une décharge noradrénalique sous stress.
        Double toutes les fréquences pendant arousal_boost_steps steps.
        """
        self._arousal_remaining = self.arousal_boost_steps
        self.current_freqs = {
            name: max(1, freq // 2)
            for name, freq in self.base_freqs.items()
        }

    def save_state(self) -> dict:
        """Sauvegarde l'état du scheduler."""
        return {
            "global_step": self.global_step,
            "arousal_remaining": self._arousal_remaining,
            "current_freqs": dict(self.current_freqs),
            "base_freqs": dict(self.base_freqs),
        }

    def load_state(self, state: dict):
        """Restaure l'état du scheduler."""
        self.global_step = state["global_step"]
        self._arousal_remaining = state["arousal_remaining"]
        self.current_freqs = state["current_freqs"]
        self.base_freqs = state["base_freqs"]
