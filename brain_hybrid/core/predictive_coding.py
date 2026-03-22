"""
Codage prédictif hiérarchique inspiré de Karl Friston.

Formalise ce que BRAIN-HYBRID fait déjà implicitement :
chaque couche prédit l'entrée de la couche inférieure (top-down),
et l'erreur de prédiction est le signal d'apprentissage.

La précision (1/variance) module l'importance de chaque erreur,
comme dans la théorie du cerveau bayésien.
"""

import torch
import torch.nn as nn
from typing import List


class PredictiveCodingLayer:
    """
    Couche de codage prédictif wrappant un BrainModule existant.

    Attributs :
      prediction : dernière prédiction top-down
      error      : erreur = actual - prediction
      precision  : 1 / variance(error) — confiance dans la prédiction
    """

    def __init__(self, brain_module, pc_lr: float = 0.001):
        self.module = brain_module
        self.pc_lr = pc_lr
        self.prediction = None
        self.error = None
        self.precision = 1.0

    def compute_error(self, actual: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
        """
        Calcule l'erreur de prédiction et la précision.

        actual    : hidden states réels de la couche
        predicted : prédiction top-down depuis la couche supérieure
        """
        self.error = actual.to(predicted.device).float() - predicted.float()
        self.precision = 1.0 / (self.error.var().item() + 1e-8)
        return self.error

    def pc_update(self, stdp_learner, dopamine_signal: float):
        """
        Mise à jour des poids via règle locale de codage prédictif.

        Δw = pc_lr * precision * dopamine * (pre_trace * error.mean())

        Compatible avec STDPLearning existant.
        """
        if self.error is None or self.module.pre_trace is None:
            return

        with torch.no_grad():
            error_magnitude = self.error.abs().mean().item()
            modulation = self.pc_lr * self.precision * dopamine_signal * error_magnitude

            for param in self.module.cfc.parameters():
                if param.requires_grad and len(param.shape) == 2:
                    # Appliquer STDP modulé par la précision
                    stdp_learner.apply(
                        param,
                        self.module.pre_trace,
                        self.module.post_trace,
                    )
                    # Ajout PC : modulation supplémentaire proportionnelle à la précision
                    noise = torch.randn_like(param) * modulation * 1e-4
                    param.data += noise.clamp(-1e-4, 1e-4)

    def get_pc_loss(self) -> float:
        """Erreur PC pour monitoring (pas pour backprop)."""
        if self.error is None:
            return 0.0
        return self.error.abs().mean().item()


class HierarchicalPC:
    """
    Codage prédictif hiérarchique — 4 couches correspondant aux points d'intercept Qwen.

    Couche[i] prédit l'entrée de couche[i-1] (top-down).
    L'erreur de couche[i] = signal d'apprentissage de couche[i-1].
    """

    def __init__(self, brain_modules, pc_lr: float = 0.001):
        self.layers = [
            PredictiveCodingLayer(module, pc_lr=pc_lr)
            for module in brain_modules
        ]

    def forward(self, layer_reps: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Passe forward hiérarchique.

        layer_reps : liste de 4 tenseurs (hidden states aux couches d'intercept)
        Retourne : liste de 3 erreurs (entre couches adjacentes)
        """
        errors = []

        for i, pc_layer in enumerate(self.layers):
            # Forward du BrainModule → prédiction de la couche suivante
            prediction, spikes = pc_layer.module(layer_reps[i])
            pc_layer.prediction = prediction

            # Erreur vs couche suivante (sauf la dernière)
            if i < len(layer_reps) - 1:
                error = pc_layer.compute_error(layer_reps[i + 1], prediction)
                errors.append(error)

        return errors

    def update_all(
        self,
        errors: List[torch.Tensor],
        dopamine_signals: list,
        stdp_learners: list,
    ):
        """
        Met à jour tous les poids via codage prédictif.

        errors          : liste d'erreurs par couche (sortie de forward())
        dopamine_signals: liste de signaux dopamine
        stdp_learners   : liste de STDPLearning
        """
        for i, pc_layer in enumerate(self.layers):
            if i < len(errors) and i < len(dopamine_signals):
                pc_layer.pc_update(stdp_learners[i], dopamine_signals[i])

    def get_errors(self) -> List[float]:
        """Retourne les erreurs PC de chaque couche."""
        return [layer.get_pc_loss() for layer in self.layers]

    def get_precisions(self) -> List[float]:
        """Retourne les précisions de chaque couche."""
        return [layer.precision for layer in self.layers]
