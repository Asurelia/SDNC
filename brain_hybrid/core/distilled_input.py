"""
Couche d'entrée enrichie par distillation teacher → student.

Fusionne les représentations brutes du student (Qwen3-4B, 2560-dim)
avec les représentations projetées du teacher (via ProjectionBridge, 2560-dim).

La fusion est pondérée par un alpha appris par couche + un gate dynamique :
  enriched = alpha * student_reps + (1 - alpha) * gate * teacher_projected

Alpha initialisé à 0.8 → le student domine au début.
Le gate ferme automatiquement si le signal teacher est peu fiable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class DistilledInputLayer(nn.Module):
    """
    Fusionne les représentations student et teacher projetées.

    Reçoit :
      - student_reps : hidden_states bruts du student LLM
      - teacher_projected : hidden_states du teacher passés par le ProjectionBridge

    Produit :
      - enriched_reps : fusion pondérée par alpha + gate
      - fusion_stats : monitoring (alphas, gates, cosine par couche)

    Le gate apprend quand le signal teacher est fiable.
    """

    def __init__(
        self,
        n_layers: int = 4,
        hidden_dim: int = 2560,
        alpha_init: float = 0.8,
    ):
        """
        Initialise la couche de fusion.

        Args:
            n_layers: Nombre de couches d'intercept (4).
            hidden_dim: Dimension des représentations (2560).
            alpha_init: Poids initial du student (0.8 = student domine).
        """
        super().__init__()

        # Un alpha par couche — appris, clampé [0, 1]
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(alpha_init))
            for _ in range(n_layers)
        ])

        # Gate dynamique — décide quand le signal teacher est fiable
        self.gate = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        student_reps: List[torch.Tensor],
        teacher_projected: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], dict]:
        """
        Fusionne student + teacher projeté.

        Args:
            student_reps: 4 tenseurs (batch, seq_len, 2560) du student.
            teacher_projected: 4 tenseurs (batch, seq_len, 2560) du bridge.

        Returns:
            (enriched_reps, fusion_stats)
        """
        enriched = []
        stats = {"alphas": [], "gates": [], "cosine_per_layer": []}

        for i, (s_rep, t_rep) in enumerate(zip(student_reps, teacher_projected)):
            alpha = self.alphas[i].clamp(0.0, 1.0)

            # Aligner seq_len
            s_min = min(s_rep.shape[1], t_rep.shape[1])
            s_rep_a = s_rep[:, :s_min, :]
            t_rep_a = t_rep[:, :s_min, :]

            # Gate dynamique
            combined = torch.cat([s_rep_a, t_rep_a], dim=-1)
            gate = self.sigmoid(self.gate(combined))

            # Fusion pondérée + gating
            enriched_rep = alpha * s_rep_a + (1 - alpha) * gate * t_rep_a
            enriched.append(enriched_rep)

            # Stats monitoring
            with torch.no_grad():
                cos = F.cosine_similarity(
                    s_rep_a.float().mean(dim=1),
                    t_rep_a.float().mean(dim=1),
                    dim=-1
                ).mean().item()

            stats["alphas"].append(alpha.item())
            stats["gates"].append(gate.mean().item())
            stats["cosine_per_layer"].append(cos)

        return enriched, stats

    def get_teacher_influence(self) -> float:
        """
        Mesure l'influence moyenne du teacher sur le student.

        (1 - alpha) par couche, moyenné.
        Démarre à ~0.2, augmente si le teacher aide.
        """
        influences = [(1 - a.clamp(0.0, 1.0).item()) for a in self.alphas]
        return sum(influences) / len(influences)
