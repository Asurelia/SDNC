import torch


class STDPLearning:
    """
    Spike-Timing-Dependent Plasticity + Neuromodulation dopamine-like.

    Règle biologique STDP :
      - Pré s'active AVANT post → renforcement (LTP)
      - Post s'active AVANT pré  → affaiblissement (LTD)

    Neuromodulation dopamine-like :
      - Signal entre 0 et 1
      - Haute erreur de prédiction → surprise → plus de dopamine
      - Plus de dopamine → amplitude d'apprentissage plus forte
      - Inspiré de la théorie TD (Temporal Difference)
    """

    def __init__(
        self,
        lr_plus: float = 0.01,
        lr_minus: float = 0.01
    ):
        self.lr_plus = lr_plus
        self.lr_minus = lr_minus
        self.dopamine_signal = 0.5

    def update_dopamine(
        self,
        prediction_error: float,
        threshold: float = 0.3
    ):
        """
        Met à jour le signal dopamine depuis l'erreur de prédiction.
        Haute erreur = surprise = plus de dopamine = apprentissage renforcé.
        """
        normalized = min(1.0, abs(prediction_error) / threshold)
        self.dopamine_signal = 0.9 * self.dopamine_signal + 0.1 * normalized
        self.dopamine_signal = max(0.0, min(1.0, self.dopamine_signal))

    def apply(
        self,
        weight: torch.Tensor,
        pre_trace: torch.Tensor,
        post_trace: torch.Tensor
    ) -> torch.Tensor:
        """
        Applique STDP sur un tenseur de poids.

        Version stable avec outer product propre et interpolation
        pour gérer les dimensions variables des poids CfC.
        Modulé par le signal dopamine.
        """
        with torch.no_grad():
            pre_mean = pre_trace.detach().float().mean(0).mean(0)
            post_mean = post_trace.detach().float().mean(0).mean(0)

            out_dim = weight.shape[0]
            in_dim = weight.shape[1]

            pre_r = torch.nn.functional.interpolate(
                pre_mean.unsqueeze(0).unsqueeze(0),
                size=in_dim,
                mode='linear',
                align_corners=False
            ).squeeze()

            post_r = torch.nn.functional.interpolate(
                post_mean.unsqueeze(0).unsqueeze(0),
                size=out_dim,
                mode='linear',
                align_corners=False
            ).squeeze()

            # LTP — pré avant post = renforcement (out_dim x in_dim)
            ltp = post_r.unsqueeze(1) * pre_r.unsqueeze(0)

            # LTD — post avant pré = affaiblissement (out_dim x in_dim)
            # Interpoler dans l'autre sens pour l'asymétrie
            pre_as_out = torch.nn.functional.interpolate(
                pre_mean.unsqueeze(0).unsqueeze(0),
                size=out_dim,
                mode='linear',
                align_corners=False
            ).squeeze()
            post_as_in = torch.nn.functional.interpolate(
                post_mean.unsqueeze(0).unsqueeze(0),
                size=in_dim,
                mode='linear',
                align_corners=False
            ).squeeze()
            ltd = pre_as_out.unsqueeze(1) * post_as_in.unsqueeze(0)

            delta = self.lr_plus * self.dopamine_signal * ltp
            delta -= self.lr_minus * self.dopamine_signal * ltd

            # Clamp strict pour la stabilité numérique
            delta = torch.clamp(delta, -1e-3, 1e-3)

            weight.data += delta.to(weight.device)

        return weight
