import torch
import torch.nn.functional as F


class HippocampalMemory:
    """
    Sparse Distributed Memory inspirée de Kanerva (1988).

    Principe :
      Les souvenirs sont stockés de façon DISTRIBUÉE sur N locations.
      Récupération par SIMILARITÉ PARTIELLE :
        fragment → souvenir complet (comme l'hippocampe biologique)
      Jamais effacée — s'enrichit en continu.

    Différence avec un vectorstore classique :
      - Adresses binaires aléatoires (hard locations)
      - Lecture/écriture dans un rayon de Hamming
      - Récupération par vote majoritaire
    """

    def __init__(
        self,
        address_dim: int = 256,
        content_dim: int = 2560,
        n_locations: int = 10000,
        activation_radius: int = 115
    ):
        self.address_dim = address_dim
        self.content_dim = content_dim
        self.n_locations = n_locations
        self.activation_radius = activation_radius

        # Hard locations — adresses binaires fixes et aléatoires
        self.addresses = torch.randint(
            0, 2, (n_locations, address_dim)
        ).float()

        # Contenu — s'accumule à chaque write
        self.contents = torch.zeros(n_locations, content_dim)

        # Compteurs d'accès par location
        self.access_counts = torch.zeros(n_locations)

        # Historique des métadonnées
        self.metadata = []

    def _to_binary_address(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Convertit un embedding continu en adresse binaire.
        Prend les address_dim premières dimensions, binarisées au seuil médian.
        """
        flat = embedding.detach().cpu().float().flatten()

        if flat.shape[0] > self.address_dim:
            flat = flat[:self.address_dim]
        elif flat.shape[0] < self.address_dim:
            flat = F.pad(flat, (0, self.address_dim - flat.shape[0]))

        return (flat > flat.median()).float()

    def _get_active_locations(self, address: torch.Tensor) -> torch.Tensor:
        """
        Retourne les indices des locations dans le rayon d'activation.
        Rayon défini par distance de Hamming ≤ activation_radius.
        """
        diffs = (self.addresses - address).abs().sum(dim=1)
        return (diffs <= self.activation_radius).nonzero(as_tuple=True)[0]

    def write(
        self,
        embedding: torch.Tensor,
        content: torch.Tensor,
        metadata: dict = None
    ):
        """
        Écrit un souvenir distribué sur toutes les locations actives.
        Écriture additive — les souvenirs s'accumulent.
        """
        address = self._to_binary_address(embedding)
        active = self._get_active_locations(address)

        if len(active) == 0:
            return

        content_flat = content.detach().cpu().float().flatten()
        if content_flat.shape[0] != self.content_dim:
            content_flat = F.interpolate(
                content_flat.unsqueeze(0).unsqueeze(0),
                size=self.content_dim,
                mode='linear',
                align_corners=False
            ).squeeze()

        # Écriture distribuée
        self.contents[active] += content_flat
        self.access_counts[active] += 1

        if metadata:
            self.metadata.append(metadata)

    def read(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Lit la mémoire depuis un embedding partiel.
        Retourne le souvenir reconstruit par vote majoritaire.
        """
        address = self._to_binary_address(embedding)
        active = self._get_active_locations(address)

        if len(active) == 0:
            return torch.zeros(self.content_dim)

        return self.contents[active].mean(dim=0)

    def consolidate(self, threshold: int = 5):
        """
        Élagage synaptique — supprime les locations peu accédées.
        Inspiré de la consolidation mémorielle pendant le sommeil.
        """
        low_access = self.access_counts < threshold
        self.contents[low_access] = 0
        self.access_counts[low_access] = 0

    def stats(self) -> dict:
        """Statistiques de la mémoire."""
        return {
            'memories_stored': len(self.metadata),
            'active_locations': (self.access_counts > 0).sum().item(),
            'total_locations': self.n_locations,
            'usage_pct': (self.access_counts > 0).float().mean().item() * 100
        }
