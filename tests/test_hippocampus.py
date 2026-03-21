import torch
from brain_hybrid.memory.hippocampus import HippocampalMemory


def test_write_read_roundtrip():
    """Un souvenir écrit doit être récupérable."""
    h = HippocampalMemory(address_dim=64, content_dim=128, n_locations=1000)
    emb = torch.randn(128)
    content = torch.randn(128)
    h.write(emb, content)
    recalled = h.read(emb)
    sim = torch.nn.functional.cosine_similarity(
        content.unsqueeze(0), recalled.unsqueeze(0)
    ).item()
    assert sim > 0.5, f"Recall trop faible : {sim:.3f}"


def test_metadata_stockee():
    """Les métadonnées doivent être stockées après write."""
    h = HippocampalMemory(address_dim=64, content_dim=128, n_locations=1000)
    h.write(torch.randn(128), torch.randn(128), metadata={'test': True})
    assert len(h.metadata) == 1
