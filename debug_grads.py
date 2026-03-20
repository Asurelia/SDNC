"""Debug: check gradient flow through the model."""
import sys; sys.path.insert(0, 'F:/sdnc')
import torch
from sdnc.config import SDNCConfig
from sdnc.model import SDNCModel

device = torch.device('cuda')
config = SDNCConfig(n_circuits=20, use_expert_choice=False, sparsity_k=3, device='cuda')
model = SDNCModel(config).to(device)

# Mock encoder
class Mock(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.p = torch.nn.Linear(3*32*32, d)
        self.preprocess = None
    def forward(self, x):
        f = x.reshape(x.shape[0], -1)
        if f.shape[1] != self.p.in_features:
            f = f[:, :self.p.in_features] if f.shape[1] > self.p.in_features else torch.cat([f, torch.zeros(f.shape[0], self.p.in_features - f.shape[1], device=f.device)], 1)
        o = self.p(f)
        return o / (o.norm(dim=-1, keepdim=True) + 1e-8)

model.vision_encoder = Mock(config.input_dim).to(device)
model.encoder = model.vision_encoder

imgs = torch.randn(5, 3, 32, 32, device=device)
labels = torch.tensor([0, 1, 2, 3, 4], device=device)

# Forward
model.reset()
result = model.forward(images=imgs)

# Check which tensors require grad
print("=== Requires grad ===")
for k, v in result.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: shape={v.shape}, requires_grad={v.requires_grad}")

# Manual prototypical loss
repr_ = result["circuit_repr"]
protos = torch.stack([repr_[labels == c].mean(0) for c in range(5)])
dists = torch.cdist(repr_, protos)
log_p = torch.log_softmax(-dists, dim=-1)
loss = torch.nn.functional.nll_loss(log_p, labels)
print(f"\nLoss: {loss.item():.4f}")
print(f"Loss requires_grad: {loss.requires_grad}")

loss.backward()

# Check gradients
print("\n=== Gradient magnitudes ===")
for name, param in model.named_parameters():
    if param.grad is not None:
        g = param.grad.abs().mean().item()
        print(f"  {name:50s} grad={g:.8f} {'<-- ZERO' if g < 1e-10 else ''}")
    else:
        print(f"  {name:50s} NO GRAD")
