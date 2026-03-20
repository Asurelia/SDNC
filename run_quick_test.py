"""Quick test: do scaled-up circuits learn in their own space?"""
import sys, random, time
sys.path.insert(0, 'F:/sdnc')

import numpy as np
import torch
from collections import defaultdict
from torchvision import datasets
from sdnc.config import SDNCConfig
from sdnc.model import SDNCModel
from sdnc.utils.device import get_device

def get_class_indices(ds):
    ci = defaultdict(list)
    for i in range(len(ds)):
        _, l = ds[i]; ci[l].append(i)
    return ci

def sample_ep(ci, cls, nw, ks, qp):
    sel = random.sample(cls, nw)
    si, sl, qi, ql = [], [], [], []
    for nl, c in enumerate(sel):
        ch = random.sample(ci[c], min(ks+qp, len(ci[c])))
        si.extend(ch[:ks]); sl.extend([nl]*ks)
        qi.extend(ch[ks:ks+qp]); ql.extend([nl]*min(qp, len(ch)-ks))
    return si, sl, qi, ql

def proto(sup, slbl, qry, nw):
    protos = torch.stack([sup[slbl == c].mean(0) for c in range(nw)])
    return -torch.cdist(qry, protos)

device = get_device("cuda")
config = SDNCConfig(
    n_circuits=1000, use_expert_choice=False, sparsity_k=3,
    circuit_repr_dim=128,
    n_way=5, k_shot=1, query_per_class=15,
    train_episodes=2000, eval_episodes=100,
    hebbian_lr=1e-4, router_bias_gamma=0.01, device='cuda',
)

print(f"Circuit: {config.ncp_inter_neurons}+{config.ncp_command_neurons}+{config.circuit_output_dim} = {config.ncp_inter_neurons+config.ncp_command_neurons+config.circuit_output_dim} neurons")
print(f"Output dim: {config.circuit_output_dim} -> repr: {config.circuit_repr_dim}")

model = SDNCModel(config)
model.to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

preprocess = model.vision_encoder.preprocess
train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=preprocess)
test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=preprocess)
train_ci = get_class_indices(train_ds)
test_ci = get_class_indices(test_ds)

# Train 200 episodes
print("\nTraining 200 episodes...")
model.train()
accs = []
t0 = time.time()
for ep in range(1, 2001):
    si, sl, qi, ql = sample_ep(train_ci, list(range(6)), 5, 1, 15)
    sup = torch.stack([train_ds[i][0] for i in si]).to(device)
    qry = torch.stack([train_ds[i][0] for i in qi]).to(device)
    sl_t, ql_t = torch.tensor(sl, device=device), torch.tensor(ql, device=device)
    model.reset()
    model.learn(images=sup, labels=sl_t)
    with torch.no_grad():
        s = model.recognize(support_labels=sl_t, support_kwargs={"images": sup}, query_kwargs={"images": qry})
        accs.append((s.argmax(-1) == ql_t).float().mean().item())
    if ep % 100 == 0:
        w = min(ep, 50)
        print(f"  [Ep {ep}] {time.time()-t0:.0f}s | acc={sum(accs[-w:])/w:.2%} | random=20%")

# Quick eval: CLIP vs SDNC 5-way
print("\nEval 50 episodes 5-way...")
model.eval()
clip_5, sdnc_5 = [], []
for _ in range(100):
    si, sl, qi, ql = sample_ep(test_ci, list(range(5)), 5, 1, 10)
    sup = torch.stack([test_ds[i][0] for i in si]).to(device)
    qry = torch.stack([test_ds[i][0] for i in qi]).to(device)
    sl_t, ql_t = torch.tensor(sl, device=device), torch.tensor(ql, device=device)
    with torch.no_grad():
        sc = model.vision_encoder(sup); qc = model.vision_encoder(qry)
        clip_5.append((proto(sc, sl_t, qc, 5).argmax(-1) == ql_t).float().mean().item())
    model.reset()
    model.learn(images=sup, labels=sl_t)
    with torch.no_grad():
        s = model.recognize(support_labels=sl_t, support_kwargs={"images": sup}, query_kwargs={"images": qry})
        sdnc_5.append((s.argmax(-1) == ql_t).float().mean().item())

c5, s5 = np.mean(clip_5), np.mean(sdnc_5)
print(f"\nCLIP 5-way: {c5:.2%}")
print(f"SDNC 5-way: {s5:.2%}")
print(f"Delta: {s5-c5:+.2%}")
print(f"{'CIRCUITS WORK' if s5 > 0.25 else 'STILL RANDOM' if s5 < 0.22 else 'LEARNING BUT SLOW'}")
