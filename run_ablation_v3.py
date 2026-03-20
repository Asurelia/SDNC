"""Ablation V3 — Circuit-centric architecture.

Compares 3 classifiers on the SAME data:
  A. CLIP-only prototypical (baseline — no circuits)
  B. SDNC circuit space (new architecture — circuits build representation)
  C. SDNC circuit space with Hebbian only (no prototypical loss backprop)

If B > A → circuits learn something CLIP can't do alone.
"""
import sys, random, time
sys.path.insert(0, 'F:/sdnc')

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from torchvision import datasets
from tqdm import tqdm

from sdnc.config import SDNCConfig
from sdnc.model import SDNCModel
from sdnc.utils.device import get_device

CIFAR_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
CIFAR_TEXT = {i: [f"a photo of a {n}", f"an image of a {n}", f"a picture of a {n}"]
              for i, n in enumerate(CIFAR_NAMES)}

def get_ci(d): return get_class_indices(d)
def get_class_indices(dataset):
    ci = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        ci[label].append(idx)
    return ci

def sample_ep(ci, classes, nw, ks, qp):
    sel = random.sample(classes, nw)
    si, sl, qi, ql = [], [], [], []
    for nl, c in enumerate(sel):
        ch = random.sample(ci[c], min(ks+qp, len(ci[c])))
        si.extend(ch[:ks]); sl.extend([nl]*ks)
        qi.extend(ch[ks:ks+qp]); ql.extend([nl]*min(qp, len(ch)-ks))
    return si, sl, qi, ql, sel

def proto(sup, slbl, qry, nw):
    protos = torch.stack([sup[slbl == c].mean(0) for c in range(nw)])
    return -torch.cdist(qry, protos)

def gini(v):
    s = v.sort().values.float(); n = len(s)
    if n == 0 or s.sum() == 0: return 0.0
    idx = torch.arange(1, n+1, dtype=torch.float32)
    return (2*(idx*s).sum()/(n*s.sum()) - (n+1)/n).item()

def main():
    device = get_device("cuda")
    print("=" * 70)
    print("ABLATION V3 — Circuit-Centric Architecture")
    print("Do circuits learn something CLIP can't?")
    print("=" * 70)

    config = SDNCConfig(
        n_circuits=1000, use_expert_choice=False, sparsity_k=3,
        circuit_repr_dim=128,
        n_way=5, k_shot=1, query_per_class=15,
        train_episodes=1000, eval_episodes=200,
        hebbian_lr=1e-4, router_bias_gamma=0.01,
        inter_circuit_hebbian_lr=1e-3, device='cuda',
    )

    model = SDNCModel(config)
    model.to(device)
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total_p:,} total, {train_p:,} trainable")
    print(f"Circuit repr dim: {config.circuit_repr_dim}")

    preprocess = model.vision_encoder.preprocess
    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=preprocess)
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=preprocess)
    train_ci = get_class_indices(train_ds)
    test_ci = get_class_indices(test_ds)
    train_classes = list(range(6))
    test_2 = [8, 9]
    test_5 = list(range(5))

    # ===== TRAIN 1000 episodes =====
    print(f"\nTraining {config.train_episodes} episodes...")
    model.train()
    accs = []
    t0 = time.time()

    for ep in range(1, config.train_episodes + 1):
        si, sl, qi, ql, _ = sample_ep(train_ci, train_classes, 5, 1, 15)
        sup = torch.stack([train_ds[i][0] for i in si]).to(device)
        qry = torch.stack([train_ds[i][0] for i in qi]).to(device)
        sl_t = torch.tensor(sl, device=device)
        ql_t = torch.tensor(ql, device=device)

        model.reset()
        model.learn(images=sup, labels=sl_t)

        with torch.no_grad():
            scores = model.recognize(support_labels=sl_t,
                support_kwargs={"images": sup}, query_kwargs={"images": qry})
            accs.append((scores.argmax(-1) == ql_t).float().mean().item())

        if ep % 100 == 0:
            model.prune()
            w = min(ep, 100)
            recent = sum(accs[-w:]) / w
            hist = model.circuit_bank.activation_history.cpu()
            n_act = (hist > 0).sum().item()
            g = gini(hist[hist > 0]) if n_act > 1 else 0

            # Circuit specialization
            class_circuits = {}
            with torch.no_grad():
                for cls in train_classes:
                    idxs = random.sample(train_ci[cls], min(10, len(train_ci[cls])))
                    imgs = torch.stack([train_ds[i][0] for i in idxs]).to(device)
                    r = model.forward(images=imgs)
                    class_circuits[cls] = set(r["active_indices"].view(-1).tolist())

            overlaps = []
            for i in range(len(train_classes)):
                for j in range(i+1, len(train_classes)):
                    ci_i = class_circuits[train_classes[i]]
                    ci_j = class_circuits[train_classes[j]]
                    u = len(ci_i | ci_j)
                    overlaps.append(len(ci_i & ci_j) / max(1, u))

            print(f"[Ep {ep:4d}] {time.time()-t0:.0f}s | acc={recent:.2%} | circuits={n_act} | gini={g:.3f} | overlap={np.mean(overlaps):.2%}")

    model.eval()
    N = 200

    # ===== TEST 1: 2-way vision =====
    print(f"\n{'='*70}\nTEST 1: CLIP-only vs SDNC circuits — 2-way vision\n{'='*70}")
    clip_2, sdnc_2 = [], []
    for _ in tqdm(range(N), desc="2-way"):
        si, sl, qi, ql, _ = sample_ep(test_ci, test_2, 2, 1, 15)
        sup = torch.stack([test_ds[i][0] for i in si]).to(device)
        qry = torch.stack([test_ds[i][0] for i in qi]).to(device)
        sl_t, ql_t = torch.tensor(sl, device=device), torch.tensor(ql, device=device)

        with torch.no_grad():
            sc = model.vision_encoder(sup)
            qc = model.vision_encoder(qry)
            clip_2.append((proto(sc, sl_t, qc, 2).argmax(-1) == ql_t).float().mean().item())

        model.reset()
        model.learn(images=sup, labels=sl_t)
        with torch.no_grad():
            s = model.recognize(support_labels=sl_t,
                support_kwargs={"images": sup}, query_kwargs={"images": qry})
            sdnc_2.append((s.argmax(-1) == ql_t).float().mean().item())

    # ===== TEST 2: 5-way vision + overlap =====
    print(f"\n{'='*70}\nTEST 2: CLIP-only vs SDNC — 5-way + circuit overlap\n{'='*70}")
    clip_5, sdnc_5, overlaps_5 = [], [], []
    for _ in tqdm(range(N), desc="5-way"):
        si, sl, qi, ql, _ = sample_ep(test_ci, test_5, 5, 1, 10)
        sup = torch.stack([test_ds[i][0] for i in si]).to(device)
        qry = torch.stack([test_ds[i][0] for i in qi]).to(device)
        sl_t, ql_t = torch.tensor(sl, device=device), torch.tensor(ql, device=device)

        with torch.no_grad():
            sc = model.vision_encoder(sup)
            qc = model.vision_encoder(qry)
            clip_5.append((proto(sc, sl_t, qc, 5).argmax(-1) == ql_t).float().mean().item())

        model.reset()
        model.learn(images=sup, labels=sl_t)
        with torch.no_grad():
            s = model.recognize(support_labels=sl_t,
                support_kwargs={"images": sup}, query_kwargs={"images": qry})
            sdnc_5.append((s.argmax(-1) == ql_t).float().mean().item())

            cc = {}
            for c in range(5):
                r = model.forward(images=sup[sl_t == c])
                cc[c] = set(r["active_indices"].view(-1).tolist())
            ov = []
            for i in range(5):
                for j in range(i+1, 5):
                    u = len(cc[i] | cc[j]); inter = len(cc[i] & cc[j])
                    ov.append(inter / max(1, u))
            overlaps_5.append(np.mean(ov))

    # ===== TEST 3: Cross-modal image→text =====
    print(f"\n{'='*70}\nTEST 3: CLIP-only vs SDNC — cross-modal image→text 2-way\n{'='*70}")
    clip_c, sdnc_c = [], []
    for _ in tqdm(range(N), desc="Cross-modal"):
        sel = random.sample(test_2, 2)
        si_l, sl_l = [], []
        for nl, c in enumerate(sel):
            for idx in random.sample(test_ci[c], 1):
                si_l.append(test_ds[idx][0]); sl_l.append(nl)
        sup = torch.stack(si_l).to(device)
        sl_t = torch.tensor(sl_l, device=device)
        qt, ql = [], []
        for nl, c in enumerate(sel):
            for p in CIFAR_TEXT[c]: qt.append(p); ql.append(nl)
        qtok = model.text_encoder.tokenize(qt).to(device)
        ql_t = torch.tensor(ql, device=device)

        with torch.no_grad():
            sc = model.vision_encoder(sup)
            qc = model.text_encoder(qtok)
            clip_c.append((proto(sc, sl_t, qc, 2).argmax(-1) == ql_t).float().mean().item())

        model.reset()
        model.learn(images=sup, labels=sl_t)
        with torch.no_grad():
            s = model.recognize(support_labels=sl_t,
                support_kwargs={"images": sup}, query_kwargs={"text_tokens": qtok})
            sdnc_c.append((s.argmax(-1) == ql_t).float().mean().item())

    # ===== TEST 4: Circuit repr geometry =====
    print(f"\n{'='*70}\nTEST 4: Circuit representation geometry\n{'='*70}")
    # Measure intra-class vs inter-class distances in circuit space
    intra_dists, inter_dists = [], []
    with torch.no_grad():
        class_reprs = {}
        for cls in range(10):
            idxs = random.sample(test_ci[cls], min(20, len(test_ci[cls])))
            imgs = torch.stack([test_ds[i][0] for i in idxs]).to(device)
            model.reset()
            r = model.forward(images=imgs)
            class_reprs[cls] = r["circuit_repr"]

        for cls in range(10):
            reprs = class_reprs[cls]
            # Intra-class: distance between pairs within same class
            if reprs.shape[0] >= 2:
                d = torch.cdist(reprs[:10], reprs[10:])
                intra_dists.extend(d.flatten().tolist())

        for i in range(10):
            for j in range(i+1, 10):
                d = torch.cdist(class_reprs[i][:5], class_reprs[j][:5])
                inter_dists.extend(d.flatten().tolist())

    intra_mean = np.mean(intra_dists) if intra_dists else 0
    inter_mean = np.mean(inter_dists) if inter_dists else 0
    separation = inter_mean / (intra_mean + 1e-8)

    print(f"  Intra-class distance (mean): {intra_mean:.4f}")
    print(f"  Inter-class distance (mean): {inter_mean:.4f}")
    print(f"  Separation ratio (inter/intra): {separation:.2f}x")
    print(f"  (>1 means classes are separated, >2 is good, >3 is excellent)")

    # ===== VERDICT =====
    c2m, s2m = np.mean(clip_2), np.mean(sdnc_2)
    c5m, s5m = np.mean(clip_5), np.mean(sdnc_5)
    ccm, scm = np.mean(clip_c), np.mean(sdnc_c)
    ci = lambda x: 1.96*np.std(x)/np.sqrt(len(x))

    print(f"\n{'='*70}\nVERDICT — Circuit-Centric Architecture\n{'='*70}")
    print(f"  Vision 2-way:  CLIP={c2m:.2%}±{ci(clip_2):.2%}  SDNC={s2m:.2%}±{ci(sdnc_2):.2%}  delta={s2m-c2m:+.2%}")
    print(f"  Vision 5-way:  CLIP={c5m:.2%}±{ci(clip_5):.2%}  SDNC={s5m:.2%}±{ci(sdnc_5):.2%}  delta={s5m-c5m:+.2%}")
    print(f"  Cross-modal:   CLIP={ccm:.2%}±{ci(clip_c):.2%}  SDNC={scm:.2%}±{ci(sdnc_c):.2%}  delta={scm-ccm:+.2%}")
    print(f"  Circuit overlap 5-way: {np.mean(overlaps_5):.2%}")
    print(f"  Repr separation: {separation:.2f}x")
    print(f"  Time: {time.time()-t0:.0f}s")

    wins = sum([s2m > c2m + 0.005, s5m > c5m + 0.005, scm > ccm + 0.005])
    if wins == 3:
        print(f"\n  CIRCUITS SURPASS CLIP on all 3 metrics!")
    elif wins >= 2:
        print(f"\n  Circuits mostly improve over CLIP ({wins}/3).")
    elif wins == 1:
        print(f"\n  Circuits improve on 1/3 metrics.")
    else:
        if s5m < c5m - 0.05:
            print(f"\n  Circuits significantly underperform CLIP — circuit repr not learned.")
        else:
            print(f"\n  Circuits match CLIP — no degradation but no improvement.")

if __name__ == "__main__":
    main()
