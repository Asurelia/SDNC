"""SDNC Ablation V2 — 1000 episodes, adaptive alpha, entropy reg."""
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
CIFAR_TEXT_PROMPTS = {
    i: [f"a photo of a {n}", f"an image of a {n}", f"a picture of a {n}"]
    for i, n in enumerate(CIFAR_NAMES)
}

def get_class_indices(dataset):
    ci = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        ci[label].append(idx)
    return ci

def sample_episode(ci, classes, n_way, k_shot, q_per):
    selected = random.sample(classes, n_way)
    s_i, s_l, q_i, q_l = [], [], [], []
    for nl, c in enumerate(selected):
        ch = random.sample(ci[c], min(k_shot + q_per, len(ci[c])))
        s_i.extend(ch[:k_shot]); s_l.extend([nl]*k_shot)
        q_i.extend(ch[k_shot:k_shot+q_per]); q_l.extend([nl]*min(q_per, len(ch)-k_shot))
    return s_i, s_l, q_i, q_l, selected

def proto_classify(sup, slbl, qry, nw):
    protos = torch.stack([sup[slbl == c].mean(0) for c in range(nw)])
    return -torch.cdist(qry, protos)

def gini(v):
    s = v.sort().values.float()
    n = len(s)
    if n == 0 or s.sum() == 0: return 0.0
    idx = torch.arange(1, n+1, dtype=torch.float32)
    return (2*(idx*s).sum()/(n*s.sum()) - (n+1)/n).item()

def main():
    device = get_device("cuda")
    print("=" * 70)
    print("SDNC ABLATION V2 — Adaptive α + Entropy Reg + 1000 Episodes")
    print("=" * 70)

    config = SDNCConfig(
        n_circuits=1000, use_expert_choice=False, sparsity_k=3,
        n_way=5, k_shot=1, query_per_class=15,
        train_episodes=1000, eval_episodes=200,
        hebbian_lr=1e-4, router_bias_gamma=0.01,
        inter_circuit_hebbian_lr=1e-3, device='cuda',
    )

    model = SDNCModel(config)
    model.to(device)

    preprocess = model.vision_encoder.preprocess
    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=preprocess)
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=preprocess)

    train_ci = get_class_indices(train_ds)
    test_ci = get_class_indices(test_ds)
    train_classes = list(range(6))
    test_2way = [8, 9]
    test_5way = list(range(5))

    # ===== TRAIN 1000 episodes with logging =====
    print(f"\nTraining 1000 episodes...")
    model.train()
    train_accs = []
    t0 = time.time()

    for ep in range(1, 1001):
        si, sl, qi, ql, _ = sample_episode(train_ci, train_classes, 5, 1, 15)
        sup = torch.stack([train_ds[i][0] for i in si]).to(device)
        qry = torch.stack([train_ds[i][0] for i in qi]).to(device)
        sl_t = torch.tensor(sl, device=device)
        ql_t = torch.tensor(ql, device=device)

        model.reset()
        model.learn(images=sup, labels=sl_t)

        with torch.no_grad():
            scores = model.recognize(support_labels=sl_t,
                support_kwargs={"images": sup}, query_kwargs={"images": qry})
            train_accs.append((scores.argmax(-1) == ql_t).float().mean().item())

        if ep % 100 == 0:
            model.prune()
            w = min(ep, 100)
            recent = sum(train_accs[-w:]) / w
            hist = model.circuit_bank.activation_history.cpu()
            n_act = (hist > 0).sum().item()
            g = gini(hist[hist > 0]) if n_act > 1 else 0
            alpha_val = model.alpha.mean().item()
            entropy = model.router._last_entropy if hasattr(model.router, '_last_entropy') else 0

            # Circuit specialization: which circuits are used per class
            class_circuit_usage = {}
            with torch.no_grad():
                for cls in train_classes:
                    idxs = random.sample(train_ci[cls], min(10, len(train_ci[cls])))
                    imgs = torch.stack([train_ds[i][0] for i in idxs]).to(device)
                    result = model.forward(images=imgs)
                    class_circuit_usage[cls] = set(result["active_indices"].view(-1).tolist())

            # Pairwise circuit overlap between classes
            overlaps = []
            for i in range(len(train_classes)):
                for j in range(i+1, len(train_classes)):
                    ci_i = class_circuit_usage[train_classes[i]]
                    ci_j = class_circuit_usage[train_classes[j]]
                    union = len(ci_i | ci_j)
                    inter = len(ci_i & ci_j)
                    overlaps.append(inter / max(1, union))
            avg_overlap = np.mean(overlaps)

            unique_per_class = [len(class_circuit_usage[c]) for c in train_classes]

            print(f"[Ep {ep:4d}] {time.time()-t0:.0f}s | acc={recent:.2%} | α={alpha_val:.4f} | entropy={entropy:.2f} | circuits={n_act} | gini={g:.3f} | overlap={avg_overlap:.2%} | unique/class={np.mean(unique_per_class):.1f}")

    model.eval()
    N = 200

    # ===== TEST 1: CLIP vs SDNC 2-way =====
    print(f"\n{'='*70}\nTEST 1: CLIP vs SDNC — 2-way vision\n{'='*70}")
    clip_2, sdnc_2 = [], []
    for _ in tqdm(range(N), desc="2-way"):
        si, sl, qi, ql, _ = sample_episode(test_ci, test_2way, 2, 1, 15)
        sup = torch.stack([test_ds[i][0] for i in si]).to(device)
        qry = torch.stack([test_ds[i][0] for i in qi]).to(device)
        sl_t, ql_t = torch.tensor(sl, device=device), torch.tensor(ql, device=device)

        with torch.no_grad():
            sc = model.vision_encoder(sup)
            qc = model.vision_encoder(qry)
            clip_2.append((proto_classify(sc, sl_t, qc, 2).argmax(-1) == ql_t).float().mean().item())

        model.reset()
        model.learn(images=sup, labels=sl_t)
        with torch.no_grad():
            s = model.recognize(support_labels=sl_t, support_kwargs={"images": sup}, query_kwargs={"images": qry})
            sdnc_2.append((s.argmax(-1) == ql_t).float().mean().item())

    # ===== TEST 1b: cross-modal 2-way =====
    print(f"\n{'='*70}\nTEST 1b: CLIP vs SDNC — cross-modal image→text 2-way\n{'='*70}")
    clip_c, sdnc_c = [], []
    for _ in tqdm(range(N), desc="Cross 2-way"):
        sel = random.sample(test_2way, 2)
        si_l, sl_l = [], []
        for nl, c in enumerate(sel):
            for idx in random.sample(test_ci[c], 1):
                si_l.append(test_ds[idx][0]); sl_l.append(nl)
        sup = torch.stack(si_l).to(device)
        sl_t = torch.tensor(sl_l, device=device)
        qt, ql = [], []
        for nl, c in enumerate(sel):
            for p in CIFAR_TEXT_PROMPTS[c]: qt.append(p); ql.append(nl)
        qtok = model.text_encoder.tokenize(qt).to(device)
        ql_t = torch.tensor(ql, device=device)

        with torch.no_grad():
            sc = model.vision_encoder(sup)
            qc = model.text_encoder(qtok)
            clip_c.append((proto_classify(sc, sl_t, qc, 2).argmax(-1) == ql_t).float().mean().item())

        model.reset()
        model.learn(images=sup, labels=sl_t)
        with torch.no_grad():
            s = model.recognize(support_labels=sl_t, support_kwargs={"images": sup}, query_kwargs={"text_tokens": qtok})
            sdnc_c.append((s.argmax(-1) == ql_t).float().mean().item())

    # ===== TEST 2: 5-way + overlap =====
    print(f"\n{'='*70}\nTEST 2: CLIP vs SDNC — 5-way vision + circuit overlap\n{'='*70}")
    clip_5, sdnc_5, overlaps_5 = [], [], []
    for _ in tqdm(range(N), desc="5-way"):
        si, sl, qi, ql, _ = sample_episode(test_ci, test_5way, 5, 1, 10)
        sup = torch.stack([test_ds[i][0] for i in si]).to(device)
        qry = torch.stack([test_ds[i][0] for i in qi]).to(device)
        sl_t, ql_t = torch.tensor(sl, device=device), torch.tensor(ql, device=device)

        with torch.no_grad():
            sc = model.vision_encoder(sup)
            qc = model.vision_encoder(qry)
            clip_5.append((proto_classify(sc, sl_t, qc, 5).argmax(-1) == ql_t).float().mean().item())

        model.reset()
        model.learn(images=sup, labels=sl_t)
        with torch.no_grad():
            s = model.recognize(support_labels=sl_t, support_kwargs={"images": sup}, query_kwargs={"images": qry})
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

    # ===== TEST 3: Similarity =====
    print(f"\n{'='*70}\nTEST 3: CLIP vs SDNC cosine similarity\n{'='*70}")
    print(f"  {'Class':12s} | {'CLIP i↔t':>8s} | {'SDNC i↔t':>8s} | {'Delta':>7s}")
    print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}")
    cs, ss = [], []
    for cls in range(10):
        idxs = random.sample(test_ci[cls], min(10, len(test_ci[cls])))
        imgs = torch.stack([test_ds[i][0] for i in idxs]).to(device)
        toks = model.text_encoder.tokenize(CIFAR_TEXT_PROMPTS[cls]).to(device)
        with torch.no_grad():
            ic = model.vision_encoder(imgs)
            tc = model.text_encoder(toks)
            clip_sim = F.cosine_similarity(ic.mean(0, keepdim=True), tc.mean(0, keepdim=True)).item()
            model.reset()
            ir = model.forward(images=imgs)
            tr = model.forward(text_tokens=toks)
            sdnc_sim = F.cosine_similarity(ir["output"].mean(0, keepdim=True), tr["output"].mean(0, keepdim=True)).item()
        cs.append(clip_sim); ss.append(sdnc_sim)
        d = sdnc_sim - clip_sim
        print(f"  {CIFAR_NAMES[cls]:12s} | {clip_sim:8.4f} | {sdnc_sim:8.4f} | {d:+7.4f}")

    ac, as_ = np.mean(cs), np.mean(ss)
    print(f"  {'Average':12s} | {ac:8.4f} | {as_:8.4f} | {as_-ac:+7.4f}")

    # ===== VERDICT =====
    c2m, s2m = np.mean(clip_2), np.mean(sdnc_2)
    ccm, scm = np.mean(clip_c), np.mean(sdnc_c)
    c5m, s5m = np.mean(clip_5), np.mean(sdnc_5)
    c2ci = 1.96*np.std(clip_2)/np.sqrt(N); s2ci = 1.96*np.std(sdnc_2)/np.sqrt(N)
    ccci = 1.96*np.std(clip_c)/np.sqrt(N); scci = 1.96*np.std(sdnc_c)/np.sqrt(N)
    c5ci = 1.96*np.std(clip_5)/np.sqrt(N); s5ci = 1.96*np.std(sdnc_5)/np.sqrt(N)

    print(f"\n{'='*70}\nVERDICT\n{'='*70}")
    print(f"  Vision 2-way:  CLIP={c2m:.2%}±{c2ci:.2%}  SDNC={s2m:.2%}±{s2ci:.2%}  delta={s2m-c2m:+.2%}")
    print(f"  Cross 2-way:   CLIP={ccm:.2%}±{ccci:.2%}  SDNC={scm:.2%}±{scci:.2%}  delta={scm-ccm:+.2%}")
    print(f"  Vision 5-way:  CLIP={c5m:.2%}±{c5ci:.2%}  SDNC={s5m:.2%}±{s5ci:.2%}  delta={s5m-c5m:+.2%}")
    print(f"  Circuit overlap 5-way: {np.mean(overlaps_5):.2%}")
    print(f"  Similarity:    CLIP={ac:.4f}  SDNC={as_:.4f}  delta={as_-ac:+.4f}")
    print(f"  Alpha (mean):  {model.alpha.mean().item():.5f}")
    print(f"  Router entropy: {model.router._last_entropy:.3f}")

    wins = sum([s2m > c2m, scm > ccm, s5m > c5m])
    print(f"\n  SDNC wins {wins}/3 metrics vs CLIP baseline.")
    if wins == 3:
        print(f"  SDNC SURPASSES CLIP on all metrics!")
    elif wins >= 2:
        print(f"  SDNC mostly improves over CLIP.")
    elif abs(s2m-c2m) < 0.01 and abs(scm-ccm) < 0.01:
        print(f"  SDNC is a CLIP wrapper — no meaningful difference.")
    else:
        print(f"  Mixed results — SDNC helps on some, hurts on others.")

if __name__ == "__main__":
    main()
