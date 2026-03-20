"""Phase 2 training: multimodal (vision+text) with Expert Choice routing.

Tests:
  1. Vision-only few-shot (compare with V1/V2)
  2. Cross-modal: learn on images → recognize via text descriptions
  3. Inter-circuit wiring evolution
"""

import random
import time
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from tqdm import tqdm

from sdnc.config import SDNCConfig
from sdnc.model import SDNCModel
from sdnc.utils.device import get_device

CIFAR_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

CIFAR_TEXT_PROMPTS = {
    i: [
        f"a photo of a {name}",
        f"an image of a {name}",
        f"a picture of a {name}",
    ]
    for i, name in enumerate(CIFAR_NAMES)
}


def get_class_indices(dataset) -> dict[int, list[int]]:
    class_indices: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_indices[label].append(idx)
    return class_indices


def sample_episode(class_indices, classes, n_way, k_shot, query_per_class):
    selected_classes = random.sample(classes, n_way)
    support_idx, support_labels, query_idx, query_labels = [], [], [], []

    for new_label, cls in enumerate(selected_classes):
        available = class_indices[cls]
        chosen = random.sample(available, min(k_shot + query_per_class, len(available)))
        support_idx.extend(chosen[:k_shot])
        support_labels.extend([new_label] * k_shot)
        query_idx.extend(chosen[k_shot:k_shot + query_per_class])
        query_labels.extend([new_label] * min(query_per_class, len(chosen) - k_shot))

    return support_idx, support_labels, query_idx, query_labels, selected_classes


def gini(values: torch.Tensor) -> float:
    sorted_vals = values.sort().values.float()
    n = len(sorted_vals)
    if n == 0 or sorted_vals.sum() == 0:
        return 0.0
    index = torch.arange(1, n + 1, dtype=torch.float32)
    return (2 * (index * sorted_vals).sum() / (n * sorted_vals.sum()) - (n + 1) / n).item()


def train_phase2(config: SDNCConfig | None = None):
    config = config or SDNCConfig()
    device = get_device(config.device)

    print("=" * 70)
    print("SDNC Phase 2 — Multimodal + Expert Choice + Inter-Circuit Wiring")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Circuits: {config.n_circuits}")
    print(f"Routing: {'ExpertChoice' if config.use_expert_choice else 'TokenChoice'}")
    print(f"  capacity_factor={config.expert_capacity_factor}" if config.use_expert_choice else f"  k={config.sparsity_k}")
    print(f"Inter-circuit wiring LR: {config.inter_circuit_hebbian_lr}")
    print(f"Episodes: {config.train_episodes} train, {config.eval_episodes} eval")
    print()

    # Build model
    print("Loading model...")
    t0 = time.time()
    model = SDNCModel(config)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model built in {time.time()-t0:.1f}s, {total_params:,} params")

    # Load data
    print("Loading CIFAR-10...")
    t0 = time.time()
    preprocess = model.vision_encoder.preprocess
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=preprocess)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=preprocess)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    train_classes = list(range(6))
    test_classes = list(range(8, 10))
    train_class_indices = get_class_indices(train_dataset)
    test_class_indices = get_class_indices(test_dataset)

    print(f"  Train: {[CIFAR_NAMES[c] for c in train_classes]}")
    print(f"  Test:  {[CIFAR_NAMES[c] for c in test_classes]}")

    # ===== TRAINING =====
    print("\n" + "=" * 70)
    print("TRAINING (Vision)")
    print("=" * 70)

    model.train()
    train_accuracies = []
    timing_start = time.time()
    milestones = [10, 50, 100, 200, 500, 1000]

    for ep in range(1, config.train_episodes + 1):
        s_idx, s_lbl, q_idx, q_lbl, _ = sample_episode(
            train_class_indices, train_classes, config.n_way, config.k_shot, config.query_per_class
        )

        support_img = torch.stack([train_dataset[i][0] for i in s_idx]).to(device)
        query_img = torch.stack([train_dataset[i][0] for i in q_idx]).to(device)
        s_lbl_t = torch.tensor(s_lbl, device=device)
        q_lbl_t = torch.tensor(q_lbl, device=device)

        model.reset()
        model.learn(images=support_img, labels=s_lbl_t)

        with torch.no_grad():
            scores = model.recognize(support_images=support_img, query_images=query_img, support_labels=s_lbl_t)
            acc = (scores.argmax(-1) == q_lbl_t).float().mean().item()
            train_accuracies.append(acc)

        if ep % 100 == 0:
            model.prune()

        if ep in milestones or ep % 100 == 0:
            elapsed = time.time() - timing_start
            window = min(ep, 50)
            recent = sum(train_accuracies[-window:]) / window
            overall = sum(train_accuracies) / len(train_accuracies)

            hist = model.circuit_bank.activation_history.cpu()
            n_active = (hist > 0).sum().item()
            gini_val = gini(hist[hist > 0]) if n_active > 1 else 0

            wiring = model.get_circuit_wiring_stats()

            print(f"\n[Ep {ep}/{config.train_episodes}] {elapsed:.0f}s")
            print(f"  Accuracy: recent={recent:.2%}, overall={overall:.2%}")
            print(f"  Circuits active: {n_active}/{config.n_circuits} ({n_active/config.n_circuits:.0%})")
            print(f"  Gini: {gini_val:.3f}")
            print(f"  Inter-circuit wiring: {wiring['nonzero_connections']} connections, density={wiring['density']:.4f}, max={wiring['max_weight']:.6f}")
            print(f"  Episodic memory: {len(model.episodic_memory)}")

    # ===== EVAL 1: Vision-only (compare V1/V2) =====
    print("\n" + "=" * 70)
    print("EVAL 1: Vision-only 2-way on test classes")
    print("=" * 70)

    model.eval()
    vision_accs = []
    n_way_eval = min(config.n_way, len(test_classes))

    for _ in tqdm(range(config.eval_episodes), desc="Vision eval"):
        s_idx, s_lbl, q_idx, q_lbl, _ = sample_episode(
            test_class_indices, test_classes, n_way_eval, config.k_shot, config.query_per_class
        )
        support_img = torch.stack([test_dataset[i][0] for i in s_idx]).to(device)
        query_img = torch.stack([test_dataset[i][0] for i in q_idx]).to(device)
        s_lbl_t = torch.tensor(s_lbl, device=device)
        q_lbl_t = torch.tensor(q_lbl, device=device)

        model.reset()
        model.learn(images=support_img, labels=s_lbl_t)

        with torch.no_grad():
            scores = model.recognize(support_images=support_img, query_images=query_img, support_labels=s_lbl_t)
            vision_accs.append((scores.argmax(-1) == q_lbl_t).float().mean().item())

    v_mean = np.mean(vision_accs)
    v_ci = 1.96 * np.std(vision_accs) / np.sqrt(len(vision_accs))
    print(f"  Vision accuracy: {v_mean:.2%} +/- {v_ci:.2%}")

    # ===== EVAL 2: Cross-modal (learn images → recognize text) =====
    print("\n" + "=" * 70)
    print("EVAL 2: Cross-modal — learn from IMAGES, query with TEXT")
    print("=" * 70)

    cross_accs = []
    for _ in tqdm(range(config.eval_episodes), desc="Cross-modal eval"):
        selected = random.sample(test_classes, n_way_eval)

        # Support: images
        support_imgs_list = []
        support_labels_list = []
        for new_lbl, cls in enumerate(selected):
            idxs = random.sample(test_class_indices[cls], config.k_shot)
            for idx in idxs:
                support_imgs_list.append(test_dataset[idx][0])
                support_labels_list.append(new_lbl)

        support_img = torch.stack(support_imgs_list).to(device)
        s_lbl_t = torch.tensor(support_labels_list, device=device)

        # Query: text descriptions
        query_texts = []
        query_labels = []
        for new_lbl, cls in enumerate(selected):
            for prompt in CIFAR_TEXT_PROMPTS[cls]:
                query_texts.append(prompt)
                query_labels.append(new_lbl)

        query_tokens = model.text_encoder.tokenize(query_texts).to(device)
        q_lbl_t = torch.tensor(query_labels, device=device)

        model.reset()
        model.learn(images=support_img, labels=s_lbl_t)

        with torch.no_grad():
            scores = model.recognize(
                support_images=support_img,
                query_text=query_tokens,
                support_labels=s_lbl_t,
            )
            cross_accs.append((scores.argmax(-1) == q_lbl_t).float().mean().item())

    c_mean = np.mean(cross_accs)
    c_ci = 1.96 * np.std(cross_accs) / np.sqrt(len(cross_accs))
    print(f"  Cross-modal accuracy: {c_mean:.2%} +/- {c_ci:.2%}")

    # ===== EVAL 3: Circuit activation overlap =====
    print("\n" + "=" * 70)
    print("EVAL 3: Do same circuits fire for 'cat image' and 'cat text'?")
    print("=" * 70)

    for cls in test_classes:
        # Image activations
        img_idx = random.sample(test_class_indices[cls], min(5, len(test_class_indices[cls])))
        imgs = torch.stack([test_dataset[i][0] for i in img_idx]).to(device)

        model.reset()
        img_result = model.forward(images=imgs)

        # Text activations
        texts = CIFAR_TEXT_PROMPTS[cls]
        tokens = model.text_encoder.tokenize(texts).to(device)
        text_result = model.forward(text_tokens=tokens)

        # Compare embedding similarity
        img_emb = img_result["embeddings"].mean(0)
        txt_emb = text_result["embeddings"].mean(0)
        cos_sim = F.cosine_similarity(img_emb.unsqueeze(0), txt_emb.unsqueeze(0)).item()

        # Compare outputs (after circuit processing)
        img_out = img_result["output"].mean(0)
        txt_out = text_result["output"].mean(0)
        cos_sim_out = F.cosine_similarity(img_out.unsqueeze(0), txt_out.unsqueeze(0)).item()

        print(f"  {CIFAR_NAMES[cls]:12s} | CLIP sim={cos_sim:.3f} | Post-circuit sim={cos_sim_out:.3f}")

    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print("PHASE 2 RESULTS SUMMARY")
    print("=" * 70)

    wiring_final = model.get_circuit_wiring_stats()
    hist = model.circuit_bank.activation_history.cpu()
    n_active = (hist > 0).sum().item()

    print(f"  Vision accuracy:      {v_mean:.2%} +/- {v_ci:.2%}")
    print(f"  Cross-modal accuracy: {c_mean:.2%} +/- {c_ci:.2%}")
    print(f"  Random baseline:      {1/n_way_eval:.2%}")
    print(f"  Circuits active:      {n_active}/{config.n_circuits}")
    print(f"  Gini:                 {gini(hist[hist > 0]) if n_active > 1 else 0:.3f}")
    print(f"  Inter-circuit wiring: {wiring_final['nonzero_connections']} connections")
    print(f"  Wiring density:       {wiring_final['density']:.4f}")
    print(f"  Total time:           {time.time()-timing_start:.0f}s")

    return model, train_accuracies, vision_accs, cross_accs


if __name__ == "__main__":
    train_phase2()
