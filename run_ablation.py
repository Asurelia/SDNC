"""SDNC Ablation Tests — Is SDNC actually doing anything beyond CLIP?

Test 1: CLIP-only prototypical baseline (no circuits)
Test 2: 5-way evaluation (harder, breaks trivial overlap)
Test 3: CLIP similarity vs post-circuit similarity
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

CIFAR_TEXT_PROMPTS = {
    i: [f"a photo of a {name}", f"an image of a {name}", f"a picture of a {name}"]
    for i, name in enumerate(CIFAR_NAMES)
}


def get_class_indices(dataset):
    ci = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        ci[label].append(idx)
    return ci


def sample_episode(class_indices, classes, n_way, k_shot, q_per_class):
    selected = random.sample(classes, n_way)
    s_idx, s_lbl, q_idx, q_lbl = [], [], [], []
    for new_lbl, cls in enumerate(selected):
        avail = class_indices[cls]
        chosen = random.sample(avail, min(k_shot + q_per_class, len(avail)))
        s_idx.extend(chosen[:k_shot])
        s_lbl.extend([new_lbl] * k_shot)
        q_idx.extend(chosen[k_shot:k_shot + q_per_class])
        q_lbl.extend([new_lbl] * min(q_per_class, len(chosen) - k_shot))
    return s_idx, s_lbl, q_idx, q_lbl, selected


def prototypical_classify(support_emb, support_labels, query_emb, n_way):
    """Pure prototypical classification on raw embeddings."""
    prototypes = []
    for c in range(n_way):
        mask = support_labels == c
        prototypes.append(support_emb[mask].mean(dim=0))
    prototypes = torch.stack(prototypes)
    dists = torch.cdist(query_emb, prototypes)
    return -dists  # negative distance = similarity score


def main():
    device = get_device("cuda")
    print("=" * 70)
    print("SDNC ABLATION TESTS — Is SDNC more than a CLIP wrapper?")
    print("=" * 70)
    print(f"Device: {device}\n")

    config = SDNCConfig(
        n_circuits=1000, use_expert_choice=False, sparsity_k=3,
        n_way=5, k_shot=1, query_per_class=15,
        train_episodes=200, eval_episodes=200,
        hebbian_lr=1e-4, router_bias_gamma=0.01,
        inter_circuit_hebbian_lr=1e-3, device='cuda',
    )

    # Build SDNC model
    print("Loading SDNC model...")
    model = SDNCModel(config)
    model.to(device)

    # Load data
    preprocess = model.vision_encoder.preprocess
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=preprocess)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=preprocess)

    # Use ALL 10 classes for harder evaluation
    all_classes = list(range(10))
    train_classes = list(range(6))
    test_classes_2way = [8, 9]
    test_classes_5way = [0, 1, 2, 3, 4]  # Use train classes for 5-way (more diverse)
    test_class_indices = get_class_indices(test_dataset)
    train_class_indices = get_class_indices(train_dataset)

    # ===== Train SDNC (200 episodes on train classes) =====
    print("\nTraining SDNC (200 episodes)...")
    model.train()
    for ep in range(1, 201):
        s_idx, s_lbl, q_idx, q_lbl, _ = sample_episode(
            train_class_indices, train_classes, 5, 1, 15)
        support_img = torch.stack([train_dataset[i][0] for i in s_idx]).to(device)
        s_lbl_t = torch.tensor(s_lbl, device=device)
        model.reset()
        model.learn(images=support_img, labels=s_lbl_t)
        if ep % 100 == 0:
            model.prune()
    model.eval()
    print("  Done.\n")

    N_EVAL = 200

    # ================================================================
    # TEST 1: CLIP-only baseline vs SDNC — 2-way
    # ================================================================
    print("=" * 70)
    print("TEST 1: CLIP-only vs SDNC — 2-way (ship vs truck)")
    print("=" * 70)

    clip_2way_accs = []
    sdnc_2way_accs = []

    for _ in tqdm(range(N_EVAL), desc="2-way"):
        s_idx, s_lbl, q_idx, q_lbl, _ = sample_episode(
            test_class_indices, test_classes_2way, 2, 1, 15)

        support_img = torch.stack([test_dataset[i][0] for i in s_idx]).to(device)
        query_img = torch.stack([test_dataset[i][0] for i in q_idx]).to(device)
        s_lbl_t = torch.tensor(s_lbl, device=device)
        q_lbl_t = torch.tensor(q_lbl, device=device)

        with torch.no_grad():
            # CLIP-only: raw embeddings, no circuits
            support_clip = model.vision_encoder(support_img)
            query_clip = model.vision_encoder(query_img)
            clip_scores = prototypical_classify(support_clip, s_lbl_t, query_clip, 2)
            clip_2way_accs.append((clip_scores.argmax(-1) == q_lbl_t).float().mean().item())

        # SDNC: full pipeline with circuits (needs gradients for learn)
        model.reset()
        model.learn(images=support_img, labels=s_lbl_t)
        with torch.no_grad():
            sdnc_scores = model.recognize(
                support_labels=s_lbl_t,
                support_kwargs={"images": support_img},
                query_kwargs={"images": query_img})
            sdnc_2way_accs.append((sdnc_scores.argmax(-1) == q_lbl_t).float().mean().item())

    clip_2 = np.mean(clip_2way_accs)
    sdnc_2 = np.mean(sdnc_2way_accs)
    clip_2_ci = 1.96 * np.std(clip_2way_accs) / np.sqrt(N_EVAL)
    sdnc_2_ci = 1.96 * np.std(sdnc_2way_accs) / np.sqrt(N_EVAL)
    delta_2 = sdnc_2 - clip_2

    print(f"\n  CLIP-only 2-way:  {clip_2:.2%} +/- {clip_2_ci:.2%}")
    print(f"  SDNC 2-way:      {sdnc_2:.2%} +/- {sdnc_2_ci:.2%}")
    print(f"  Delta:            {delta_2:+.2%} {'(SDNC helps)' if delta_2 > 0 else '(SDNC hurts)' if delta_2 < 0 else '(no difference)'}")

    # ================================================================
    # TEST 1b: CLIP-only vs SDNC — cross-modal image→text 2-way
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 1b: CLIP-only vs SDNC — cross-modal image→text 2-way")
    print("=" * 70)

    clip_cross_accs = []
    sdnc_cross_accs = []

    for _ in tqdm(range(N_EVAL), desc="Cross-modal 2-way"):
        selected = random.sample(test_classes_2way, 2)

        # Support: images
        support_imgs, support_labels = [], []
        for new_lbl, cls in enumerate(selected):
            idxs = random.sample(test_class_indices[cls], 1)
            for idx in idxs:
                support_imgs.append(test_dataset[idx][0])
                support_labels.append(new_lbl)

        support_img = torch.stack(support_imgs).to(device)
        s_lbl_t = torch.tensor(support_labels, device=device)

        # Query: text
        query_texts, query_labels = [], []
        for new_lbl, cls in enumerate(selected):
            for prompt in CIFAR_TEXT_PROMPTS[cls]:
                query_texts.append(prompt)
                query_labels.append(new_lbl)

        query_tokens = model.text_encoder.tokenize(query_texts).to(device)
        q_lbl_t = torch.tensor(query_labels, device=device)

        with torch.no_grad():
            # CLIP-only
            support_clip = model.vision_encoder(support_img)
            query_clip = model.text_encoder(query_tokens)
            clip_scores = prototypical_classify(support_clip, s_lbl_t, query_clip, 2)
            clip_cross_accs.append((clip_scores.argmax(-1) == q_lbl_t).float().mean().item())

        # SDNC
        model.reset()
        model.learn(images=support_img, labels=s_lbl_t)
        with torch.no_grad():
            sdnc_scores = model.recognize(
                support_labels=s_lbl_t,
                support_kwargs={"images": support_img},
                query_kwargs={"text_tokens": query_tokens})
            sdnc_cross_accs.append((sdnc_scores.argmax(-1) == q_lbl_t).float().mean().item())

    clip_c = np.mean(clip_cross_accs)
    sdnc_c = np.mean(sdnc_cross_accs)
    clip_c_ci = 1.96 * np.std(clip_cross_accs) / np.sqrt(N_EVAL)
    sdnc_c_ci = 1.96 * np.std(sdnc_cross_accs) / np.sqrt(N_EVAL)
    delta_c = sdnc_c - clip_c

    print(f"\n  CLIP-only cross:  {clip_c:.2%} +/- {clip_c_ci:.2%}")
    print(f"  SDNC cross:       {sdnc_c:.2%} +/- {sdnc_c_ci:.2%}")
    print(f"  Delta:            {delta_c:+.2%}")

    # ================================================================
    # TEST 2: 5-way evaluation + circuit overlap
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 2: 5-way evaluation (harder) + circuit overlap")
    print("=" * 70)

    # Use 5 classes from the training set for 5-way (we have enough classes)
    five_way_classes = list(range(5))  # airplane, automobile, bird, cat, deer
    five_way_indices = get_class_indices(test_dataset)

    clip_5way_accs = []
    sdnc_5way_accs = []
    overlap_per_episode = []

    for _ in tqdm(range(N_EVAL), desc="5-way"):
        s_idx, s_lbl, q_idx, q_lbl, _ = sample_episode(
            five_way_indices, five_way_classes, 5, 1, 10)

        support_img = torch.stack([test_dataset[i][0] for i in s_idx]).to(device)
        query_img = torch.stack([test_dataset[i][0] for i in q_idx]).to(device)
        s_lbl_t = torch.tensor(s_lbl, device=device)
        q_lbl_t = torch.tensor(q_lbl, device=device)

        with torch.no_grad():
            # CLIP-only
            support_clip = model.vision_encoder(support_img)
            query_clip = model.vision_encoder(query_img)
            clip_scores = prototypical_classify(support_clip, s_lbl_t, query_clip, 5)
            clip_5way_accs.append((clip_scores.argmax(-1) == q_lbl_t).float().mean().item())

        # SDNC
        model.reset()
        model.learn(images=support_img, labels=s_lbl_t)
        with torch.no_grad():
            sdnc_scores = model.recognize(
                support_labels=s_lbl_t,
                support_kwargs={"images": support_img},
                query_kwargs={"images": query_img})
            sdnc_5way_accs.append((sdnc_scores.argmax(-1) == q_lbl_t).float().mean().item())

            # Circuit overlap: get activation patterns per class
            class_circuits = {}
            for c in range(5):
                mask = s_lbl_t == c
                class_img = support_img[mask]
                result = model.forward(images=class_img)
                class_circuits[c] = set(result["active_indices"].view(-1).tolist())

            # Pairwise overlap
            total_overlap = 0
            n_pairs = 0
            for i in range(5):
                for j in range(i+1, 5):
                    union = len(class_circuits[i] | class_circuits[j])
                    inter = len(class_circuits[i] & class_circuits[j])
                    total_overlap += inter / max(1, union)
                    n_pairs += 1
            avg_overlap = total_overlap / n_pairs
            overlap_per_episode.append(avg_overlap)

    clip_5 = np.mean(clip_5way_accs)
    sdnc_5 = np.mean(sdnc_5way_accs)
    clip_5_ci = 1.96 * np.std(clip_5way_accs) / np.sqrt(N_EVAL)
    sdnc_5_ci = 1.96 * np.std(sdnc_5way_accs) / np.sqrt(N_EVAL)
    delta_5 = sdnc_5 - clip_5
    avg_overlap = np.mean(overlap_per_episode)

    print(f"\n  CLIP-only 5-way:  {clip_5:.2%} +/- {clip_5_ci:.2%}")
    print(f"  SDNC 5-way:      {sdnc_5:.2%} +/- {sdnc_5_ci:.2%}")
    print(f"  Delta:            {delta_5:+.2%}")
    print(f"  Random baseline:  20.00%")
    print(f"  Avg circuit overlap between classes: {avg_overlap:.2%}")
    print(f"  (0%=fully different circuits per class, 100%=same circuits for all)")

    # ================================================================
    # TEST 3: Cosine similarity CLIP vs post-circuit
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 3: CLIP embedding similarity vs post-circuit similarity")
    print("=" * 70)

    print(f"\n  {'Class':12s} | {'CLIP img↔txt':>12s} | {'SDNC img↔txt':>12s} | {'Delta':>8s} | {'CLIP img↔img':>12s} | {'SDNC img↔img':>12s}")
    print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}-+-{'-'*12}-+-{'-'*12}")

    clip_sims = []
    sdnc_sims = []

    for cls in range(10):
        imgs_idx = random.sample(test_class_indices[cls], min(10, len(test_class_indices[cls])))
        imgs = torch.stack([test_dataset[i][0] for i in imgs_idx]).to(device)
        texts = CIFAR_TEXT_PROMPTS[cls]
        tokens = model.text_encoder.tokenize(texts).to(device)

        with torch.no_grad():
            # CLIP raw
            img_clip = model.vision_encoder(imgs)  # (10, 512)
            txt_clip = model.text_encoder(tokens)  # (3, 512)
            clip_it = F.cosine_similarity(img_clip.mean(0, keepdim=True), txt_clip.mean(0, keepdim=True)).item()

            # CLIP image-image intra-class similarity
            clip_ii = F.cosine_similarity(img_clip[:5].mean(0, keepdim=True), img_clip[5:].mean(0, keepdim=True)).item()

            # SDNC post-circuit
            model.reset()
            img_result = model.forward(images=imgs)
            txt_result = model.forward(text_tokens=tokens)
            sdnc_it = F.cosine_similarity(
                img_result["output"].mean(0, keepdim=True),
                txt_result["output"].mean(0, keepdim=True)
            ).item()

            sdnc_ii = F.cosine_similarity(
                img_result["output"][:5].mean(0, keepdim=True),
                img_result["output"][5:].mean(0, keepdim=True)
            ).item()

            clip_sims.append(clip_it)
            sdnc_sims.append(sdnc_it)

            delta_sim = sdnc_it - clip_it
            print(f"  {CIFAR_NAMES[cls]:12s} | {clip_it:12.4f} | {sdnc_it:12.4f} | {delta_sim:+8.4f} | {clip_ii:12.4f} | {sdnc_ii:12.4f}")

    avg_clip = np.mean(clip_sims)
    avg_sdnc = np.mean(sdnc_sims)
    print(f"\n  Average        | {avg_clip:12.4f} | {avg_sdnc:12.4f} | {avg_sdnc - avg_clip:+8.4f}")

    # ================================================================
    # VERDICT
    # ================================================================
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    print(f"\n  Test 1 (2-way vision):     CLIP={clip_2:.2%} vs SDNC={sdnc_2:.2%}  delta={delta_2:+.2%}")
    print(f"  Test 1b (2-way cross):     CLIP={clip_c:.2%} vs SDNC={sdnc_c:.2%}  delta={delta_c:+.2%}")
    print(f"  Test 2 (5-way vision):     CLIP={clip_5:.2%} vs SDNC={sdnc_5:.2%}  delta={delta_5:+.2%}")
    print(f"  Test 2 circuit overlap:    {avg_overlap:.2%}")
    print(f"  Test 3 avg similarity:     CLIP={avg_clip:.4f} vs SDNC={avg_sdnc:.4f}")

    if abs(delta_2) < 0.02 and abs(delta_c) < 0.02 and abs(delta_5) < 0.02:
        print(f"\n  CONCLUSION: SDNC is a wrapper around CLIP.")
        print(f"  The circuits do not meaningfully improve over raw CLIP embeddings.")
        print(f"  The architecture works but the learning signal is too weak.")
    elif delta_2 > 0.02 or delta_5 > 0.02:
        print(f"\n  CONCLUSION: SDNC ADDS VALUE over CLIP baseline.")
        print(f"  The circuits improve classification by {max(delta_2, delta_5):+.2%} on vision.")
    else:
        print(f"\n  CONCLUSION: SDNC slightly degrades CLIP performance.")
        print(f"  The residual modulation is adding noise, not signal.")


if __name__ == "__main__":
    main()
