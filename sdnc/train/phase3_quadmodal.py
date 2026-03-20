"""Phase 3 training: quad-modal (vision, text, audio, signals).

Tests whether the same circuits fire across all 4 modalities
for the same concept — the ultimate test of universal representation.
"""

import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from tqdm import tqdm

from sdnc.config import SDNCConfig
from sdnc.model import SDNCModel
from sdnc.encoders.signal_encoder import SignalEncoder
from sdnc.utils.device import get_device

CIFAR_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

CIFAR_TEXT_PROMPTS = {
    i: [f"a photo of a {name}", f"an image of a {name}", f"a picture of a {name}"]
    for i, name in enumerate(CIFAR_NAMES)
}

# Simulated audio features per class (since we can't load real audio in CIFAR)
# In production these would come from Whisper on actual audio clips.
# Here we use CLIP text features of onomatopoeia/descriptions as proxy.
CIFAR_AUDIO_DESCRIPTIONS = {
    0: ["airplane engine sound", "jet flying overhead"],
    1: ["car engine revving", "automobile horn honking"],
    2: ["bird chirping", "bird singing tweet tweet"],
    3: ["cat meowing", "cat purring sound"],
    4: ["deer footsteps in forest", "deer call in nature"],
    5: ["dog barking", "dog panting woof woof"],
    6: ["frog croaking ribbit", "frog in pond water splash"],
    7: ["horse galloping hooves", "horse neighing sound"],
    8: ["ship horn foghorn", "waves crashing against ship"],
    9: ["truck engine diesel rumble", "truck beeping reversing"],
}

# Simulated signal contexts per class
CIFAR_SIGNAL_CONTEXTS = {
    0: {"mouse_x": 0.5, "mouse_y": 0.3, "timestamp": 36000, "app_id": 1},   # looking at sky
    1: {"mouse_x": 0.5, "mouse_y": 0.7, "timestamp": 54000, "app_id": 2},   # traffic cam
    2: {"mouse_x": 0.3, "mouse_y": 0.2, "timestamp": 21600, "app_id": 1},   # morning birdwatch
    3: {"mouse_x": 0.5, "mouse_y": 0.5, "timestamp": 72000, "app_id": 3},   # pet photos
    4: {"mouse_x": 0.7, "mouse_y": 0.4, "timestamp": 18000, "app_id": 1},   # nature cam
    5: {"mouse_x": 0.4, "mouse_y": 0.6, "timestamp": 64800, "app_id": 3},   # pet photos
    6: {"mouse_x": 0.6, "mouse_y": 0.8, "timestamp": 3600, "app_id": 1},    # pond cam
    7: {"mouse_x": 0.8, "mouse_y": 0.3, "timestamp": 43200, "app_id": 4},   # ranch app
    8: {"mouse_x": 0.2, "mouse_y": 0.5, "timestamp": 50400, "app_id": 5},   # marine app
    9: {"mouse_x": 0.9, "mouse_y": 0.7, "timestamp": 57600, "app_id": 2},   # traffic
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


def train_phase3(config: SDNCConfig | None = None):
    config = config or SDNCConfig()
    device = get_device(config.device)

    print("=" * 70)
    print("SDNC Phase 3 — Quad-Modal (Vision + Text + Audio + Signals)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Circuits: {config.n_circuits}, k={config.sparsity_k}")
    print(f"Episodes: {config.train_episodes} train, {config.eval_episodes} eval")
    print()

    # Build model
    print("Loading model...")
    t0 = time.time()
    model = SDNCModel(config)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Built in {time.time()-t0:.1f}s, {total_params:,} params")

    # Load data
    print("Loading CIFAR-10...")
    preprocess = model.vision_encoder.preprocess
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=preprocess)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=preprocess)

    train_classes = list(range(6))
    test_classes = list(range(8, 10))
    train_class_indices = get_class_indices(train_dataset)
    test_class_indices = get_class_indices(test_dataset)

    print(f"  Train: {[CIFAR_NAMES[c] for c in train_classes]}")
    print(f"  Test:  {[CIFAR_NAMES[c] for c in test_classes]}")

    # ===== TRAINING (vision only, same as Phase 2) =====
    print("\n" + "=" * 70)
    print("TRAINING (Vision)")
    print("=" * 70)

    model.train()
    train_accs = []
    timing_start = time.time()

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
            scores = model.recognize(
                support_labels=s_lbl_t,
                support_kwargs={"images": support_img},
                query_kwargs={"images": query_img},
            )
            acc = (scores.argmax(-1) == q_lbl_t).float().mean().item()
            train_accs.append(acc)

        if ep % 100 == 0:
            model.prune()

        if ep in [10, 50, 100, 200] or ep % 100 == 0:
            elapsed = time.time() - timing_start
            window = min(ep, 50)
            recent = sum(train_accs[-window:]) / window
            hist = model.circuit_bank.activation_history.cpu()
            n_active = (hist > 0).sum().item()
            g = gini(hist[hist > 0]) if n_active > 1 else 0
            wiring = model.get_circuit_wiring_stats()
            print(f"[Ep {ep}] {elapsed:.0f}s | acc={recent:.2%} | circuits={n_active}/{config.n_circuits} | gini={g:.3f} | wiring={wiring['nonzero_connections']} conn")

    # ===== EVAL 1: Vision 2-way =====
    print("\n" + "=" * 70)
    print("EVAL 1: Vision-only 2-way")
    print("=" * 70)

    model.eval()
    n_way_eval = min(config.n_way, len(test_classes))
    vision_accs = []

    for _ in tqdm(range(config.eval_episodes), desc="Vision"):
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
            scores = model.recognize(support_labels=s_lbl_t,
                                     support_kwargs={"images": support_img},
                                     query_kwargs={"images": query_img})
            vision_accs.append((scores.argmax(-1) == q_lbl_t).float().mean().item())

    v_mean = np.mean(vision_accs)
    v_ci = 1.96 * np.std(vision_accs) / np.sqrt(len(vision_accs))
    print(f"  Vision: {v_mean:.2%} +/- {v_ci:.2%}")

    # ===== EVAL 2: Cross-modal image→text =====
    print("\n" + "=" * 70)
    print("EVAL 2: Cross-modal image→text")
    print("=" * 70)

    cross_text_accs = []
    for _ in tqdm(range(config.eval_episodes), desc="Image→Text"):
        selected = random.sample(test_classes, n_way_eval)
        support_imgs, support_labels = [], []
        for new_lbl, cls in enumerate(selected):
            idxs = random.sample(test_class_indices[cls], config.k_shot)
            for idx in idxs:
                support_imgs.append(test_dataset[idx][0])
                support_labels.append(new_lbl)

        support_img = torch.stack(support_imgs).to(device)
        s_lbl_t = torch.tensor(support_labels, device=device)

        query_texts, query_labels = [], []
        for new_lbl, cls in enumerate(selected):
            for prompt in CIFAR_TEXT_PROMPTS[cls]:
                query_texts.append(prompt)
                query_labels.append(new_lbl)

        query_tokens = model.text_encoder.tokenize(query_texts).to(device)
        q_lbl_t = torch.tensor(query_labels, device=device)

        model.reset()
        model.learn(images=support_img, labels=s_lbl_t)

        with torch.no_grad():
            scores = model.recognize(support_labels=s_lbl_t,
                                     support_kwargs={"images": support_img},
                                     query_kwargs={"text_tokens": query_tokens})
            cross_text_accs.append((scores.argmax(-1) == q_lbl_t).float().mean().item())

    ct_mean = np.mean(cross_text_accs)
    ct_ci = 1.96 * np.std(cross_text_accs) / np.sqrt(len(cross_text_accs))
    print(f"  Image→Text: {ct_mean:.2%} +/- {ct_ci:.2%}")

    # ===== EVAL 3: Cross-modal image→audio (simulated via CLIP text) =====
    print("\n" + "=" * 70)
    print("EVAL 3: Cross-modal image→audio (simulated)")
    print("=" * 70)

    cross_audio_accs = []
    for _ in tqdm(range(config.eval_episodes), desc="Image→Audio"):
        selected = random.sample(test_classes, n_way_eval)
        support_imgs, support_labels = [], []
        for new_lbl, cls in enumerate(selected):
            idxs = random.sample(test_class_indices[cls], config.k_shot)
            for idx in idxs:
                support_imgs.append(test_dataset[idx][0])
                support_labels.append(new_lbl)

        support_img = torch.stack(support_imgs).to(device)
        s_lbl_t = torch.tensor(support_labels, device=device)

        # Simulate audio features via CLIP text encoding of audio descriptions
        # This tests whether the circuit bank generalizes across modalities
        # even when the audio is encoded through the text path
        query_texts, query_labels = [], []
        for new_lbl, cls in enumerate(selected):
            for desc in CIFAR_AUDIO_DESCRIPTIONS[cls]:
                query_texts.append(desc)
                query_labels.append(new_lbl)

        query_tokens = model.text_encoder.tokenize(query_texts).to(device)
        q_lbl_t = torch.tensor(query_labels, device=device)

        model.reset()
        model.learn(images=support_img, labels=s_lbl_t)

        with torch.no_grad():
            scores = model.recognize(support_labels=s_lbl_t,
                                     support_kwargs={"images": support_img},
                                     query_kwargs={"text_tokens": query_tokens})
            cross_audio_accs.append((scores.argmax(-1) == q_lbl_t).float().mean().item())

    ca_mean = np.mean(cross_audio_accs)
    ca_ci = 1.96 * np.std(cross_audio_accs) / np.sqrt(len(cross_audio_accs))
    print(f"  Image→Audio: {ca_mean:.2%} +/- {ca_ci:.2%}")

    # ===== EVAL 4: Cross-modal image→signals =====
    print("\n" + "=" * 70)
    print("EVAL 4: Cross-modal image→signals")
    print("=" * 70)

    cross_signal_accs = []
    for _ in tqdm(range(config.eval_episodes), desc="Image→Signal"):
        selected = random.sample(test_classes, n_way_eval)
        support_imgs, support_labels = [], []
        for new_lbl, cls in enumerate(selected):
            idxs = random.sample(test_class_indices[cls], config.k_shot)
            for idx in idxs:
                support_imgs.append(test_dataset[idx][0])
                support_labels.append(new_lbl)

        support_img = torch.stack(support_imgs).to(device)
        s_lbl_t = torch.tensor(support_labels, device=device)

        # Signal queries
        query_signals, query_labels = [], []
        for new_lbl, cls in enumerate(selected):
            ctx = CIFAR_SIGNAL_CONTEXTS[cls]
            for _ in range(3):  # 3 signal samples per class
                sig = SignalEncoder.make_context_signal(**ctx)
                query_signals.append(sig.squeeze(0))
                query_labels.append(new_lbl)

        query_sig = torch.stack(query_signals).to(device)
        q_lbl_t = torch.tensor(query_labels, device=device)

        model.reset()
        model.learn(images=support_img, labels=s_lbl_t)

        with torch.no_grad():
            scores = model.recognize(support_labels=s_lbl_t,
                                     support_kwargs={"images": support_img},
                                     query_kwargs={"signals": query_sig})
            cross_signal_accs.append((scores.argmax(-1) == q_lbl_t).float().mean().item())

    cs_mean = np.mean(cross_signal_accs)
    cs_ci = 1.96 * np.std(cross_signal_accs) / np.sqrt(len(cross_signal_accs))
    print(f"  Image→Signal: {cs_mean:.2%} +/- {cs_ci:.2%}")

    # ===== EVAL 5: Circuit activation overlap across modalities =====
    print("\n" + "=" * 70)
    print("EVAL 5: Cross-modal circuit activation overlap")
    print("=" * 70)

    for cls in test_classes:
        model.reset()

        # Image
        img_idx = random.sample(test_class_indices[cls], min(5, len(test_class_indices[cls])))
        imgs = torch.stack([test_dataset[i][0] for i in img_idx]).to(device)
        img_circuits = model.get_activation_pattern(images=imgs)

        # Text
        texts = CIFAR_TEXT_PROMPTS[cls]
        tokens = model.text_encoder.tokenize(texts).to(device)
        txt_circuits = model.get_activation_pattern(text_tokens=tokens)

        # Audio (simulated)
        audio_texts = CIFAR_AUDIO_DESCRIPTIONS[cls]
        audio_tokens = model.text_encoder.tokenize(audio_texts).to(device)
        aud_circuits = model.get_activation_pattern(text_tokens=audio_tokens)

        # Signal
        ctx = CIFAR_SIGNAL_CONTEXTS[cls]
        sig = SignalEncoder.make_context_signal(**ctx).to(device)
        sig_circuits = model.get_activation_pattern(signals=sig)

        # Compute overlap: how many circuits are shared across modalities
        img_set = set(img_circuits.view(-1).tolist())
        txt_set = set(txt_circuits.view(-1).tolist())
        aud_set = set(aud_circuits.view(-1).tolist())
        sig_set = set(sig_circuits.view(-1).tolist())

        img_txt = len(img_set & txt_set) / max(1, len(img_set | txt_set))
        img_aud = len(img_set & aud_set) / max(1, len(img_set | aud_set))
        img_sig = len(img_set & sig_set) / max(1, len(img_set | sig_set))
        all_four = len(img_set & txt_set & aud_set & sig_set)
        any_four = len(img_set | txt_set | aud_set | sig_set)

        # Embedding similarities
        img_result = model.forward(images=imgs)
        txt_result = model.forward(text_tokens=tokens)
        sig_result = model.forward(signals=sig)

        img_emb = img_result["output"].mean(0)
        txt_emb = txt_result["output"].mean(0)
        sig_emb = sig_result["output"].mean(0)

        sim_it = F.cosine_similarity(img_emb.unsqueeze(0), txt_emb.unsqueeze(0)).item()
        sim_is = F.cosine_similarity(img_emb.unsqueeze(0), sig_emb.unsqueeze(0)).item()

        print(f"\n  {CIFAR_NAMES[cls]:12s}")
        print(f"    Circuits: img={sorted(img_set)[:8]}... txt={sorted(txt_set)[:8]}...")
        print(f"    Overlap:  img∩txt={img_txt:.0%}  img∩aud={img_aud:.0%}  img∩sig={img_sig:.0%}")
        print(f"    All 4 modalities share: {all_four}/{any_four} circuits ({all_four/max(1,any_four):.0%})")
        print(f"    Cosine sim: img↔txt={sim_it:.3f}  img↔sig={sim_is:.3f}")

    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print("PHASE 3 RESULTS")
    print("=" * 70)

    hist = model.circuit_bank.activation_history.cpu()
    n_active = (hist > 0).sum().item()
    g = gini(hist[hist > 0]) if n_active > 1 else 0
    wiring = model.get_circuit_wiring_stats()

    print(f"  Vision:       {v_mean:.2%} +/- {v_ci:.2%}")
    print(f"  Image→Text:   {ct_mean:.2%} +/- {ct_ci:.2%}")
    print(f"  Image→Audio:  {ca_mean:.2%} +/- {ca_ci:.2%}")
    print(f"  Image→Signal: {cs_mean:.2%} +/- {cs_ci:.2%}")
    print(f"  Random:       {1/n_way_eval:.2%}")
    print(f"  Circuits:     {n_active}/{config.n_circuits}")
    print(f"  Gini:         {g:.3f}")
    print(f"  Wiring:       {wiring['nonzero_connections']} connections")
    print(f"  Time:         {time.time()-timing_start:.0f}s")

    return model


if __name__ == "__main__":
    train_phase3()
