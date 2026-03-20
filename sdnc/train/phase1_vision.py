"""Phase 1 training: few-shot vision via episodic Hebbian learning.

No global backpropagation — only local Hebbian weight updates.
Training is episodic: sample N-way K-shot episodes, learn, test.
"""

import random
import time
from collections import defaultdict, Counter

import torch
from torchvision import datasets
from tqdm import tqdm

from sdnc.config import SDNCConfig
from sdnc.model import SDNCModel
from sdnc.utils.device import get_device


def get_class_indices(dataset) -> dict[int, list[int]]:
    """Group dataset indices by class label."""
    class_indices: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_indices[label].append(idx)
    return class_indices


def sample_episode(
    class_indices: dict[int, list[int]],
    classes: list[int],
    n_way: int,
    k_shot: int,
    query_per_class: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Sample an N-way K-shot episode."""
    selected_classes = random.sample(classes, n_way)

    support_idx = []
    support_labels = []
    query_idx = []
    query_labels = []

    for new_label, cls in enumerate(selected_classes):
        available = class_indices[cls]
        chosen = random.sample(available, min(k_shot + query_per_class, len(available)))

        support = chosen[:k_shot]
        queries = chosen[k_shot : k_shot + query_per_class]

        support_idx.extend(support)
        support_labels.extend([new_label] * len(support))
        query_idx.extend(queries)
        query_labels.extend([new_label] * len(queries))

    return support_idx, support_labels, query_idx, query_labels


def load_dataset(config: SDNCConfig, encoder):
    """Load CIFAR-10 with CLIP preprocessing."""
    preprocess = encoder.preprocess

    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=preprocess,
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=preprocess,
    )

    return dataset, test_dataset


def print_activation_distribution(activation_history: torch.Tensor, episode: int, n_circuits: int):
    """Print detailed circuit activation distribution."""
    hist = activation_history.cpu()
    total_activations = hist.sum().item()
    n_ever_active = (hist > 0).sum().item()
    n_never_active = (hist == 0).sum().item()

    print(f"\n  --- Circuit Activation Distribution (episode {episode}) ---")
    print(f"  Total activations recorded: {int(total_activations)}")
    print(f"  Circuits ever active:  {n_ever_active}/{n_circuits} ({n_ever_active/n_circuits:.1%})")
    print(f"  Circuits NEVER active: {n_never_active}/{n_circuits} ({n_never_active/n_circuits:.1%})")

    if total_activations > 0:
        # Top 20 most activated circuits
        top_vals, top_ids = hist.topk(min(20, n_circuits))
        print(f"  Top 20 circuits by activation count:")
        for i, (cid, count) in enumerate(zip(top_ids.tolist(), top_vals.tolist())):
            bar = "█" * int(count / total_activations * 100)
            print(f"    Circuit {cid:4d}: {int(count):6d} ({count/total_activations:5.1%}) {bar}")

        # Distribution stats
        active_hist = hist[hist > 0]
        if len(active_hist) > 1:
            print(f"  Among active circuits:")
            print(f"    Mean activations: {active_hist.mean():.1f}")
            print(f"    Std activations:  {active_hist.std():.1f}")
            print(f"    Min activations:  {active_hist.min():.0f}")
            print(f"    Max activations:  {active_hist.max():.0f}")
            print(f"    Gini coefficient: {gini(active_hist):.3f} (0=uniform, 1=single circuit)")


def gini(values: torch.Tensor) -> float:
    """Compute Gini coefficient — measures inequality in distribution."""
    sorted_vals = values.sort().values.float()
    n = len(sorted_vals)
    if n == 0 or sorted_vals.sum() == 0:
        return 0.0
    index = torch.arange(1, n + 1, dtype=torch.float32)
    return (2 * (index * sorted_vals).sum() / (n * sorted_vals.sum()) - (n + 1) / n).item()


def train_phase1(config: SDNCConfig | None = None):
    """Run Phase 1 episodic training with full instrumentation."""
    config = config or SDNCConfig()
    device = get_device(config.device)

    print("=" * 70)
    print("SDNC Phase 1 Training — Sparse Dynamic Neural Circuits")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Circuits: {config.n_circuits}")
    print(f"Sparsity: k={config.sparsity_k}, ratio={config.sparsity_ratio:.2%}")
    print(f"N-way: {config.n_way}, K-shot: {config.k_shot}")
    print(f"Hebbian LR: {config.hebbian_lr}")
    print(f"Episodes: {config.train_episodes} train, {config.eval_episodes} eval")
    print(f"NCP wiring: inter={config.ncp_inter_neurons}, cmd={config.ncp_command_neurons}, motor={config.circuit_output_dim}")
    print()

    # Build model
    print("Loading model...")
    t0 = time.time()
    model = SDNCModel(config)
    model.to(device)
    print(f"  Model built in {time.time()-t0:.1f}s")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Load data
    print("\nLoading CIFAR-10...")
    t0 = time.time()
    train_dataset, test_dataset = load_dataset(config, model.encoder)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  Train: {len(train_dataset)} images, Test: {len(test_dataset)} images")

    # Split classes
    all_classes = list(range(10))
    cifar_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'horse', 'ship', 'frog', 'truck']
    train_classes = all_classes[:6]
    val_classes = all_classes[6:8]
    test_classes = all_classes[8:]

    print(f"  Train classes: {[cifar_names[c] for c in train_classes]}")
    print(f"  Val classes:   {[cifar_names[c] for c in val_classes]}")
    print(f"  Test classes:  {[cifar_names[c] for c in test_classes]}")

    train_class_indices = get_class_indices(train_dataset)
    test_class_indices = get_class_indices(test_dataset)

    # ===== TRAINING LOOP =====
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    model.train()
    train_accuracies = []
    episode_circuit_ids = []  # Track which circuits are used each episode
    checkpoint_milestones = [10, 50, 100, 200, 500, 1000]
    timing_start = time.time()

    for episode_idx in range(config.train_episodes):
        # Sample episode
        support_idx, support_labels, query_idx, query_labels = sample_episode(
            train_class_indices,
            train_classes,
            n_way=config.n_way,
            k_shot=config.k_shot,
            query_per_class=config.query_per_class,
        )

        # Collect images
        support_images = torch.stack([train_dataset[i][0] for i in support_idx]).to(device)
        query_images = torch.stack([train_dataset[i][0] for i in query_idx]).to(device)
        support_labels_t = torch.tensor(support_labels, device=device)
        query_labels_t = torch.tensor(query_labels, device=device)

        # Reset circuit states for new episode
        model.reset()

        # Learn from support (Hebbian)
        learn_result = model.learn(support_images, support_labels_t)

        # Track which circuits fired
        active_ids = learn_result["active_indices"].cpu().view(-1).tolist()
        episode_circuit_ids.extend(active_ids)

        # Recognize queries
        with torch.no_grad():
            scores = model.recognize(support_images, query_images, support_labels_t)
            predictions = scores.argmax(dim=-1)
            accuracy = (predictions == query_labels_t).float().mean().item()
            train_accuracies.append(accuracy)

        # Periodic pruning
        if (episode_idx + 1) % 100 == 0:
            model.prune()

        # ===== CHECKPOINT REPORTS =====
        ep = episode_idx + 1
        if ep in checkpoint_milestones or ep % 100 == 0:
            elapsed = time.time() - timing_start
            window = min(ep, 50)
            recent_acc = sum(train_accuracies[-window:]) / window
            overall_acc = sum(train_accuracies) / len(train_accuracies)

            print(f"\n{'='*70}")
            print(f"CHECKPOINT: Episode {ep}/{config.train_episodes} ({elapsed:.1f}s elapsed)")
            print(f"{'='*70}")
            print(f"  Accuracy (last {window} episodes): {recent_acc:.2%}")
            print(f"  Accuracy (overall):                {overall_acc:.2%}")
            print(f"  Random baseline ({config.n_way}-way):          {1/config.n_way:.2%}")

            # Unique circuits used in last N episodes
            recent_ids = episode_circuit_ids[-(window * config.sparsity_k * config.n_way * config.k_shot):]
            unique_recent = len(set(recent_ids))
            counter = Counter(recent_ids)
            print(f"  Unique circuits used (last {window} eps): {unique_recent}/{config.n_circuits}")

            # Full activation distribution
            print_activation_distribution(
                model.circuit_bank.activation_history, ep, config.n_circuits
            )

            # Router bias stats
            bias = model.router.expert_bias.cpu()
            print(f"\n  Router bias stats:")
            print(f"    Mean: {bias.mean():.6f}")
            print(f"    Std:  {bias.std():.6f}")
            print(f"    Min:  {bias.min():.6f}")
            print(f"    Max:  {bias.max():.6f}")

            # Weight stats of shared CfC
            cfc_params = list(model.circuit_bank.circuit.cfc.parameters())
            if cfc_params:
                all_weights = torch.cat([p.data.flatten() for p in cfc_params])
                print(f"\n  CfC weight stats:")
                print(f"    Mean: {all_weights.mean():.6f}")
                print(f"    Std:  {all_weights.std():.6f}")
                print(f"    Sparsity (|w|<1e-5): {(all_weights.abs() < 1e-5).float().mean():.1%}")

            # Episodic memory stats
            print(f"\n  Episodic memory: {len(model.episodic_memory)} stored")

    # ===== FINAL EVALUATION =====
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON HELD-OUT TEST CLASSES")
    print(f"Classes: {[cifar_names[c] for c in test_classes]}")
    print("=" * 70)

    model.eval()
    eval_accuracies = []
    n_way_eval = min(config.n_way, len(test_classes))

    for _ in tqdm(range(config.eval_episodes), desc="Evaluating"):
        support_idx, support_labels, query_idx, query_labels = sample_episode(
            test_class_indices,
            test_classes,
            n_way=n_way_eval,
            k_shot=config.k_shot,
            query_per_class=config.query_per_class,
        )

        support_images = torch.stack([test_dataset[i][0] for i in support_idx]).to(device)
        query_images = torch.stack([test_dataset[i][0] for i in query_idx]).to(device)
        support_labels_t = torch.tensor(support_labels, device=device)
        query_labels_t = torch.tensor(query_labels, device=device)

        model.reset()
        model.learn(support_images, support_labels_t)

        with torch.no_grad():
            scores = model.recognize(support_images, query_images, support_labels_t)
            predictions = scores.argmax(dim=-1)
            accuracy = (predictions == query_labels_t).float().mean().item()
            eval_accuracies.append(accuracy)

    import numpy as np
    acc_array = np.array(eval_accuracies)
    mean_acc = acc_array.mean()
    std_acc = acc_array.std()
    ci_95 = 1.96 * std_acc / np.sqrt(len(eval_accuracies))

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Setup: {n_way_eval}-way {config.k_shot}-shot, {config.eval_episodes} episodes")
    print(f"  Accuracy: {mean_acc:.2%} +/- {ci_95:.2%}")
    print(f"  95% CI: [{mean_acc - ci_95:.2%}, {mean_acc + ci_95:.2%}]")
    print(f"  Random baseline: {1/n_way_eval:.2%}")
    print(f"  Target: >70%")
    print(f"  Verdict: {'PASS' if mean_acc > 0.7 else 'FAIL'}")

    # Final activation distribution
    print_activation_distribution(
        model.circuit_bank.activation_history,
        config.train_episodes,
        config.n_circuits,
    )

    total_time = time.time() - timing_start
    print(f"\nTotal training time: {total_time:.1f}s")

    return model, train_accuracies, eval_accuracies


if __name__ == "__main__":
    train_phase1()
