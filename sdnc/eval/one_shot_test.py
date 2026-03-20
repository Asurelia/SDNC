"""One-shot evaluation: test if the model recognizes patterns after 1-3 expositions.

Runs the standard N-way K-shot episodic evaluation protocol.
Reports accuracy with 95% confidence intervals over 600 episodes.
"""

import random
from collections import defaultdict

import numpy as np
import torch
from torchvision import datasets
from tqdm import tqdm

from sdnc.config import SDNCConfig
from sdnc.model import SDNCModel
from sdnc.utils.device import get_device


def evaluate_one_shot(
    model: SDNCModel,
    dataset,
    test_classes: list[int],
    config: SDNCConfig,
    n_episodes: int = 600,
    device: torch.device | None = None,
) -> dict:
    """Run N-way K-shot episodic evaluation.

    Args:
        model: Trained SDNCModel.
        dataset: Test dataset.
        test_classes: Classes to evaluate on (disjoint from train).
        config: Model config.
        n_episodes: Number of evaluation episodes.
        device: Device to use.

    Returns:
        Dict with accuracy, CI, sparsity stats.
    """
    device = device or get_device(config.device)
    model.to(device)
    model.eval()

    # Group indices by class
    class_indices: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if label in test_classes:
            class_indices[label].append(idx)

    n_way = min(config.n_way, len(test_classes))
    accuracies = []
    sparsity_ratios = []
    circuit_diversities = []

    for _ in tqdm(range(n_episodes), desc="One-shot evaluation"):
        selected_classes = random.sample(test_classes, n_way)

        support_images_list = []
        support_labels_list = []
        query_images_list = []
        query_labels_list = []

        for new_label, cls in enumerate(selected_classes):
            available = class_indices[cls]
            k = config.k_shot
            q = config.query_per_class

            chosen = random.sample(available, min(k + q, len(available)))
            support = chosen[:k]
            queries = chosen[k : k + q]

            for idx in support:
                img, _ = dataset[idx]
                support_images_list.append(img)
                support_labels_list.append(new_label)

            for idx in queries:
                img, _ = dataset[idx]
                query_images_list.append(img)
                query_labels_list.append(new_label)

        support_images = torch.stack(support_images_list).to(device)
        query_images = torch.stack(query_images_list).to(device)
        support_labels = torch.tensor(support_labels_list, device=device)
        query_labels = torch.tensor(query_labels_list, device=device)

        # Reset for new episode
        model.reset()

        # Learn from support (1-K shots)
        model.learn(support_images, support_labels)

        # Recognize queries
        with torch.no_grad():
            scores = model.recognize(support_images, query_images, support_labels)
            predictions = scores.argmax(dim=-1)
            accuracy = (predictions == query_labels).float().mean().item()
            accuracies.append(accuracy)

        # Track sparsity
        sparsity = config.sparsity_ratio
        sparsity_ratios.append(sparsity)

        # Track circuit diversity
        hist = model.circuit_bank.activation_history
        diversity = (hist > 0).float().mean().item()
        circuit_diversities.append(diversity)

    # Compute statistics
    acc_array = np.array(accuracies)
    mean_acc = acc_array.mean()
    std_acc = acc_array.std()
    ci_95 = 1.96 * std_acc / np.sqrt(n_episodes)

    results = {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "ci_95": ci_95,
        "ci_low": mean_acc - ci_95,
        "ci_high": mean_acc + ci_95,
        "n_episodes": n_episodes,
        "n_way": n_way,
        "k_shot": config.k_shot,
        "mean_sparsity": np.mean(sparsity_ratios),
        "mean_circuit_diversity": np.mean(circuit_diversities),
        "all_accuracies": accuracies,
    }

    return results


def print_results(results: dict):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print("SDNC One-Shot Evaluation Results")
    print("=" * 60)
    print(f"  Setup: {results['n_way']}-way {results['k_shot']}-shot")
    print(f"  Episodes: {results['n_episodes']}")
    print(f"  Accuracy: {results['mean_accuracy']:.2%} +/- {results['ci_95']:.2%}")
    print(f"  95% CI: [{results['ci_low']:.2%}, {results['ci_high']:.2%}]")
    print(f"  Sparsity: {results['mean_sparsity']:.2%} circuits active")
    print(f"  Circuit diversity: {results['mean_circuit_diversity']:.2%}")
    print()

    # Verify constraints
    sparsity_ok = results["mean_sparsity"] <= 0.05
    target_ok = results["mean_accuracy"] > 0.70

    print("Constraints:")
    print(f"  [{'PASS' if sparsity_ok else 'FAIL'}] Max 5% circuits active")
    print(f"  [{'PASS' if target_ok else 'FAIL'}] Accuracy > 70%")
    print("=" * 60)


def run_evaluation(config: SDNCConfig | None = None):
    """Run standalone evaluation."""
    config = config or SDNCConfig()
    device = get_device(config.device)

    model = SDNCModel(config)
    model.to(device)

    # Load test data
    preprocess = model.encoder.preprocess
    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=preprocess,
    )

    test_classes = [8, 9]  # ship, truck

    results = evaluate_one_shot(
        model=model,
        dataset=test_dataset,
        test_classes=test_classes,
        config=config,
        device=device,
    )

    print_results(results)
    return results


if __name__ == "__main__":
    run_evaluation()
