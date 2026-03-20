"""Phase 2 hybrid: Token Choice (best perf) + Multimodal + Inter-circuit wiring."""
import sys
sys.path.insert(0, 'F:/sdnc')

from sdnc.config import SDNCConfig
from sdnc.train.phase2_multimodal import train_phase2

config = SDNCConfig(
    n_circuits=1000,
    use_expert_choice=False,        # Token Choice — preserves CfC state consistency
    sparsity_k=3,
    n_way=5,
    k_shot=1,
    query_per_class=15,
    train_episodes=200,
    eval_episodes=100,
    hebbian_lr=1e-4,
    pruning_threshold=1e-5,
    router_bias_gamma=0.01,
    router_noise_std=0.1,
    inter_circuit_hebbian_lr=1e-3,
    inter_circuit_decay=0.999,
    inter_circuit_prune_threshold=1e-4,
    device='cuda',
)

print(f"=== Phase 2 HYBRID: TokenChoice + Multimodal + InterCircuit Wiring ===")
model, train_acc, vision_acc, cross_acc = train_phase2(config)
