import sys
sys.path.insert(0, 'F:/sdnc')

from sdnc.config import SDNCConfig
from sdnc.train.phase2_multimodal import train_phase2

config = SDNCConfig(
    n_circuits=1000,
    use_expert_choice=True,
    expert_capacity_factor=1.25,
    sparsity_k=3,
    n_way=5,
    k_shot=1,
    query_per_class=15,
    train_episodes=200,
    eval_episodes=100,
    hebbian_lr=1e-4,
    pruning_threshold=1e-5,
    router_bias_gamma=0.01,
    inter_circuit_hebbian_lr=1e-3,
    inter_circuit_decay=0.999,
    inter_circuit_prune_threshold=1e-4,
    device='cuda',
)

model, train_acc, vision_acc, cross_acc = train_phase2(config)
