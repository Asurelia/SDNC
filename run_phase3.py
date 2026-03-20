import sys
sys.path.insert(0, 'F:/sdnc')

from sdnc.config import SDNCConfig
from sdnc.train.phase3_quadmodal import train_phase3

config = SDNCConfig(
    n_circuits=1000,
    use_expert_choice=False,
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
    device='cuda',
)

model = train_phase3(config)
