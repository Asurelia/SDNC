import sys
sys.path.insert(0, 'F:/sdnc')

from sdnc.config import SDNCConfig
from sdnc.train.phase1_vision import train_phase1

config = SDNCConfig(
    n_circuits=1000,
    sparsity_k=3,               # Fix 1: k=3 (was 1)
    n_way=5,
    k_shot=1,
    query_per_class=15,
    train_episodes=200,
    eval_episodes=100,
    hebbian_lr=1e-4,
    pruning_threshold=1e-5,
    router_bias_gamma=0.01,     # Fix 2: 10x stronger (was 0.001)
    router_noise_std=0.1,
    device='cuda',
)

print(f"=== V2 Config: k={config.sparsity_k}, gamma={config.router_bias_gamma}, Hebbian on CfC=ON ===")
print(f"Sparsity ratio: {config.sparsity_ratio:.2%}")

model, train_acc, eval_acc = train_phase1(config)
