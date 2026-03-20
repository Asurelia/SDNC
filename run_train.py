import sys
sys.path.insert(0, 'F:/sdnc')

from sdnc.config import SDNCConfig
from sdnc.train.phase1_vision import train_phase1

config = SDNCConfig(
    n_circuits=1000,
    sparsity_k=1,
    n_way=5,
    k_shot=1,
    query_per_class=15,
    train_episodes=200,
    eval_episodes=100,
    hebbian_lr=1e-4,
    pruning_threshold=1e-5,
    device='cuda',
)

model, train_acc, eval_acc = train_phase1(config)
