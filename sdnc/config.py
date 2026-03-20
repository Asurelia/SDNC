"""SDNC configuration."""

from dataclasses import dataclass, field


@dataclass
class SDNCConfig:
    """Central configuration for the SDNC model."""

    # Circuit bank
    n_circuits: int = 1000
    circuit_dim: int = 8
    circuit_output_dim: int = 32

    # Circuit representation space (the space where classification happens)
    circuit_repr_dim: int = 128

    # Input (CLIP ViT-B/32 output)
    input_dim: int = 512

    # Sparse routing
    sparsity_k: int = 3
    router_noise_std: float = 0.1
    router_bias_gamma: float = 0.01
    use_expert_choice: bool = True
    expert_capacity_factor: float = 1.25

    # Inter-circuit wiring (Objective 3)
    inter_circuit_hebbian_lr: float = 1e-3
    inter_circuit_decay: float = 0.999
    inter_circuit_prune_threshold: float = 1e-4

    # Hebbian learning
    hebbian_lr: float = 1e-4
    pruning_threshold: float = 1e-5

    # NCP wiring (scaled up for autonomous representation)
    ncp_inter_neurons: int = 48
    ncp_command_neurons: int = 32
    ncp_sensory_fanout: int = 6
    ncp_inter_fanout: int = 4
    ncp_recurrent_command_synapses: int = 4
    ncp_motor_fanin: int = 8

    # Temporal integrator
    integrator_state_dim: int = 64

    # Memory
    memory_capacity: int = 10000
    salience_threshold: float = 0.7
    working_memory_decay: float = 0.99

    # Vision + Text encoder (CLIP)
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "openai"

    # Audio encoder (Whisper tiny)
    whisper_model: str = "tiny"
    whisper_dim: int = 384

    # Signal encoder
    signal_input_dim: int = 8
    signal_hidden_dim: int = 64

    # Training
    n_way: int = 5
    k_shot: int = 1
    query_per_class: int = 15
    train_episodes: int = 1000
    eval_episodes: int = 600

    # Device
    device: str = "auto"

    @property
    def max_active_circuits(self) -> int:
        return max(1, self.sparsity_k)

    @property
    def sparsity_ratio(self) -> float:
        return self.max_active_circuits / self.n_circuits
