"""Configuration for the autonomous interaction system."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AutonomousConfig:
    """Runtime settings for interaction-driven SDNC learning."""

    input_dim: int = 256
    n_circuits: int = 256
    max_circuits: int | None = None
    max_active_ratio: float = 0.05
    activation_threshold: float = 0.58
    state_decay: float = 0.92
    state_influence: float = 0.15

    circuit_lr: float = 0.08
    connection_lr: float = 0.04
    connection_decay: float = 0.995
    memory_salience_threshold: float = 0.35
    confidence_threshold: float = 0.62

    memory_path: Path = Path("data/autonomous_memory.sqlite3")
    state_path: Path = Path("data/autonomous_circuits.npz")
    file_queue_path: Path = Path("data/file_queue")
    workspace_root: Path = Path(".")

    max_tool_calls: int = 3
    memory_top_k: int = 5
    tool_result_limit: int = 5
    web_timeout_s: float = 8.0
    allow_web: bool = True
    allow_file_tools: bool = True
    max_upload_mb: int = 64
    training_file_text_chars: int = 16000

    # Optional remote sync. Local SQLite remains the source of truth.
    sync_to_convex: bool = False
    convex_url: str | None = None
    convex_event_mutation: str = "sdnc:ingestEvent"

    # Self-improvement governance. Nothing architectural is adopted from one
    # source: candidates must survive a small consensus gate.
    auto_improve_enabled: bool = True
    improvement_interval: int = 20
    min_growth_evidence: int = 3
    min_consensus_sources: int = 2
    min_growth_salience: float = 0.55
    min_growth_novelty: float = 0.35
    growth_similarity_threshold: float = 0.62
    max_growth_per_cycle: int = 2
    connection_prune_threshold: float = 1e-5
    exploration_enabled: bool = True
    min_experiment_gain: float = 0.12
    existing_coverage_threshold: float = 0.92
    procedure_min_successes: int = 2
    procedure_min_success_rate: float = 0.65
    procedure_prune_failures: int = 3

    # Replay/sleep consolidation. Replay spends a small bounded plasticity
    # budget on already useful experiences; it must not become hidden training.
    replay_recent_limit: int = 200
    replay_batch_size: int = 12
    replay_min_salience: float = 0.45
    replay_strength: float = 0.35
    replay_max_drift: float = 0.08

    # Memory compaction keeps SQLite useful without erasing source evidence.
    # First compaction stores compact prototypes; raw episodes remain intact.
    memory_compaction_recent_limit: int = 500
    memory_compaction_batch_size: int = 80
    memory_compaction_min_group_size: int = 3
    memory_compaction_max_summary_chars: int = 900

    # Neuro-symbolic rules. Rules are local, provenance-backed hints for
    # routing and explanation; contradictory evidence weakens them.
    rule_recent_limit: int = 300
    rule_min_evidence: int = 2
    rule_min_confidence: float = 0.62
    rule_match_similarity: float = 0.58
    rule_counterexample_penalty: float = 0.14
    rule_success_boost: float = 0.08

    # Sensory prototypes compact repeated multimodal events into reusable
    # identity-like anchors without storing raw media in hot context.
    sensory_prototype_min_reliability: float = 0.45
    sensory_prototype_similarity: float = 0.72
    sensory_prototype_top_k: int = 5

    # Self-directed learning. External advisors are disabled unless explicitly
    # configured; sources provide hypotheses, never direct truth.
    lack_novelty_threshold: float = 0.55
    lack_min_severity: float = 0.35
    learning_min_sources: int = 2
    learning_min_consensus_score: float = 0.62
    learning_min_experiment_score: float = 0.55
    external_advisor_timeout_s: float = 20.0
    allow_external_advisors: bool = False
    external_advisor_commands: tuple[tuple[str, ...], ...] = ()

    # Cognitive budgets. These are SDNC's answer to dense, always-on
    # computation: spend more only when uncertainty/task shape requires it.
    default_cognitive_mode: str = "fast"
    fast_context_segments: int = 4
    think_context_segments: int = 12
    max_context_segments: int = 32
    fast_context_chars: int = 1200
    think_context_chars: int = 5000
    max_context_chars: int = 16000
    max_memory_top_k: int = 12
    max_tool_calls_max_mode: int = 8
    fast_hot_experts: int = 4
    think_hot_experts: int = 12
    max_hot_experts: int = 32

    # Resource budget estimates for local hardware. They are intentionally
    # conservative and describe hot residency, not total cold capacity.
    total_vram_gb: float = 16.0
    reserved_vram_gb: float = 2.0
    core_hot_vram_gb: float = 0.0
    core_hot_ram_gb: float = 0.25
    state_hot_vram_gb: float = 0.0
    state_hot_ram_gb: float = 0.10
    avg_hot_expert_vram_gb: float = 0.08
    avg_hot_expert_ram_gb: float = 0.02
    context_segment_ram_mb: float = 1.0
    cold_library_ram_gb: float = 0.5
    cold_library_disk_gb: float = 0.0
    context_lod_similarity: float = 0.82

    # Sparse cognitive core / global workspace. This is the bounded center that
    # organizes attention and proposals; it must not become a dense prompt dump.
    cognitive_workspace_slots: int = 16
    cognitive_attention_focus: int = 6

    # Expert lifecycle. Experts are living assets, not fixed weights.
    expert_active_utility: float = 0.62
    expert_min_utility: float = 0.30
    expert_retire_after_trials: int = 5
    expert_retire_failures: int = 3
    expert_duplicate_similarity: float = 0.96

    def __post_init__(self) -> None:
        if self.max_circuits is None:
            self.max_circuits = max(self.n_circuits, self.n_circuits * 2)
        if self.max_active_ratio > 0.05:
            raise ValueError("max_active_ratio must stay <= 0.05 for SDNC sparsity")
        if self.n_circuits > self.max_circuits:
            raise ValueError("n_circuits cannot exceed max_circuits")
        if self.default_cognitive_mode not in {"fast", "think", "max"}:
            raise ValueError("default_cognitive_mode must be fast, think, or max")

    @property
    def max_active_circuits(self) -> int:
        return max(1, int(self.n_circuits * self.max_active_ratio))

    @property
    def sparsity_ratio(self) -> float:
        return self.max_active_circuits / self.n_circuits
