"""End-to-end interaction learning system."""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from time import time
from typing import Any

import numpy as np

from sdnc.agent.budget import BudgetManager
from sdnc.agent.compaction import MemoryCompactionCycle, MemoryCompactionReport
from sdnc.agent.config import AutonomousConfig
from sdnc.agent.context_lod import ContextLODCompressor
from sdnc.agent.cognitive_core import CognitiveCore, CognitiveWorkspace
from sdnc.agent.curriculum import (
    CurriculumReport,
    curriculum_manifest,
    run_guided_curriculum as run_guided_curriculum_cycle,
)
from sdnc.agent.dataset_ingestion import DatasetIngestor
from sdnc.agent.encoding import HashingExperienceEncoder
from sdnc.agent.expert_atlas import payload_summary
from sdnc.agent.experts import ExpertManager
from sdnc.agent.learning import LearningCycleReport, SelfDirectedLearner
from sdnc.agent.memory import ConversationExampleRecord, PersistentMemory, TrainingFileRecord
from sdnc.agent.multimodal import (
    AUDIO_EXTENSIONS,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    LocalMultimodalEncoder,
    ModalitySample,
)
from sdnc.agent.perception import PerceptionBus, SensoryEvent
from sdnc.agent.planner import ActionPlan, ActionPlanner
from sdnc.agent.plasticity import LocalCircuitLearner
from sdnc.agent.replay import SleepConsolidationCycle, SleepReport
from sdnc.agent.rules import RuleConsolidationReport, RuleEngine, RuleMatch
from sdnc.agent.sensory_prototypes import SensoryPrototypeLearner, SensoryPrototypeMatch
from sdnc.agent.self_improvement import ImprovementReport, SelfImprovementCycle
from sdnc.agent.sync import ConvexEventMirror, NullEventMirror, SafeEventMirror
from sdnc.agent.tools import (
    CalculatorTool,
    FileReadTool,
    FileSearchTool,
    MemoryRecallTool,
    ToolRegistry,
    WebSearchTool,
)
from sdnc.agent.types import Feedback, InteractionResult, MemoryRecord, ToolResult

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".jsonl",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".html",
    ".css",
    ".yaml",
    ".yml",
    ".xml",
    ".vtt",
    ".srt",
}
TRAINING_FILE_STATUSES = {"queued", "running", "done", "failed", "held"}


class InteractionLearningSystem:
    """A runnable SDNC agent that learns from interaction.

    The system is intentionally not a chatbot wrapper around a large model.
    Its central loop is local:

        observation -> sparse circuits -> memory/tools -> feedback -> local update
    """

    def __init__(
        self,
        config: AutonomousConfig | None = None,
        registry: ToolRegistry | None = None,
    ):
        self.config = config or AutonomousConfig()
        self.config.workspace_root = Path(self.config.workspace_root).resolve()
        self.encoder = HashingExperienceEncoder(self.config.input_dim)
        self.budget_manager = BudgetManager(self.config)
        self.context_compressor = ContextLODCompressor(
            self.encoder,
            similarity_threshold=self.config.context_lod_similarity,
        )
        self.cognitive_core = CognitiveCore(self.config)
        self.action_planner = ActionPlanner(self.config)
        self.multimodal_encoder = LocalMultimodalEncoder(self.config.input_dim)
        self.perception = PerceptionBus(self.multimodal_encoder, self.encoder)
        self.memory = PersistentMemory(self.config.memory_path, self.config.input_dim)
        self.sensory_prototypes = SensoryPrototypeLearner(self.config, self.memory)
        self.expert_manager = ExpertManager(self.config, self.memory)
        self.learner = LocalCircuitLearner(self.config)
        self.learner.load(self.config.state_path)
        self.registry = registry or self._default_registry()
        self.improver = SelfImprovementCycle(self.config, self.memory, self.learner)
        self.sleeper = SleepConsolidationCycle(self.config, self.memory, self.learner)
        self.compactor = MemoryCompactionCycle(self.config, self.memory)
        self.rule_engine = RuleEngine(self.config, self.memory)
        self.gap_learner = SelfDirectedLearner(
            self.config,
            self.memory,
            self.encoder,
            available_tools=self.registry.names(),
            expert_manager=self.expert_manager,
        )
        self.event_mirror = self._build_event_mirror()
        self._last_result: InteractionResult | None = None
        self._interactions_since_improvement = 0

    def close(self) -> None:
        self.learner.save(self.config.state_path)
        self.memory.close()

    def interact(
        self,
        text: str,
        context: dict[str, Any] | None = None,
        feedback: Feedback | None = None,
        learn: bool = True,
    ) -> InteractionResult:
        """Run one complete observe-act-learn cycle."""
        context = dict(context or {})
        budget = self.budget_manager.choose(text, explicit_mode=str(context.get("mode", "")))
        context_packet = self.context_compressor.compress(text, context, budget)
        resource_snapshot = self.budget_manager.snapshot(budget)
        encoding_text = context_packet.routing_text
        enriched_context = {
            **context,
            "__sdnc_budget": {
                "mode": budget.mode,
                "memory_top_k": budget.memory_top_k,
                "max_tool_calls": budget.max_tool_calls,
                "max_hot_experts": budget.max_hot_experts,
            },
            "__sdnc_context_lod": {
                "original_chars": context_packet.original_chars,
                "segment_count": len(context_packet.segments),
                "prototype_count": len(context_packet.prototypes),
                "compression_ratio": context_packet.compression_ratio,
                "estimated_tokens_saved": context_packet.estimated_tokens_saved,
            },
        }
        embedding = self.encoder.encode(encoding_text, enriched_context)
        expert_report = self.expert_manager.select_for_interaction(embedding, budget)
        activation = self.learner.activate(embedding)
        memories = self.memory.retrieve_similar(embedding, top_k=budget.memory_top_k)
        conversation_examples = self.memory.retrieve_conversation_examples(
            self.encoder.encode(text),
            top_k=max(5, budget.memory_top_k),
            min_confidence=0.2,
        )
        rule_matches = self.rule_engine.match(embedding, top_k=5)

        tool_names = self._choose_tools(
            text,
            embedding,
            activation_confidence=activation.confidence,
            max_tool_calls=budget.max_tool_calls,
            rule_matches=rule_matches,
        )
        workspace = self.cognitive_core.start(
            input_text=text,
            mode=budget.mode,
            activation=activation,
            memories=memories,
            tool_names=tool_names,
            experts=expert_report.selected_hot,
            context_summary=context_packet.global_summary,
        )
        action_plan = self.action_planner.plan(
            workspace,
            available_tools=tool_names,
            memory_count=len(memories),
            hot_expert_count=len(expert_report.selected_hot),
        )
        planned_tool_names = self._tool_names_for_plan(tool_names, action_plan, mode=budget.mode)
        planned_tool_names = self._ensure_required_tools(text, tool_names, planned_tool_names)
        tool_context = {
            **enriched_context,
            "embedding": embedding,
            "active_circuits": activation.indices,
            "memories": memories,
            "conversation_examples": conversation_examples,
            "context_packet": context_packet,
            "cognitive_budget": budget,
            "cognitive_workspace": workspace,
            "action_plan": action_plan,
            "active_experts": expert_report.selected_hot,
            "active_rules": rule_matches,
        }
        tool_results = self._run_tools(planned_tool_names, text, tool_context)

        salience = self._salience(activation.novelty, memories, tool_results, feedback)
        feedback_score = feedback.clipped_score() if feedback else None
        workspace = self.cognitive_core.complete(
            workspace,
            tool_results=tool_results,
            salience=salience,
            feedback_score=feedback_score,
        )
        self.expert_manager.record_outcome(
            expert_report.selected_hot,
            success=activation.confidence >= self.config.confidence_threshold
            or any(result.success for result in tool_results),
        )
        outcome = feedback.text if feedback else None
        episode_id = ""
        expert_rule_link_count = 0

        if learn:
            self.learner.learn(embedding, activation, salience, feedback_score or 0.0)
            if salience >= self.config.memory_salience_threshold or feedback is not None:
                memory_context = {
                    **enriched_context,
                    "__sdnc": {
                        "confidence": activation.confidence,
                        "novelty": activation.novelty,
                        "tool_names": planned_tool_names,
                        "proposed_tool_names": tool_names,
                        "successful_tools": [
                            result.tool_name for result in tool_results if result.success
                        ],
                        "cognitive_trace_id": workspace.id,
                        "surprise": workspace.surprise,
                        "uncertainty": workspace.uncertainty,
                        "action_plan": action_plan.to_payload(),
                    },
                }
                episode_id = self.memory.store_episode(
                    text=text,
                    context=memory_context,
                    embedding=embedding,
                    active_circuits=activation.indices,
                    salience=salience,
                    outcome=outcome,
                    feedback_score=feedback_score,
                )
            self._persist_cognitive_trace(workspace, episode_id, text, action_plan=action_plan)
            self._learn_tool_preferences(text, embedding, tool_results, feedback)
            if episode_id:
                self.rule_engine.record_rule_outcomes(
                    rule_matches,
                    successful_tools={result.tool_name for result in tool_results if result.success},
                    episode_id=episode_id,
                )
            expert_rule_link_count = self._attach_rules_to_experts(
                rule_matches,
                expert_report.selected_hot,
                episode_id=episode_id,
                workspace_id=workspace.id,
            )
            self._interactions_since_improvement += 1
            improvement_report = self._maybe_self_improve()
            self.learner.save(self.config.state_path)
        else:
            improvement_report = None

        response, conversation_decision = self._synthesize_response(
            text,
            activation,
            memories,
            conversation_examples,
            tool_results,
            salience,
        )
        result = InteractionResult(
            episode_id=episode_id,
            timestamp=time(),
            input_text=text,
            response=response,
            activation=activation,
            memories=memories,
            tool_results=tool_results,
            learned=learn,
            metadata={
                "salience": salience,
                "tool_names": planned_tool_names,
                "proposed_tool_names": tool_names,
                "improvement_report": improvement_report,
                "n_circuits": self.config.n_circuits,
                "cognitive_budget": _budget_payload(budget),
                "context_lod": _context_payload(context_packet),
                "resource_budget": _resource_payload(resource_snapshot),
                "expert_lifecycle": _expert_report_payload(expert_report),
                "conversation_examples": _conversation_examples_payload(conversation_examples),
                "conversation_decision": conversation_decision,
                "neuro_symbolic_rules": _rule_matches_payload(rule_matches),
                "rule_attachments": {"expert_links": expert_rule_link_count},
                "cognitive_core": _cognitive_workspace_payload(workspace),
                "action_plan": action_plan.to_payload(),
                "introspection": _introspection_payload(
                    mode=budget.mode,
                    activation=activation,
                    memories=memories,
                    proposed_tools=tool_names,
                    executed_tools=planned_tool_names,
                    tool_results=tool_results,
                    expert_report=expert_report,
                    rule_matches=rule_matches,
                    workspace=workspace,
                    action_plan=action_plan,
                ),
            },
        )
        self._last_result = result
        self._emit_event(
            "interaction",
            {
                "episode_id": episode_id,
                "text": text,
                "active_count": activation.active_count,
                "active_circuits": activation.indices,
                "confidence": activation.confidence,
                "novelty": activation.novelty,
                "salience": salience,
                "tools": planned_tool_names,
                "proposed_tools": tool_names,
                "mode": budget.mode,
                "context_compression_ratio": context_packet.compression_ratio,
                "hot_vram_gb": resource_snapshot.hot_vram_gb,
                "hot_experts": [expert.name for expert in expert_report.selected_hot],
                "rules": [match.rule.name for match in rule_matches],
                "expert_rule_links": expert_rule_link_count,
                "cognitive_trace_id": workspace.id,
                "uncertainty": workspace.uncertainty,
                "surprise": workspace.surprise,
                "attention_focus": list(workspace.attention_focus),
                "planner_action": action_plan.selected.action,
                "planner_score": action_plan.selected.score,
                "introspection": result.metadata["introspection"],
            },
        )
        return result

    def teach_response(
        self,
        prompt: str,
        response: str,
        *,
        source: str = "user_teach",
        context: dict[str, Any] | None = None,
    ) -> InteractionResult:
        """Teach SDNC one source-backed conversation response.

        This is an explicit user teaching path, not hidden distillation. It
        stores the example, gives it positive local feedback, and exposes the
        learned example id so later interactions and corrections remain
        inspectable.
        """
        prompt = " ".join(str(prompt or "").split())
        response = " ".join(str(response or "").split())
        if not prompt:
            raise ValueError("prompt is required")
        if not response:
            raise ValueError("response is required")

        teach_context = {
            **dict(context or {}),
            "kind": "conversation_teach",
            "source": source,
        }
        embedding = self.encoder.encode(prompt)
        activation = self.learner.activate(embedding)
        memories = self.memory.retrieve_similar(embedding, top_k=self.config.memory_top_k)
        example_id = self.memory.upsert_conversation_example(
            source=source,
            prompt=prompt,
            response=response,
            embedding=embedding,
            confidence=0.96,
            payload={
                "kind": "direct_teach",
                "source": source,
                "taught_at": time(),
            },
        )
        self.memory.record_conversation_feedback(example_id, success=True)
        self.memory.record_source_feedback(source, accepted=True)
        salience = 1.0
        self.learner.learn(embedding, activation, salience, 1.0)
        episode_id = self.memory.store_episode(
            text=f"TEACH conversation: {prompt}",
            context={
                **teach_context,
                "conversation_example_id": example_id,
                "target_response": response,
            },
            embedding=embedding,
            active_circuits=activation.indices,
            salience=salience,
            outcome=response,
            feedback_score=1.0,
        )
        self.learner.save(self.config.state_path)

        result = InteractionResult(
            episode_id=episode_id,
            timestamp=time(),
            input_text=prompt,
            response=f"Appris. Quand tu dis : {prompt} | je répondrai : {response}",
            activation=activation,
            memories=memories,
            tool_results=[],
            learned=True,
            metadata={
                "salience": salience,
                "tool_names": [],
                "proposed_tool_names": [],
                "conversation_teach": {
                    "example_id": example_id,
                    "source": source,
                    "prompt": prompt[:360],
                    "response": response[:500],
                    "confidence": 0.96,
                },
                "conversation_decision": {
                    "intent": _response_intent(prompt),
                    "accepted": True,
                    "selected_id": example_id,
                    "selected_similarity": 1.0,
                    "selected_source": source,
                    "selected_prompt": prompt[:240],
                    "rejected_count": 0,
                    "best_rejected_id": "",
                    "best_rejected_similarity": 0.0,
                    "best_rejected_prompt": "",
                    "reason": "direct_teach",
                },
                "introspection": {
                    "mode": "teach",
                    "open_laboratory": False,
                    "circuit_trace": [
                        {
                            "index": idx,
                            "weight": weight,
                            "score": score,
                        }
                        for idx, weight, score in zip(
                            activation.indices,
                            activation.weights,
                            activation.scores,
                        )
                    ],
                    "tool_trace": [],
                    "memory_trace": [
                        {
                            "id": memory.id,
                            "similarity": round(memory.similarity, 4),
                            "salience": round(memory.salience, 4),
                            "text": memory.text[:240],
                        }
                        for memory in memories
                    ],
                    "conversation_trace": {
                        "example_id": example_id,
                        "source": source,
                        "prompt": prompt[:240],
                        "response": response[:360],
                    },
                },
            },
        )
        self._last_result = result
        self._emit_event(
            "teaching",
            {
                "episode_id": episode_id,
                "example_id": example_id,
                "source": source,
                "prompt": prompt,
                "response": response,
                "active_count": activation.active_count,
                "active_circuits": activation.indices,
                "confidence": activation.confidence,
                "novelty": activation.novelty,
                "salience": salience,
            },
        )
        return result

    def learn_sample(
        self,
        sample: ModalitySample,
        context: dict[str, Any] | None = None,
        learn: bool = True,
        use_tools: bool = False,
    ) -> InteractionResult:
        """Learn from one text/image/audio/video observation.

        Dataset ingestion calls this method with `use_tools=False` so SDNC can
        absorb large corpora without running web/file tools on every row.
        """
        return self.observe([sample], context=context, learn=learn, use_tools=use_tools)

    def observe(
        self,
        samples: list[ModalitySample],
        context: dict[str, Any] | None = None,
        learn: bool = True,
        use_tools: bool = False,
    ) -> InteractionResult:
        """Bind one or more sensory samples and learn the resulting event.

        This is the SDNC multimodal path: sensors become compact signals,
        signals become a bound event, and only that event enters sparse
        circuits/memory/experts.
        """
        base_context = dict(context or {})
        event = self.perception.observe(samples, context=base_context)
        prototype_matches = self.sensory_prototypes.match(event)
        budget = self.budget_manager.choose(event.summary, explicit_mode=str(base_context.get("mode", "")))
        resource_snapshot = self.budget_manager.snapshot(budget)
        expert_report = self.expert_manager.select_for_interaction(event.embedding, budget)
        observation_context = {
            **event.context,
            "__sdnc_observation": {
                "event_id": event.id,
                "modalities": event.modalities,
                "source": event.source,
                "binding_score": event.binding_score,
                "reliability": event.reliability,
                "signal_count": len(event.signals),
                "prototype_matches": [match.to_payload() for match in prototype_matches],
            },
        }
        activation = self.learner.activate(event.embedding)
        memories = self.memory.retrieve_similar(event.embedding, top_k=budget.memory_top_k)
        sensory_memories = self.memory.retrieve_sensory_bindings(event.embedding, top_k=min(5, budget.memory_top_k))
        rule_matches = self.rule_engine.match(event.embedding, top_k=5)
        tool_names = (
            self._choose_tools(
                event.summary,
                event.embedding,
                activation_confidence=activation.confidence,
                max_tool_calls=budget.max_tool_calls,
                rule_matches=rule_matches,
            )
            if use_tools
            else []
        )
        workspace = self.cognitive_core.start(
            input_text=event.summary,
            mode=budget.mode,
            activation=activation,
            memories=memories,
            tool_names=tool_names,
            experts=expert_report.selected_hot,
            context_summary=f"{event.source} {'+'.join(event.modalities)}",
        )
        action_plan = self.action_planner.plan(
            workspace,
            available_tools=tool_names,
            memory_count=len(memories),
            hot_expert_count=len(expert_report.selected_hot),
        )
        planned_tool_names = self._tool_names_for_plan(tool_names, action_plan, mode=budget.mode)
        tool_context = {
            **observation_context,
            "embedding": event.embedding,
            "active_circuits": activation.indices,
            "memories": memories,
            "sensory_memories": sensory_memories,
            "cognitive_workspace": workspace,
            "action_plan": action_plan,
            "active_experts": expert_report.selected_hot,
            "active_rules": rule_matches,
        }
        tool_results = self._run_tools(planned_tool_names, event.summary, tool_context)
        salience = float(
            np.clip(
                0.65 * self._salience(activation.novelty, memories, tool_results, feedback=None)
                + 0.35 * event.reliability,
                0.0,
                1.0,
            )
        )
        workspace = self.cognitive_core.complete(
            workspace,
            tool_results=tool_results,
            salience=salience,
        )
        self.expert_manager.record_outcome(
            expert_report.selected_hot,
            success=salience >= self.config.memory_salience_threshold,
        )
        episode_id = ""
        expert_rule_link_count = 0
        prototype_rule_link_count = 0

        if learn:
            self.learner.learn(event.embedding, activation, salience, 0.0)
            if salience >= self.config.memory_salience_threshold:
                episode_id = self.memory.store_episode(
                    text=event.summary,
                    context=observation_context,
                    embedding=event.embedding,
                    active_circuits=activation.indices,
                    salience=salience,
                )
            self._persist_cognitive_trace(workspace, episode_id, event.summary, action_plan=action_plan)
            if episode_id and use_tools:
                self.rule_engine.record_rule_outcomes(
                    rule_matches,
                    successful_tools={result.tool_name for result in tool_results if result.success},
                    episode_id=episode_id,
                )
            learned_prototype = self.sensory_prototypes.learn(event, episode_id, prototype_matches)
            expert_rule_link_count = self._attach_rules_to_experts(
                rule_matches,
                expert_report.selected_hot,
                episode_id=episode_id,
                workspace_id=workspace.id,
            )
            prototype_rule_link_count = self._attach_rules_to_sensory_prototype(
                rule_matches,
                learned_prototype,
                episode_id=episode_id,
                event_id=event.id,
                workspace_id=workspace.id,
            )
            self.memory.store_sensory_binding(
                event_id=event.id,
                episode_id=episode_id,
                source=event.source,
                modalities=event.modalities,
                sample_ids=event.sample_ids,
                summary=event.summary,
                embedding=event.embedding,
                binding_score=event.binding_score,
                salience=salience,
                features=_sensory_features_payload(event),
                context=observation_context,
            )
            self._interactions_since_improvement += 1
            improvement_report = self._maybe_self_improve()
            self.learner.save(self.config.state_path)
        else:
            improvement_report = None
            learned_prototype = None

        result = InteractionResult(
            episode_id=episode_id,
            timestamp=time(),
            input_text=event.summary,
            response=self._synthesize_observation_response(event, activation, memories, salience),
            activation=activation,
            memories=memories,
            tool_results=tool_results,
            learned=learn,
            metadata={
                "salience": salience,
                "tool_names": planned_tool_names,
                "proposed_tool_names": tool_names,
                "improvement_report": improvement_report,
                "n_circuits": self.config.n_circuits,
                "modality": event.modalities[0] if len(event.modalities) == 1 else "+".join(event.modalities),
                "source": event.source,
                "sample_id": event.sample_ids[0] if event.sample_ids else None,
                "features": event.signals[0].features if len(event.signals) == 1 else _sensory_features_payload(event),
                "sensory_event": _sensory_event_payload(event, sensory_memories),
                "sensory_prototypes": _sensory_prototypes_payload(prototype_matches, learned_prototype),
                "cognitive_budget": _budget_payload(budget),
                "resource_budget": _resource_payload(resource_snapshot),
                "expert_lifecycle": _expert_report_payload(expert_report),
                "neuro_symbolic_rules": _rule_matches_payload(rule_matches),
                "rule_attachments": {
                    "expert_links": expert_rule_link_count,
                    "sensory_prototype_links": prototype_rule_link_count,
                },
                "cognitive_core": _cognitive_workspace_payload(workspace),
                "action_plan": action_plan.to_payload(),
                "introspection": _introspection_payload(
                    mode=budget.mode,
                    activation=activation,
                    memories=memories,
                    proposed_tools=tool_names,
                    executed_tools=planned_tool_names,
                    tool_results=tool_results,
                    expert_report=expert_report,
                    rule_matches=rule_matches,
                    workspace=workspace,
                    action_plan=action_plan,
                ),
            },
        )
        self._last_result = result
        self._emit_event(
            "observation",
            {
                "episode_id": episode_id,
                "event_id": event.id,
                "modality": event.modalities[0] if len(event.modalities) == 1 else "+".join(event.modalities),
                "modalities": event.modalities,
                "source": event.source,
                "sample_ids": event.sample_ids,
                "labels": [signal.label for signal in event.signals if signal.label],
                "binding_score": event.binding_score,
                "reliability": event.reliability,
                "signal_count": len(event.signals),
                "active_count": activation.active_count,
                "active_circuits": activation.indices,
                "confidence": activation.confidence,
                "novelty": activation.novelty,
                "salience": salience,
                "mode": budget.mode,
                "tools": planned_tool_names,
                "proposed_tools": tool_names,
                "hot_vram_gb": resource_snapshot.hot_vram_gb,
                "hot_experts": [expert.name for expert in expert_report.selected_hot],
                "rules": [match.rule.name for match in rule_matches],
                "expert_rule_links": expert_rule_link_count,
                "sensory_prototype_rule_links": prototype_rule_link_count,
                "sensory_prototypes": [match.prototype.key for match in prototype_matches],
                "learned_sensory_prototype": learned_prototype.key if learned_prototype else "",
                "cognitive_trace_id": workspace.id,
                "uncertainty": workspace.uncertainty,
                "surprise": workspace.surprise,
                "attention_focus": list(workspace.attention_focus),
                "planner_action": action_plan.selected.action,
                "planner_score": action_plan.selected.score,
                "introspection": result.metadata["introspection"],
            },
        )
        return result

    def give_feedback(self, score: float, text: str = "") -> InteractionResult:
        """Apply feedback to the last interaction and reinforce locally."""
        if self._last_result is None:
            return InteractionResult.empty("", "No previous interaction to reinforce.")

        feedback = Feedback(score=score, text=text)
        embedding = self.encoder.encode(
            self._last_result.input_text,
            {"feedback": text, "previous_episode": self._last_result.episode_id},
        )
        salience = max(self._last_result.metadata.get("salience", 0.0), abs(feedback.clipped_score()))
        self.learner.learn(embedding, self._last_result.activation, salience, feedback.clipped_score())
        if self._last_result.episode_id:
            self.memory.update_episode_feedback(
                self._last_result.episode_id,
                feedback.clipped_score(),
                text,
            )
        self._learn_tool_preferences(
            self._last_result.input_text,
            embedding,
            self._last_result.tool_results,
            feedback,
        )
        decision = self._last_result.metadata.get("conversation_decision") or {}
        selected_example_id = str(decision.get("selected_id") or "")
        if selected_example_id:
            self.memory.record_conversation_feedback(
                selected_example_id,
                success=feedback.clipped_score() >= 0.0,
            )
        self.learner.save(self.config.state_path)
        self._last_result.response = "Feedback integrated into local circuits and procedural memory."
        self._last_result.metadata["feedback_score"] = feedback.clipped_score()
        self._last_result.metadata["conversation_feedback"] = {
            "example_id": selected_example_id,
            "updated": bool(selected_example_id),
            "success": feedback.clipped_score() >= 0.0,
        }
        self._emit_event(
            "feedback",
            {
                "episode_id": self._last_result.episode_id,
                "score": feedback.clipped_score(),
                "text": text,
                "conversation_example_id": selected_example_id,
            },
        )
        return self._last_result

    def recent_memories(self, limit: int = 10) -> list[MemoryRecord]:
        return self.memory.recent(limit)

    def recent_sensory_bindings(self, limit: int = 10):
        return self.memory.recent_sensory_bindings(limit)

    def recent_sensory_prototypes(self, limit: int = 10):
        return self.memory.recent_sensory_prototypes(limit)

    def recent_cognitive_traces(self, limit: int = 10):
        return self.memory.recent_cognitive_traces(limit)

    def recent_memory_compactions(self, limit: int = 10):
        return self.memory.recent_memory_compactions(limit)

    def queue_training_file(
        self,
        name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        status: str = "queued",
        payload: dict[str, Any] | None = None,
    ) -> TrainingFileRecord:
        """Store a local file for later SDNC processing."""
        if status not in TRAINING_FILE_STATUSES:
            raise ValueError(f"unsupported training file status: {status}")
        if len(data) > self.config.max_upload_mb * 1024 * 1024:
            raise ValueError(f"file is larger than {self.config.max_upload_mb} MB")
        safe_name = _safe_filename(name)
        digest = hashlib.sha256(data).hexdigest()
        queue_dir = Path(self.config.file_queue_path).resolve()
        queue_dir.mkdir(parents=True, exist_ok=True)
        modality = _modality_for_file(safe_name, content_type)
        preview = _file_preview(data, safe_name, content_type, limit=1200)
        file_id = str(uuid.uuid4())
        target_dir = queue_dir / file_id
        target_dir.mkdir(parents=True, exist_ok=True)
        path = str((target_dir / safe_name).resolve())
        Path(path).write_bytes(data)
        file_id = self.memory.add_training_file(
            name=safe_name,
            path=path,
            content_type=content_type,
            size_bytes=len(data),
            sha256=digest,
            status=status,
            modality=modality,
            preview=preview,
            payload=payload or {},
            file_id=file_id,
        )
        record = self.memory.get_training_file(file_id)
        self._emit_event(
            "file",
            {
                "action": "queued",
                "file_id": file_id,
                "name": safe_name,
                "status": status,
                "modality": modality,
                "size_bytes": len(data),
            },
        )
        return record

    def list_training_files(self, status: str | None = None, limit: int = 200) -> list[TrainingFileRecord]:
        return self.memory.list_training_files(status=status, limit=limit)

    def training_file_summary(self) -> dict[str, int]:
        return self.memory.training_file_summary()

    def set_training_file_status(self, file_id: str, status: str) -> TrainingFileRecord:
        if status not in TRAINING_FILE_STATUSES:
            raise ValueError(f"unsupported training file status: {status}")
        self.memory.update_training_file(file_id, status=status, error="")
        record = self.memory.get_training_file(file_id)
        self._emit_event(
            "file",
            {"action": "status", "file_id": file_id, "name": record.name, "status": status},
        )
        return record

    def process_training_file(
        self,
        file_id: str,
        mode: str = "fast",
        learn: bool = True,
        use_tools: bool = False,
    ) -> InteractionResult:
        record = self.memory.get_training_file(file_id)
        if record is None:
            raise KeyError(file_id)
        self.memory.update_training_file(file_id, status="running", error="")
        self._emit_event(
            "file",
            {"action": "started", "file_id": file_id, "name": record.name, "status": "running"},
        )
        try:
            if record.modality == "dataset":
                return self._process_dataset_file(record, mode=mode, learn=learn)
            sample = self._sample_from_training_file(record)
            result = self.observe(
                [sample],
                context={"mode": mode, "training_file_id": file_id, "ui": "web"},
                learn=learn,
                use_tools=use_tools,
            )
            self.memory.update_training_file(
                file_id,
                status="done",
                processed_episode_id=result.episode_id,
                error="",
                payload={
                    "last_confidence": result.activation.confidence,
                    "last_salience": result.metadata.get("salience", 0.0),
                    "last_mode": mode,
                },
            )
            self._emit_event(
                "file",
                {
                    "action": "done",
                    "file_id": file_id,
                    "name": record.name,
                    "status": "done",
                    "episode_id": result.episode_id,
                    "confidence": result.activation.confidence,
                    "salience": result.metadata.get("salience", 0.0),
                },
            )
            return result
        except Exception as exc:
            self.memory.update_training_file(file_id, status="failed", error=f"{type(exc).__name__}: {exc}")
            self._emit_event(
                "file",
                {
                    "action": "failed",
                    "file_id": file_id,
                    "name": record.name,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise

    def _process_dataset_file(
        self,
        record: TrainingFileRecord,
        mode: str = "fast",
        learn: bool = True,
    ) -> InteractionResult:
        records, metadata = _dataset_records_from_path(Path(record.path), row_limit=self.config.training_dataset_row_limit)
        report = DatasetIngestor(self).ingest_records(
            records,
            source=f"file:{record.name}",
            limit=metadata.get("row_limit"),
            text_columns=metadata.get("text_columns") or None,
            label_columns=metadata.get("label_columns") or None,
            build_pack=True,
            learn=learn,
        )
        result = report.last_result or InteractionResult.empty(record.name, "Dataset ingestion completed without observable rows.")
        payload = {
            "last_confidence": result.activation.confidence,
            "last_salience": result.metadata.get("salience", 0.0),
            "last_mode": mode,
            "dataset": {
                **metadata,
                **report.to_payload(),
            },
        }
        self.memory.update_training_file(
            record.id,
            status="done",
            processed_episode_id=result.episode_id,
            error="; ".join(report.errors[:3]),
            payload=payload,
        )
        result.metadata["dataset_ingestion"] = payload["dataset"]
        result.response = (
            "Dataset ingestion complete.\n"
            f"rows={report.records_seen} samples={report.samples_seen} episodes={report.episodes_stored} "
            f"errors={len(report.errors)}\n"
            + result.response
        )
        self._emit_event(
            "file",
            {
                "action": "dataset_done",
                "file_id": record.id,
                "name": record.name,
                "status": "done",
                "episode_id": result.episode_id,
                "records_seen": report.records_seen,
                "samples_seen": report.samples_seen,
                "episodes_stored": report.episodes_stored,
                "errors": report.errors[:5],
            },
        )
        return result

    def process_next_training_file(
        self,
        mode: str = "fast",
        learn: bool = True,
        use_tools: bool = False,
    ) -> InteractionResult | None:
        queued = self.memory.list_training_files(status="queued", limit=1)
        if not queued:
            return None
        return self.process_training_file(queued[0].id, mode=mode, learn=learn, use_tools=use_tools)

    def _sample_from_training_file(self, record: TrainingFileRecord) -> ModalitySample:
        path = Path(record.path)
        metadata = {
            "file_id": record.id,
            "file_name": record.name,
            "content_type": record.content_type,
            "size_bytes": record.size_bytes,
            "sha256": record.sha256,
            "queue_status": record.status,
        }
        source = f"file:{record.name}"
        if record.modality == "text":
            text = _read_text_payload(path, self.config.training_file_text_chars)
            return ModalitySample(
                modality="text",
                content=text,
                text=text,
                label=record.payload.get("label"),
                source=source,
                sample_id=record.id,
                metadata=metadata,
            )
        if record.modality in {"image", "audio", "video"}:
            return ModalitySample(
                modality=record.modality,
                content=str(path),
                text=record.preview or record.name,
                label=record.payload.get("label"),
                source=source,
                sample_id=record.id,
                metadata=metadata,
            )
        return ModalitySample(
            modality="text",
            content=record.preview or f"{record.name} {record.content_type} {record.sha256}",
            text=record.preview or f"{record.name} {record.content_type} {record.sha256}",
            label=record.payload.get("label"),
            source=source,
            sample_id=record.id,
            metadata=metadata,
        )

    def run_self_improvement(self) -> ImprovementReport:
        """Run a bounded maintenance cycle immediately."""
        report = self.improver.run()
        self._interactions_since_improvement = 0
        self.learner.save(self.config.state_path)
        self._emit_event(
            "improvement",
            {
                "accepted_count": report.accepted_count,
                "summary": report.summary(),
                "actions": [
                    {
                        "kind": action.kind,
                        "accepted": action.accepted,
                        "reason": action.reason,
                    }
                    for action in report.actions
                ],
                "n_circuits": self.config.n_circuits,
            },
        )
        return report

    def run_sleep_cycle(
        self,
        preview: bool = False,
        batch_size: int | None = None,
    ) -> SleepReport:
        """Run or preview a bounded replay/sleep consolidation cycle."""
        report = self.sleeper.run(preview=preview, batch_size=batch_size)
        if not preview:
            self.learner.save(self.config.state_path)
        self._emit_event(
            "sleep",
            {
                **report.to_payload(),
                "n_circuits": self.config.n_circuits,
            },
        )
        return report

    def run_memory_compaction(
        self,
        preview: bool = False,
        batch_size: int | None = None,
    ) -> MemoryCompactionReport:
        """Compact repeated memory patterns without deleting source evidence."""
        report = self.compactor.run(preview=preview, batch_size=batch_size)
        self._emit_event(
            "compaction",
            {
                **report.to_payload(),
                "memory_compaction_summary": self.memory_compaction_summary(),
            },
        )
        return report

    def run_rule_consolidation(self) -> RuleConsolidationReport:
        """Extract provenance-backed rules from recent verified traces."""
        report = self.rule_engine.consolidate_recent()
        self._emit_event(
            "rules",
            {
                **report.to_payload(),
                "rule_summary": self.rule_summary(),
                "rule_conflict_summary": self.rule_conflict_summary(),
            },
        )
        return report

    def curriculum_manifest(self) -> dict[str, Any]:
        """Return beginner-safe guided training steps."""
        return curriculum_manifest()

    def run_guided_curriculum(
        self,
        step_id: str | None = None,
        mode: str = "think",
        sleep_preview: bool = True,
        batch_size: int = 6,
    ) -> CurriculumReport:
        """Run one guided training step or the full local curriculum."""
        report = run_guided_curriculum_cycle(
            self,
            step_id=step_id,
            mode=mode,
            sleep_preview=sleep_preview,
            batch_size=batch_size,
        )
        self.learner.save(self.config.state_path)
        self._emit_event(
            "curriculum",
            {
                **report.to_payload(),
                "rule_summary": self.rule_summary(),
                "sensory_prototype_summary": self.sensory_prototype_summary(),
            },
        )
        return report

    def rule_summary(self) -> dict[str, Any]:
        rules = self.memory.list_rules()
        by_status: dict[str, int] = {}
        for rule in rules:
            by_status[rule.status] = by_status.get(rule.status, 0) + 1
        return {
            "total": len(rules),
            "by_status": by_status,
            "top": [
                {
                    "name": rule.name,
                    "trigger_pattern": rule.trigger_pattern,
                    "action_tool": rule.action_tool,
                    "confidence": rule.confidence,
                    "status": rule.status,
                    "provenance_count": len(rule.provenance),
                    "counterexample_count": len(rule.counterexamples),
                }
                for rule in rules[:8]
            ],
        }

    def rule_link_summary(self) -> dict[str, Any]:
        links = self.memory.list_rule_links(limit=200)
        by_target_kind: dict[str, int] = {}
        by_relation: dict[str, int] = {}
        for link in links:
            by_target_kind[link.target_kind] = by_target_kind.get(link.target_kind, 0) + 1
            by_relation[link.relation] = by_relation.get(link.relation, 0) + 1
        return {
            "total": len(links),
            "by_target_kind": by_target_kind,
            "by_relation": by_relation,
            "latest": [
                {
                    "rule_id": link.rule_id,
                    "target_kind": link.target_kind,
                    "target_id": link.target_id,
                    "relation": link.relation,
                    "confidence": link.confidence,
                    "provenance_count": len(link.provenance),
                    "rule_name": link.payload.get("rule_name", ""),
                    "target_name": link.payload.get("target_name", link.payload.get("prototype_key", "")),
                }
                for link in links[:8]
            ],
        }

    def rule_conflict_summary(self) -> dict[str, Any]:
        conflicts = self.memory.list_rule_conflicts(limit=200)
        by_status: dict[str, int] = {}
        by_topic: dict[str, int] = {}
        for conflict in conflicts:
            by_status[conflict.status] = by_status.get(conflict.status, 0) + 1
            by_topic[conflict.topic] = by_topic.get(conflict.topic, 0) + 1
        return {
            "total": len(conflicts),
            "by_status": by_status,
            "by_topic": dict(sorted(by_topic.items(), key=lambda item: item[1], reverse=True)[:8]),
            "latest": [
                {
                    "topic": conflict.topic,
                    "status": conflict.status,
                    "left_rule_id": conflict.left_rule_id,
                    "right_rule_id": conflict.right_rule_id,
                    "reason": conflict.reason,
                    "left_tool": conflict.payload.get("left_action_tool", ""),
                    "right_tool": conflict.payload.get("right_action_tool", ""),
                }
                for conflict in conflicts[:8]
            ],
        }

    def memory_compaction_summary(self) -> dict[str, Any]:
        compactions = self.memory.recent_memory_compactions(limit=200)
        protected_total = sum(len(item.protected_episode_ids) for item in compactions)
        return {
            "total": len(compactions),
            "episode_count": sum(item.episode_count for item in compactions),
            "protected_evidence_count": protected_total,
            "top": [
                {
                    "key": item.key,
                    "episode_count": item.episode_count,
                    "protected_count": len(item.protected_episode_ids),
                    "rule_count": len(item.rule_ids),
                    "rule_link_count": len(item.rule_link_ids),
                    "summary": item.summary[:220],
                }
                for item in sorted(compactions, key=lambda record: record.episode_count, reverse=True)[:8]
            ],
        }

    def rule_records(self):
        return self.memory.list_rules()

    def rule_link_records(self, limit: int = 100):
        return self.memory.list_rule_links(limit=limit)

    def rule_conflict_records(self, limit: int = 100):
        return self.memory.list_rule_conflicts(limit=limit)

    def set_rule_status(self, rule_id: str, status: str, reason: str = ""):
        if status not in {"enabled", "disabled", "rejected"}:
            raise ValueError(f"unsupported rule status: {status}")
        rule = self.memory.set_rule_status(
            rule_id,
            status,
            payload={
                "last_manual_status": status,
                "last_manual_status_reason": reason,
                "last_manual_status_at": time(),
            },
        )
        self._emit_event(
            "rules",
            {
                "action": "status",
                "rule_id": rule.id,
                "rule_name": rule.name,
                "status": rule.status,
                "reason": reason,
            },
        )
        return rule

    def record_rule_feedback(self, rule_id: str, score: float, note: str = ""):
        clipped = float(np.clip(score, -1.0, 1.0))
        if clipped == 0.0:
            raise ValueError("rule feedback score cannot be zero")
        evidence_id = f"manual:{uuid.uuid4()}"
        if clipped > 0:
            confidence_delta = self.config.rule_success_boost * clipped
            success = True
            counterexample = False
        else:
            confidence_delta = self.config.rule_counterexample_penalty * clipped
            success = False
            counterexample = True
        self.memory.record_rule_evidence(
            rule_id,
            success=success,
            episode_id=evidence_id,
            confidence_delta=confidence_delta,
            counterexample=counterexample,
        )
        rule = self._rule_by_id(rule_id)
        self._emit_event(
            "rules",
            {
                "action": "feedback",
                "rule_id": rule.id,
                "rule_name": rule.name,
                "score": clipped,
                "note": note,
                "confidence": rule.confidence,
                "status": rule.status,
                "evidence_id": evidence_id,
            },
        )
        return rule

    def sensory_prototype_summary(self) -> dict[str, Any]:
        prototypes = self.memory.recent_sensory_prototypes(limit=200)
        by_modality: dict[str, int] = {}
        for prototype in prototypes:
            key = "+".join(prototype.modalities) or "unknown"
            by_modality[key] = by_modality.get(key, 0) + 1
        return {
            "total": len(prototypes),
            "by_modality": by_modality,
            "top": [
                {
                    "key": prototype.key,
                    "modalities": prototype.modalities,
                    "observation_count": prototype.observation_count,
                    "confidence": prototype.confidence,
                    "sources": prototype.sources[:4],
                }
                for prototype in sorted(
                    prototypes,
                    key=lambda item: (item.observation_count, item.confidence),
                    reverse=True,
                )[:8]
            ],
        }

    def learn_from_last_gap(self) -> LearningCycleReport:
        """Investigate the last interaction's weaknesses and consolidate proof."""
        if self._last_result is None:
            return LearningCycleReport(
                timestamp=time(),
                gaps=[],
                proposals=[],
                candidates=[],
                verdicts=[],
                consolidated=0,
            )
        report = self.gap_learner.run(self._last_result)
        self.learner.save(self.config.state_path)
        self._emit_event(
            "learning",
            {
                "summary": report.summary(),
                "gaps": [
                    {
                        "kind": gap.kind,
                        "description": gap.description,
                        "severity": gap.severity,
                    }
                    for gap in report.gaps
                ],
                "proposal_count": len(report.proposals),
                "candidate_count": len(report.candidates),
                "accepted_count": sum(1 for verdict in report.verdicts if verdict.accepted),
                "consolidated": report.consolidated,
            },
        )
        return report

    def recent_events(self, limit: int = 50, after_id: int | None = None):
        return self.memory.recent_events(limit=limit, after_id=after_id)

    def _rule_by_id(self, rule_id: str):
        for rule in self.memory.list_rules():
            if rule.id == rule_id:
                return rule
        raise KeyError(f"rule not found: {rule_id}")

    def _attach_rules_to_experts(
        self,
        rule_matches: list[RuleMatch],
        experts,
        episode_id: str,
        workspace_id: str,
    ) -> int:
        total = 0
        provenance = [item for item in [episode_id, workspace_id] if item]
        for expert in experts:
            total += self.rule_engine.attach_matches(
                rule_matches,
                target_kind="expert",
                target_id=expert.id,
                relation="influences_hot_expert",
                provenance=provenance,
                payload={
                    "target_name": expert.name,
                    "expert_kind": expert.kind,
                    "expert_status": expert.status,
                },
            )
        return total

    def _attach_rules_to_sensory_prototype(
        self,
        rule_matches: list[RuleMatch],
        learned_prototype,
        episode_id: str,
        event_id: str,
        workspace_id: str,
    ) -> int:
        if learned_prototype is None:
            return 0
        provenance = [item for item in [episode_id, event_id, workspace_id] if item]
        return self.rule_engine.attach_matches(
            rule_matches,
            target_kind="sensory_prototype",
            target_id=learned_prototype.id,
            relation="matches_sensory_prototype",
            provenance=provenance,
            payload={
                "prototype_key": learned_prototype.key,
                "modalities": learned_prototype.modalities,
                "observation_count": learned_prototype.observation_count,
            },
        )

    def _persist_cognitive_trace(
        self,
        workspace: CognitiveWorkspace,
        episode_id: str,
        input_text: str,
        action_plan: ActionPlan | None = None,
    ) -> None:
        payload = workspace.to_payload()
        if action_plan is not None:
            payload["action_plan"] = action_plan.to_payload()
        self.memory.store_cognitive_trace(
            trace_id=workspace.id,
            episode_id=episode_id,
            input_text=input_text,
            mode=workspace.mode,
            prediction=workspace.prediction,
            observation=workspace.observation,
            attention=list(workspace.attention_focus),
            surprise=workspace.surprise,
            uncertainty=workspace.uncertainty,
            payload=payload,
        )

    def _default_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(MemoryRecallTool(self.memory, top_k=self.config.memory_top_k))
        registry.register(CalculatorTool())
        if self.config.allow_file_tools:
            registry.register(FileSearchTool(self.config.workspace_root, limit=self.config.tool_result_limit))
            registry.register(FileReadTool(self.config.workspace_root))
        if self.config.allow_web:
            registry.register(WebSearchTool(timeout_s=self.config.web_timeout_s, limit=self.config.tool_result_limit))
        return registry

    def _build_event_mirror(self):
        if self.config.sync_to_convex and self.config.convex_url:
            return SafeEventMirror(
                ConvexEventMirror(
                    self.config.convex_url,
                    mutation=self.config.convex_event_mutation,
                )
            )
        return SafeEventMirror(NullEventMirror())

    def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        event = self.memory.append_event(event_type, payload)
        self.event_mirror.publish(event)

    def _choose_tools(
        self,
        text: str,
        embedding: np.ndarray,
        activation_confidence: float,
        max_tool_calls: int | None = None,
        rule_matches: list[RuleMatch] | None = None,
    ) -> list[str]:
        lowered = text.lower()
        selected: list[str] = ["memory_recall"]

        question_like = "?" in text or any(
            marker in lowered
            for marker in [
                "cherche",
                "recherche",
                "internet",
                "web",
                "source",
                "docs",
                "recent",
                "latest",
                "actuel",
                "aujourd",
            ]
        )

        path_match = re.search(r"[A-Za-z0-9_./\\-]+\.[A-Za-z0-9_]+", text)
        file_like = bool(path_match) or any(marker in lowered for marker in ["fichier", "code", ".py", ".md", "projet", "repo"])
        if self.config.allow_file_tools and file_like:
            if path_match:
                selected.append("file_read")
            selected.append("file_search")

        if self.config.allow_web and (question_like or activation_confidence < self.config.confidence_threshold):
            selected.append("web_search")

        if _looks_like_math(text):
            selected.append("calculator")

        for procedure in self.memory.retrieve_procedures(embedding, top_k=3):
            if procedure.similarity > 0.55 and procedure.success_rate >= 0.5:
                selected.append(procedure.tool_name)

        for match in rule_matches or []:
            selected.append(match.tool_name)

        deduped: list[str] = []
        for name in selected:
            if name not in deduped and self.registry.get(name) is not None:
                deduped.append(name)
        if max_tool_calls is None or max_tool_calls <= 0:
            return deduped
        return deduped[:max_tool_calls]

    def _run_tools(
        self,
        names: list[str],
        text: str,
        context: dict[str, Any],
    ) -> list[ToolResult]:
        return [self.registry.run(name, text, context) for name in names]

    def _tool_names_for_plan(
        self,
        proposed: list[str],
        action_plan: ActionPlan,
        mode: str | None = None,
    ) -> list[str]:
        if mode == "open":
            return [name for name in proposed if self.registry.get(name) is not None]
        if action_plan.selected.action == "use_tools":
            selected = ["memory_recall", *action_plan.selected.tool_names]
        elif action_plan.selected.action == "recall_memory":
            selected = ["memory_recall"]
        else:
            selected = []
        return [name for name in selected if name in proposed and self.registry.get(name) is not None]

    def _ensure_required_tools(
        self,
        text: str,
        proposed: list[str],
        planned: list[str],
    ) -> list[str]:
        required: list[str] = []
        if _looks_like_math(text) and "calculator" in proposed:
            required.append("calculator")
        merged = list(planned)
        for name in required:
            if name not in merged and self.registry.get(name) is not None:
                merged.append(name)
        return merged

    def _maybe_self_improve(self) -> ImprovementReport | None:
        if not self.config.auto_improve_enabled:
            return None
        if self._interactions_since_improvement < self.config.improvement_interval:
            return None
        return self.run_self_improvement()

    def _salience(
        self,
        novelty: float,
        memories: list[MemoryRecord],
        tool_results: list[ToolResult],
        feedback: Feedback | None,
    ) -> float:
        best_memory = max((memory.similarity for memory in memories), default=0.0)
        memory_novelty = 1.0 - max(0.0, best_memory)
        tool_signal = 0.15 if any(result.success and result.content for result in tool_results) else 0.0
        feedback_signal = abs(feedback.clipped_score()) if feedback else 0.0
        return float(np.clip(0.45 * novelty + 0.35 * memory_novelty + tool_signal + feedback_signal, 0.0, 1.0))

    def _learn_tool_preferences(
        self,
        text: str,
        embedding: np.ndarray,
        tool_results: list[ToolResult],
        feedback: Feedback | None,
    ) -> None:
        feedback_positive = feedback is not None and feedback.clipped_score() > 0
        feedback_negative = feedback is not None and feedback.clipped_score() < 0
        for result in tool_results:
            success = result.success and not feedback_negative
            if feedback_positive:
                success = True
            self.memory.record_tool_result(result.tool_name, success)
            if feedback is not None:
                self.memory.upsert_procedure(
                    name=f"use:{result.tool_name}:{self._procedure_key(text)}",
                    description=f"Use {result.tool_name} for interactions like: {text[:120]}",
                    tool_name=result.tool_name,
                    trigger_embedding=embedding,
                    success=success,
                    payload={"last_feedback": feedback.text, "tool_metadata": result.metadata},
                )

    def _procedure_key(self, text: str) -> str:
        words = re.findall(r"[A-Za-z0-9_]{3,}", text.lower())
        return "-".join(words[:5]) or "general"

    def _synthesize_response(
        self,
        text: str,
        activation,
        memories: list[MemoryRecord],
        conversation_examples: list[ConversationExampleRecord],
        tool_results: list[ToolResult],
        salience: float,
    ) -> tuple[str, dict[str, Any]]:
        decision = _conversation_decision_base(text, conversation_examples)
        for result in tool_results:
            if result.success and result.tool_name == "calculator":
                return result.content.strip(), {**decision, "reason": "tool:calculator", "accepted": False}

        normalized_input = _normalized_text(text)
        if conversation_examples:
            for example in conversation_examples:
                if _normalized_text(example.prompt) == normalized_input:
                    return example.response, _conversation_decision_selected(
                        text,
                        example,
                        reason="exact_prompt",
                    )
            compatible_examples = [
                example
                for example in conversation_examples
                if _conversation_example_is_compatible(text, example)
            ]
            if compatible_examples:
                best = compatible_examples[0]
                if best.similarity >= _conversation_similarity_threshold(text, best):
                    return best.response, _conversation_decision_selected(
                        text,
                        best,
                        reason="compatible_example",
                    )

        intent = _response_intent(text)
        if intent == "greeting":
            return (
                "Salut, je suis là. Je peux apprendre avec tes exemples, tes fichiers et tes retours.",
                {**decision, "reason": "greeting_fallback", "accepted": False},
            )
        if intent == "translation":
            return (
                (
                    "Je reconnais une demande de traduction, mais je n'ai pas encore trouvé "
                    "un exemple assez compatible pour répondre sans inventer."
                ),
                {**decision, "reason": "translation_gap", "accepted": False},
            )
        if intent == "math":
            return (
                "Je reconnais un problème de calcul, mais mon outil n'a pas encore produit de résultat fiable.",
                {**decision, "reason": "math_tool_gap", "accepted": False},
            )

        if conversation_examples:
            best = conversation_examples[0]
            return (
                "Je retrouve des exemples proches, mais ils ne correspondent pas assez à l'intention. "
                f"Meilleur exemple rejeté : {best.prompt[:180]}",
                {**decision, "reason": "incompatible_examples", "accepted": False},
            )

        if memories:
            best_memory = memories[0]
            return (
                "Je n'ai pas encore appris une réponse directe à ça. "
                f"Je retrouve surtout ceci : {best_memory.text[:240]}",
                {**decision, "reason": "episodic_memory_only", "accepted": False},
            )

        lines = [
            "SDNC interaction cycle complete.",
            f"confidence={activation.confidence:.3f} novelty={activation.novelty:.3f} salience={salience:.3f}",
            f"active_circuits={activation.indices}",
        ]
        if memories:
            best = memories[0]
            lines.append(f"best_memory sim={best.similarity:.3f}: {best.text[:160]}")
        else:
            lines.append("best_memory: none")
        for result in tool_results:
            status = "ok" if result.success else "error"
            content = result.content.replace("\n", " | ")[:500]
            lines.append(f"tool:{result.tool_name}:{status}: {content}")
        if not tool_results:
            lines.append("tools: none selected")
        return "\n".join(lines), {**decision, "reason": "diagnostic_fallback", "accepted": False}

    def _synthesize_observation_response(
        self,
        event: SensoryEvent,
        activation,
        memories: list[MemoryRecord],
        salience: float,
    ) -> str:
        lines = [
            f"Observed sensory event {event.id} from {event.source}.",
            f"modalities={'+'.join(event.modalities)} binding={event.binding_score:.3f} reliability={event.reliability:.3f}",
            f"confidence={activation.confidence:.3f} novelty={activation.novelty:.3f} salience={salience:.3f}",
            f"active_circuits={activation.indices}",
        ]
        if memories:
            best = memories[0]
            lines.append(f"nearest_memory sim={best.similarity:.3f}: {best.text[:160]}")
        else:
            lines.append("nearest_memory: none")
        return "\n".join(lines)


def _budget_payload(budget) -> dict[str, Any]:
    return {
        "mode": budget.mode,
        "memory_top_k": budget.memory_top_k,
        "max_tool_calls": budget.max_tool_calls,
        "max_context_segments": budget.max_context_segments,
        "max_context_chars": budget.max_context_chars,
        "allow_gap_learning": budget.allow_gap_learning,
        "allow_external_advisors": budget.allow_external_advisors,
        "max_hot_experts": budget.max_hot_experts,
        "notes": list(budget.notes),
    }


def _context_payload(packet) -> dict[str, Any]:
    return {
        "original_chars": packet.original_chars,
        "segment_count": len(packet.segments),
        "prototype_count": len(packet.prototypes),
        "compression_ratio": packet.compression_ratio,
        "estimated_tokens_saved": packet.estimated_tokens_saved,
        "global_summary": packet.global_summary[:500],
    }


def _resource_payload(snapshot) -> dict[str, Any]:
    return {
        "total_vram_gb": snapshot.total_vram_gb,
        "usable_vram_gb": snapshot.usable_vram_gb,
        "hot_vram_gb": round(snapshot.hot_vram_gb, 4),
        "cold_ram_gb": round(snapshot.cold_ram_gb, 4),
        "cold_disk_gb": round(snapshot.cold_disk_gb, 4),
        "within_budget": snapshot.within_budget,
        "components": [
            {
                "name": component.name,
                "kind": component.kind,
                "estimated_vram_gb": component.estimated_vram_gb,
                "estimated_ram_gb": component.estimated_ram_gb,
                "hot": component.hot,
            }
            for component in snapshot.components
        ],
    }


def _safe_filename(name: str) -> str:
    cleaned = Path(name or "upload.bin").name
    cleaned = re.sub(r"[^A-Za-z0-9_.() -]+", "_", cleaned).strip(" .")
    return cleaned[:160] or "upload.bin"


def _modality_for_file(name: str, content_type: str) -> str:
    suffix = Path(name).suffix.lower()
    content_type = content_type.lower()
    if suffix in {".parquet"}:
        return "dataset"
    if suffix in IMAGE_EXTENSIONS or content_type.startswith("image/"):
        return "image"
    if suffix in AUDIO_EXTENSIONS or content_type.startswith("audio/"):
        return "audio"
    if suffix in VIDEO_EXTENSIONS or content_type.startswith("video/"):
        return "video"
    if suffix in TEXT_EXTENSIONS or content_type.startswith("text/") or "json" in content_type:
        return "text"
    return "text"


def _file_preview(data: bytes, name: str, content_type: str, limit: int = 1200) -> str:
    suffix = Path(name).suffix.lower()
    if suffix == ".parquet":
        return f"{name} | parquet dataset | {len(data)} bytes"
    if suffix in TEXT_EXTENSIONS or content_type.startswith("text/") or "json" in content_type:
        return _decode_bytes(data)[:limit]
    return f"{name} | {content_type or 'application/octet-stream'} | {len(data)} bytes"


def _dataset_records_from_path(path: Path, row_limit: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix != ".parquet":
        raise RuntimeError(f"unsupported dataset file type: {suffix}")
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError("Install `polars` or a parquet reader to process parquet datasets.") from exc

    frame = pl.read_parquet(path)
    columns = list(frame.columns)
    text_columns = [name for name in ("source", "target", "text", "prompt", "question", "answer") if name in columns]
    if not text_columns:
        text_columns = [name for name, dtype in zip(frame.columns, frame.dtypes) if str(dtype).lower() in {"string", "str", "utf8"}]
    label_columns = [name for name in ("target", "label", "answer") if name in columns]
    selected = frame.head(row_limit) if row_limit is not None else frame
    records = [
        {key: ("" if value is None else str(value)) for key, value in row.items()}
        for row in selected.iter_rows(named=True)
    ]
    return records, {
        "path": str(path),
        "format": "parquet",
        "row_count": frame.height,
        "processed_rows": len(records),
        "row_limit": row_limit,
        "columns": columns,
        "text_columns": text_columns,
        "label_columns": label_columns,
    }


def _read_text_payload(path: Path, limit: int) -> str:
    data = path.read_bytes()[: max(1, limit * 4)]
    return _decode_bytes(data)[:limit]


def _decode_bytes(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(encoding, errors="strict")
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _sensory_event_payload(event: SensoryEvent, sensory_memories) -> dict[str, Any]:
    return {
        "id": event.id,
        "source": event.source,
        "modalities": event.modalities,
        "sample_ids": event.sample_ids,
        "binding_score": event.binding_score,
        "reliability": event.reliability,
        "signals": [
            {
                "modality": signal.modality,
                "source": signal.source,
                "sample_id": signal.sample_id,
                "label": signal.label,
                "reliability": signal.reliability,
                "summary": signal.summary[:500],
                "features": signal.features,
            }
            for signal in event.signals
        ],
        "nearby_bindings": [
            {
                "id": binding.id,
                "source": binding.source,
                "modalities": binding.modalities,
                "summary": binding.summary[:300],
                "similarity": binding.similarity,
                "binding_score": binding.binding_score,
                "salience": binding.salience,
            }
            for binding in sensory_memories
        ],
    }


def _sensory_prototypes_payload(
    matches: list[SensoryPrototypeMatch],
    learned_prototype,
) -> dict[str, Any]:
    return {
        "matches": [match.to_payload() for match in matches],
        "learned": (
            {
                "id": learned_prototype.id,
                "key": learned_prototype.key,
                "modalities": learned_prototype.modalities,
                "observation_count": learned_prototype.observation_count,
                "confidence": learned_prototype.confidence,
            }
            if learned_prototype
            else None
        ),
    }


def _conversation_examples_payload(examples: list[ConversationExampleRecord]) -> list[dict[str, Any]]:
    return [
        {
            "id": example.id,
            "source": example.source,
            "prompt": example.prompt[:240],
            "response": example.response[:360],
            "similarity": round(example.similarity, 4),
            "confidence": round(example.confidence, 4),
            "success_rate": round(example.success_rate, 4),
        }
        for example in examples
    ]


def _conversation_decision_base(
    text: str,
    examples: list[ConversationExampleRecord],
) -> dict[str, Any]:
    best = examples[0] if examples else None
    return {
        "intent": _response_intent(text),
        "accepted": False,
        "selected_id": "",
        "selected_similarity": 0.0,
        "selected_source": "",
        "selected_prompt": "",
        "rejected_count": len(examples),
        "best_rejected_id": best.id if best else "",
        "best_rejected_similarity": round(best.similarity, 4) if best else 0.0,
        "best_rejected_prompt": best.prompt[:240] if best else "",
        "reason": "no_conversation_example",
    }


def _conversation_decision_selected(
    text: str,
    example: ConversationExampleRecord,
    reason: str,
) -> dict[str, Any]:
    return {
        "intent": _response_intent(text),
        "accepted": True,
        "selected_id": example.id,
        "selected_similarity": round(example.similarity, 4),
        "selected_source": example.source,
        "selected_prompt": example.prompt[:240],
        "rejected_count": 0,
        "best_rejected_id": "",
        "best_rejected_similarity": 0.0,
        "best_rejected_prompt": "",
        "reason": reason,
    }


def _normalized_text(text: str) -> str:
    return " ".join(re.findall(r"[A-Za-z0-9_]+", text.lower()))


def _looks_like_math(text: str) -> bool:
    lowered = text.lower()
    if re.search(r"\d+\s*[-+*/%]\s*\d+", text):
        return True
    has_two_numbers = len(re.findall(r"\d+(?:[,.]\d+)?", text)) >= 2
    math_markers = [
        "calcule",
        "résous",
        "resous",
        "combien",
        "reste",
        "donne",
        "mange",
        "perd",
        "retire",
        "enlève",
        "enleve",
        "ajoute",
        "gagne",
        "fois",
        "divise",
        "somme",
        "soustra",
    ]
    return has_two_numbers and any(marker in lowered for marker in math_markers)


def _response_intent(text: str) -> str:
    lowered = text.lower()
    if _looks_like_math(text):
        return "math"
    if any(
        marker in lowered
        for marker in [
            "traduis",
            "traduire",
            "translate",
            "translation",
            "en français",
            "en francais",
            "en anglais",
            "en espagnol",
        ]
    ):
        return "translation"
    if any(marker in lowered for marker in ["convertis", "convertir", "majuscule", "minuscule"]):
        return "transform"
    if any(marker in lowered for marker in ["raconte", "histoire", "conte", "roman"]):
        return "story"
    if any(marker in lowered for marker in ["conseil", "conseils", "astuce", "astuces"]):
        return "advice"
    words = re.findall(r"[A-Za-zÀ-ÿ0-9_]+", lowered)
    if len(words) <= 8 and any(marker in lowered for marker in ["bonjour", "salut", "coucou", "comment vas"]):
        return "greeting"
    return "general"


def _conversation_example_is_compatible(
    user_text: str,
    example: ConversationExampleRecord,
) -> bool:
    user_intent = _response_intent(user_text)
    example_intent = _response_intent(example.prompt)
    if _normalized_text(user_text) == _normalized_text(example.prompt):
        return True
    if user_intent == "math":
        return False
    if user_intent == "greeting":
        return example_intent == "greeting"
    if user_intent in {"translation", "transform"}:
        return example_intent == user_intent and example.similarity >= 0.82
    if user_intent in {"story", "advice"}:
        return (
            example_intent == user_intent
            and example.similarity >= 0.68
            and _content_overlap(user_text, example.prompt) >= 0.34
        )
    if example_intent != user_intent and example_intent != "general":
        return False
    return example.similarity >= 0.72 or _content_overlap(user_text, example.prompt) >= 0.50


def _conversation_similarity_threshold(
    user_text: str,
    example: ConversationExampleRecord,
) -> float:
    intent = _response_intent(user_text)
    if _normalized_text(user_text) == _normalized_text(example.prompt):
        return 0.0
    if intent in {"translation", "transform"}:
        return 0.88
    if intent in {"story", "advice"}:
        return 0.72
    if intent == "greeting":
        return 0.62
    return 0.74


def _content_overlap(left: str, right: str) -> float:
    left_tokens = _content_tokens(left)
    right_tokens = _content_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens))


def _content_tokens(text: str) -> set[str]:
    stopwords = {
        "avec",
        "dans",
        "pour",
        "une",
        "des",
        "les",
        "est",
        "que",
        "qui",
        "quoi",
        "comment",
        "donne",
        "donner",
        "raconte",
        "histoire",
        "courte",
        "conseil",
        "conseils",
        "traduis",
        "traduire",
        "phrase",
        "contexte",
        "apprendre",
        "parler",
        "suivant",
        "suivante",
    }
    return {
        token
        for token in re.findall(r"[A-Za-zÀ-ÿ0-9_]+", text.lower())
        if len(token) > 2 and token not in stopwords
    }


def _sensory_features_payload(event: SensoryEvent) -> dict[str, Any]:
    return {
        "modalities": event.modalities,
        "binding_score": event.binding_score,
        "reliability": event.reliability,
        "signals": [
            {
                "modality": signal.modality,
                "sample_id": signal.sample_id,
                "label": signal.label,
                "reliability": signal.reliability,
                "features": signal.features,
            }
            for signal in event.signals
        ],
    }


def _expert_report_payload(report) -> dict[str, Any]:
    return {
        "selected_hot": [
            {
                "id": expert.id,
                "name": expert.name,
                "kind": expert.kind,
                "status": expert.status,
                "utility": expert.utility,
                "similarity": expert.similarity,
                "estimated_vram_gb": expert.estimated_vram_gb,
                "estimated_ram_gb": expert.estimated_ram_gb,
                "payload": _expert_payload_summary(expert.payload),
            }
            for expert in report.selected_hot
        ],
        "cooled": report.cooled,
        "retired": report.retired,
        "total_hot_vram_gb": report.total_hot_vram_gb,
        "total_hot_ram_gb": report.total_hot_ram_gb,
    }


def _expert_payload_summary(payload: dict[str, Any]) -> dict[str, Any] | None:
    raw_payload = payload.get("expert_payload") if isinstance(payload, dict) else None
    if not isinstance(raw_payload, dict):
        return None
    try:
        return payload_summary(raw_payload)
    except (KeyError, TypeError, ValueError):
        return {"integrity_ok": False, "error": "invalid expert payload"}


def _introspection_payload(
    *,
    mode: str,
    activation,
    memories: list[MemoryRecord],
    proposed_tools: list[str],
    executed_tools: list[str],
    tool_results: list[ToolResult],
    expert_report,
    rule_matches: list[RuleMatch],
    workspace: CognitiveWorkspace,
    action_plan: ActionPlan,
) -> dict[str, Any]:
    """Inspectable SDNC trace without relying on hidden chain-of-thought."""
    executed = set(executed_tools)
    successful = {result.tool_name for result in tool_results if result.success}
    failed = {result.tool_name for result in tool_results if not result.success}
    return {
        "mode": mode,
        "open_laboratory": mode == "open",
        "policy": "observe_all_proposals" if mode == "open" else action_plan.policy,
        "circuit_trace": [
            {
                "id": index,
                "weight": round(float(weight), 6),
                "score": round(float(score), 6),
                "state": "active",
            }
            for index, weight, score in zip(activation.indices, activation.weights, activation.scores)
        ],
        "tool_trace": [
            {
                "name": name,
                "proposed": True,
                "executed": name in executed,
                "success": name in successful,
                "failed": name in failed,
                "planner_selected": name in action_plan.selected.tool_names or name == "memory_recall",
                "reason": _tool_trace_reason(name, workspace),
            }
            for name in proposed_tools
        ],
        "skipped_tools": [name for name in proposed_tools if name not in executed],
        "memory_trace": [
            {
                "id": memory.id,
                "similarity": round(float(memory.similarity), 4),
                "salience": round(float(memory.salience), 4),
                "feedback_score": memory.feedback_score,
                "text": memory.text[:240],
            }
            for memory in memories[:20]
        ],
        "expert_trace": [
            {
                "id": expert.id,
                "name": expert.name,
                "kind": expert.kind,
                "status": expert.status,
                "utility": round(float(expert.utility), 4),
                "similarity": round(float(expert.similarity), 4),
                "hot": expert.hot,
            }
            for expert in expert_report.selected_hot
        ],
        "rule_trace": [
            {
                "rule_id": match.rule.id,
                "name": match.rule.name,
                "tool": match.tool_name,
                "confidence": round(float(match.rule.confidence), 4),
                "similarity": round(float(match.rule.similarity), 4),
                "score": round(float(match.score), 4),
            }
            for match in rule_matches
        ],
        "planner_trace": {
            "selected_action": action_plan.selected.action,
            "selected_tools": list(action_plan.selected.tool_names),
            "selected_score": round(float(action_plan.selected.score), 4),
            "candidates": [candidate.to_payload() for candidate in action_plan.candidates],
        },
        "workspace_trace": workspace.to_payload(),
    }


def _tool_trace_reason(name: str, workspace: CognitiveWorkspace) -> str:
    for slot in workspace.slots:
        if slot.kind == "tool" and slot.key == name:
            return str(slot.payload.get("reason") or slot.summary)
    return "learned or inherited proposal"


def _rule_matches_payload(matches: list[RuleMatch]) -> list[dict[str, Any]]:
    return [match.to_payload() for match in matches]


def _cognitive_workspace_payload(workspace: CognitiveWorkspace) -> dict[str, Any]:
    return workspace.to_payload()
