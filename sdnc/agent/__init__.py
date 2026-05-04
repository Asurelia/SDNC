"""Autonomous interaction layer for SDNC.

This package turns the SDNC idea into a runnable interaction system:
observe, activate sparse circuits, use tools, store memory, receive
feedback, and update only local associations.
"""

from sdnc.agent.config import AutonomousConfig
from sdnc.agent.system import InteractionLearningSystem

__all__ = ["AutonomousConfig", "InteractionLearningSystem"]
