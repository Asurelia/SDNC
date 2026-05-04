"""Optional remote sync backends for SDNC.

SQLite is the local source of truth. These helpers mirror local events outward
without making SDNC depend on a cloud database.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Protocol

from sdnc.agent.memory import SyncEvent


class EventMirror(Protocol):
    """Best-effort remote event mirror."""

    def publish(self, event: SyncEvent) -> None:
        ...


class NullEventMirror:
    """No-op mirror used by default."""

    def publish(self, event: SyncEvent) -> None:
        return None


class ConvexEventMirror:
    """Optional Convex mirror using the official Python client if installed.

    The Convex backend must expose a mutation that accepts one event object.
    Suggested mutation name: `sdnc:ingestEvent`.
    """

    def __init__(self, convex_url: str, mutation: str = "sdnc:ingestEvent"):
        try:
            from convex import ConvexClient
        except ImportError as exc:
            raise RuntimeError(
                "Convex sync requested but the `convex` Python package is not installed."
            ) from exc
        self.client = ConvexClient(convex_url)
        self.mutation = mutation

    def publish(self, event: SyncEvent) -> None:
        self.client.mutation(
            self.mutation,
            {
                "id": event.id,
                "timestamp": event.timestamp,
                "eventType": event.event_type,
                "source": event.source,
                "payload": event.payload,
            },
        )


class SafeEventMirror:
    """Wrap a mirror so sync failures never break SDNC learning."""

    def __init__(self, mirror: EventMirror):
        self.mirror = mirror
        self.last_error: str | None = None

    def publish(self, event: SyncEvent) -> None:
        try:
            self.mirror.publish(event)
            self.last_error = None
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"


def event_to_dict(event: SyncEvent) -> dict[str, Any]:
    return asdict(event)
