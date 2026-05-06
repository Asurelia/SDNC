"""Command-line interface for the autonomous SDNC loop."""

from __future__ import annotations

import argparse
from pathlib import Path

from sdnc.agent.config import AutonomousConfig
from sdnc.agent.system import InteractionLearningSystem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SDNC interaction learner.")
    parser.add_argument("--memory", type=Path, default=Path("data/autonomous_memory.sqlite3"))
    parser.add_argument("--state", type=Path, default=Path("data/autonomous_circuits.npz"))
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--no-web", action="store_true", help="Disable web search tool.")
    parser.add_argument("--sync-to-convex", action="store_true", help="Mirror local events to Convex.")
    parser.add_argument("--convex-url", default=None)
    parser.add_argument("--convex-mutation", default="sdnc:ingestEvent")
    parser.add_argument("--once", type=str, default=None, help="Run one interaction and exit.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = AutonomousConfig(
        memory_path=args.memory,
        state_path=args.state,
        workspace_root=args.workspace,
        allow_web=not args.no_web,
        sync_to_convex=args.sync_to_convex,
        convex_url=args.convex_url,
        convex_event_mutation=args.convex_mutation,
    )
    system = InteractionLearningSystem(config)
    try:
        if args.once:
            result = system.interact(args.once)
            print(result.response)
            return 0

        print(
            "SDNC interaction learner. Commands: /feedback <score> [text], "
            "/recent, /improve, /sleep, /sleep-preview, /compact, "
            "/compact-preview, /rules, /learn, /quit"
        )
        while True:
            text = input("sdnc> ").strip()
            if not text:
                continue
            if text in {"/quit", "/exit"}:
                return 0
            if text.startswith("/feedback"):
                _, _, rest = text.partition(" ")
                score_raw, _, note = rest.partition(" ")
                try:
                    score = float(score_raw)
                except ValueError:
                    print("Usage: /feedback <score -1..1> [note]")
                    continue
                print(system.give_feedback(score, note).response)
                continue
            if text.startswith("/recent"):
                for memory in system.recent_memories(limit=5):
                    print(f"{memory.id[:8]} salience={memory.salience:.3f}: {memory.text[:160]}")
                continue
            if text.startswith("/improve"):
                print(system.run_self_improvement().summary())
                continue
            if text.startswith("/sleep-preview"):
                print(system.run_sleep_cycle(preview=True).summary())
                continue
            if text.startswith("/sleep"):
                print(system.run_sleep_cycle().summary())
                continue
            if text.startswith("/compact-preview"):
                print(system.run_memory_compaction(preview=True).summary())
                continue
            if text.startswith("/compact"):
                print(system.run_memory_compaction().summary())
                continue
            if text.startswith("/rules"):
                print(system.run_rule_consolidation().summary())
                continue
            if text.startswith("/learn"):
                print(system.learn_from_last_gap().summary())
                continue
            result = system.interact(text)
            print(result.response)
    finally:
        system.close()


if __name__ == "__main__":
    raise SystemExit(main())
