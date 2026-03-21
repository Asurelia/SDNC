"""
Mode interactif BRAIN-HYBRID.

Usage : python -m brain_hybrid.chat
"""

import os
import sys
import glob
from .model import BrainHybridModel
from .config import BrainConfig

CHECKPOINT_DIR = "checkpoints"
SAVE_EVERY = 10


def find_latest_checkpoint():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "step_*.pt")))
    return files[-1] if files else None


def save_checkpoint(model):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"step_{model.step_count:06d}.pt")
    model.save_state(path)


def print_status(result):
    errs = " | ".join(f"{e:.4f}" for e in result["prediction_errors"])
    dops = " | ".join(f"{d:.3f}" for d in result["dopamine"])
    print(f"  Erreurs     : [{errs}]")
    print(f"  Dopamine    : [{dops}]")
    print(f"  Err moyenne : {result['mean_error']:.4f}")
    print(f"  Memoires    : {result['memories_stored']}")
    print(f"  Step        : {result['step']}")


def main():
    print("=" * 50)
    print("  BRAIN-HYBRID — Mode interactif")
    print("=" * 50)
    print()

    config = BrainConfig()
    model = BrainHybridModel(config)

    ckpt = find_latest_checkpoint()
    if ckpt:
        model.load_state(ckpt)
        print(f"Reprise depuis {ckpt}")
    else:
        print("Nouveau run — aucun checkpoint")

    print()
    print("Commandes :")
    print("  remember <query>  — interroger l'hippocampe")
    print("  stats             — statistiques hippocampe")
    print("  save              — sauvegarder maintenant")
    print("  quit / exit       — quitter (sauvegarde auto)")
    print()

    try:
        while True:
            try:
                user_input = input(f"[step {model.step_count}] > ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                break

            if user_input.lower() == "stats":
                print(f"  {model.hippocampus.stats()}")
                if model.error_history:
                    print(f"  Err init: {model.error_history[0]:.4f}")
                    print(f"  Err last: {model.error_history[-1]:.4f}")
                continue

            if user_input.lower() == "save":
                save_checkpoint(model)
                continue

            if user_input.lower().startswith("remember"):
                query = user_input[8:].strip()
                if not query:
                    print("  Usage : remember <query>")
                    continue
                recalled = model.remember(query)
                norm = recalled.norm().item()
                if norm < 1e-6:
                    print("  Aucun souvenir trouve.")
                else:
                    print(f"  Souvenir recupere — norme : {norm:.4f}")
                continue

            # Forward + learn
            result = model.forward(user_input, learn=True)
            print()
            print_status(result)
            print(f"  Reponse   : {result['response'][:120]}")
            print()

            # Sauvegarde auto
            if model.step_count % SAVE_EVERY == 0:
                save_checkpoint(model)

    except KeyboardInterrupt:
        print()

    # Sauvegarde finale
    if model.step_count > 0:
        save_checkpoint(model)
        print("Sauvegarde finale effectuee.")

    print("Au revoir.")


if __name__ == "__main__":
    main()
