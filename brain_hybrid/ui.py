"""
Interface visuelle BRAIN-HYBRID (Gradio).

Usage : python -m brain_hybrid.ui
"""

import os
import glob
import gradio as gr
import pandas as pd
from .model import BrainHybridModel
from .config import BrainConfig

CHECKPOINT_DIR = "checkpoints"
SAVE_EVERY = 10

# --- Global model ---
model = None


def load_model():
    global model
    config = BrainConfig(model_name="Qwen/Qwen3-4B")
    model = BrainHybridModel(config)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "step_*.pt")))
    if files:
        model.load_state(files[-1])


def save_checkpoint():
    if model and model.step_count > 0:
        path = os.path.join(CHECKPOINT_DIR, f"step_{model.step_count:06d}.pt")
        model.save_state(path)


def chat_fn(message, history):
    if not message.strip():
        return "", history

    # Remember command
    if message.strip().lower().startswith("remember"):
        query = message.strip()[8:].strip()
        if not query:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "Usage : `remember <query>`"})
            return "", history
        recalled = model.remember(query)
        norm = recalled.norm().item()
        if norm < 1e-6:
            resp = "Aucun souvenir trouve."
        else:
            resp = f"Souvenir recupere — norme : **{norm:.4f}**"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": resp})
        return "", history

    # Forward + learn
    result = model.forward(message, learn=True)

    # Auto-save
    if model.step_count % SAVE_EVERY == 0:
        save_checkpoint()

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": result["response"]})
    return "", history


def get_metrics_md():
    if model is None or model.step_count == 0:
        return "En attente..."
    last_err = model.error_history[-1] if model.error_history else 0
    dop = model.stdp_learners[0].dopamine_signal
    mem_stats = model.hippocampus.stats()
    return f"""| Metrique | Valeur |
|----------|--------|
| **Step** | {model.step_count} |
| **Erreur moyenne** | {last_err:.4f} |
| **Dopamine** | {dop:.4f} |
| **Memoires** | {mem_stats['memories_stored']} |
| **Locations actives** | {mem_stats['active_locations']} / {mem_stats['total_locations']} |
| **Usage hippocampe** | {mem_stats['usage_pct']:.1f}% |
| **Global state norm** | {model.global_state.norm().item():.4f} |"""


def get_error_plot():
    if model is None or len(model.error_history) < 2:
        return pd.DataFrame({"step": [0], "error": [0]})
    return pd.DataFrame({
        "step": list(range(1, len(model.error_history) + 1)),
        "error": model.error_history,
    })


def get_dopamine_plot():
    if model is None or model.step_count == 0:
        return pd.DataFrame({"module": ["M0"], "dopamine": [0.5]})
    names = [f"M{i}" for i in range(len(model.stdp_learners))]
    vals = [s.dopamine_signal for s in model.stdp_learners]
    return pd.DataFrame({"module": names, "dopamine": vals})


def build_ui():
    with gr.Blocks(
        title="BRAIN-HYBRID",
        fill_width=True,
    ) as demo:

        gr.Markdown("# BRAIN-HYBRID — Cerveau Artificiel")
        gr.Markdown("Qwen3-4B + CfC + SNN + STDP + Hippocampe | Tape `remember <query>` pour interroger la memoire")

        timer = gr.Timer(3)

        with gr.Row():
            # LEFT — Chat
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    height=520,
                    placeholder="Ecris quelque chose pour que le cerveau apprenne...",
                )
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Message...",
                        show_label=False,
                        scale=4,
                    )
                    send_btn = gr.Button("Envoyer", variant="primary", scale=1)
                with gr.Row():
                    save_btn = gr.Button("Sauvegarder", size="sm")
                    clear_btn = gr.ClearButton([msg, chatbot], value="Effacer", size="sm")

            # RIGHT — Metrics
            with gr.Column(scale=1, min_width=340):
                gr.Markdown("### Metriques en temps reel")
                metrics_md = gr.Markdown(value=get_metrics_md, every=timer)

                gr.Markdown("### Erreur de prediction")
                error_plot = gr.LinePlot(
                    value=get_error_plot,
                    every=timer,
                    x="step",
                    y="error",
                    title="",
                    y_title="Erreur",
                    x_title="Step",
                    height=220,
                )

                gr.Markdown("### Dopamine par module")
                dop_plot = gr.BarPlot(
                    value=get_dopamine_plot,
                    every=timer,
                    x="module",
                    y="dopamine",
                    title="",
                    y_title="Signal",
                    height=200,
                    y_lim=[0, 1],
                )

        # Events
        msg.submit(chat_fn, [msg, chatbot], [msg, chatbot])
        send_btn.click(chat_fn, [msg, chatbot], [msg, chatbot])
        save_btn.click(save_checkpoint)

    return demo


def main():
    print("Chargement du modele...")
    load_model()
    print("Lancement de l'interface...")
    demo = build_ui()
    demo.queue().launch(inbrowser=True)


if __name__ == "__main__":
    main()
