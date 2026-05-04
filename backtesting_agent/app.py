"""
CCAR Backtesting Agent — Gradio Web UI
=======================================
Single-tab chatbox UI with dataset selector and auto-generated summary charts.

Run with:
    python backtesting_agent/app.py
"""
from __future__ import annotations

import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

_repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / "src"))

import gradio as gr
from backtesting_agent.agent import build_agent, _load_config

# ── Paths ─────────────────────────────────────────────────────────────────────
SAMPLE_DIR  = Path(__file__).parent / "sample_data"
ROUND1_PATH = str(SAMPLE_DIR / "ccar_round1.parquet")
ROUND2_PATH = str(SAMPLE_DIR / "ccar_round2.parquet")
OUTPUT_DIR  = str(Path(__file__).parent.parent / "backtesting_output")

# Timestamp updated on Clear — gallery only shows charts from the current session
_session_start_time: float = time.time()

# ── Build agent once at startup ───────────────────────────────────────────────
_PATH_HINT = f"""
Available sample data files (tell the user these paths if they ask):
  Round 1: {ROUND1_PATH}
  Round 2: {ROUND2_PATH}

IMPORTANT: Never call user_input() or input(). If a file path is not
specified by the user, ask them which dataset they want to use.
"""
agent = build_agent(extra_instructions=_PATH_HINT)


# ── Dataset choices from config ───────────────────────────────────────────────
def _get_dataset_choices() -> list[str]:
    cfg = _load_config()
    return list(cfg.get("datasets", {}).keys())

def _get_dataset_path(dataset_name: str) -> str:
    cfg = _load_config()
    return cfg.get("datasets", {}).get(dataset_name, "")


# ── Summary chart generation ──────────────────────────────────────────────────
def generate_summary_charts(dataset_name: str):
    """
    For each target variable in config, generate a side-by-side bar chart of
    actual vs predicted by statement month at portfolio level.

    'Average across horizons' means:
      Step 1 — for each (statement_month, horizon): sum actual/predicted across accounts
      Step 2 — for each statement_month: average those horizon sums
    """
    if not dataset_name:
        return None

    try:
        import polars as pl
    except ImportError:
        return None

    cfg       = _load_config()
    datasets  = cfg.get("datasets", {})
    targets   = cfg.get("target_variables", {})
    time_cols = cfg.get("time_columns", {})
    stmt_col  = time_cols.get("statement_month", "statement_month")
    horizon_col = time_cols.get("horizon", "horizon")

    path = datasets.get(dataset_name, "")
    if not path or not os.path.exists(path):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f"Dataset not found:\n{path}",
                transform=ax.transAxes, ha="center", va="center", color="red")
        ax.axis("off")
        plt.tight_layout()
        return fig

    if not targets:
        return None

    n   = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]

    colors = {"actual": "#4c72b0", "predicted": "#dd8452"}

    for ax, (target_name, cols) in zip(axes, targets.items()):
        actual_col = cols.get("actual", "")
        pred_col   = cols.get("predicted", "")

        try:
            needed = list({stmt_col, horizon_col, actual_col, pred_col})
            lf = pl.scan_parquet(path).select(needed)

            # Step 1: portfolio sum per (statement_month, horizon)
            agg1 = (
                lf.group_by([stmt_col, horizon_col])
                  .agg([pl.col(actual_col).sum().alias("act"),
                        pl.col(pred_col).sum().alias("pred")])
                  .collect()
                  .to_pandas()
            )

            # Step 2: average across horizons per statement_month
            agg2 = (
                agg1.groupby(stmt_col, as_index=False)
                    .agg({"act": "mean", "pred": "mean"})
                    .sort_values(stmt_col)
                    .reset_index(drop=True)
            )

            labels = [str(d)[:7] for d in agg2[stmt_col]]
            x      = np.arange(len(labels))
            width  = 0.35

            ax.bar(x - width / 2, agg2["act"]  / 1e6, width,
                   label="Actual",    color=colors["actual"],    alpha=0.85)
            ax.bar(x + width / 2, agg2["pred"] / 1e6, width,
                   label="Predicted", color=colors["predicted"], alpha=0.85)

            # Thin out x-tick labels so they don't overlap
            step = max(1, len(labels) // 12)
            ax.set_xticks(x[::step])
            ax.set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Value ($M)")
            ax.set_title(
                f"{target_name}\nActual vs Predicted by Statement Month\n"
                f"(Portfolio level, avg across horizons)",
                fontsize=10,
            )
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")

        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading {target_name}:\n{e}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="red")
            ax.set_title(target_name)

    fig.suptitle(
        f"Dataset: {dataset_name} — Portfolio Summary (avg across horizons)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    return fig


# ── Example prompts ───────────────────────────────────────────────────────────
EXAMPLES = [
    f"Inspect the file at '{ROUND1_PATH}' and summarise its schema.",
    f"Show me the Payment MPE trend by statement month at portfolio level using '{ROUND1_PATH}'.",
    f"Show me EOS actual vs predicted by horizon at portfolio level using '{ROUND1_PATH}'.",
    f"Show me PurchaseVolume AMPE by statement month at account level using '{ROUND1_PATH}'.",
    f"Compare Round 1 ('{ROUND1_PATH}') and Round 2 ('{ROUND2_PATH}') Payment MPE by horizon.",
]


# ── Ambiguity pre-flight check ────────────────────────────────────────────────
def _build_clarify_system() -> str:
    cfg      = _load_config()
    datasets = cfg.get("datasets", {})
    targets  = cfg.get("target_variables", {})

    dataset_block = ""
    if datasets:
        lines = ["Known dataset names (treat any of these as a valid file reference):"]
        for name, path in datasets.items():
            lines.append(f"  - \"{name}\" → {path}")
        dataset_block = "\n" + "\n".join(lines)

    target_names = ", ".join(targets.keys()) if targets else "Payment, PurchaseVolume, EOS"

    return f"""You are a request parser for a backtesting analysis tool.
Review the FULL conversation and decide if any critical information is STILL unanswered.
Consider what the user has already provided in prior messages before asking anything.
{dataset_block}

The available target variables are: {target_names}.

Critical missing information means ANY of the following:
1. No dataset/file path is provided AND the user has not referenced a known dataset name.
   If the user mentions a known dataset name (e.g. "round1"), treat it as resolved.
2. The target variable is ambiguous — not stated and not obvious from context.
3. The analysis level is not specified (portfolio or account?).
4. A plot of raw values is requested and it is unclear whether to show actual,
   predicted, or both — skip this if the user is asking for MPE or AMPE.

Do NOT ask about EOS stock vs flow logic — the agent will handle that.

RULES (follow strictly):
- Check ALL four points above in one pass.
- If ONE OR MORE pieces are missing, ask about ALL of them in a SINGLE combined
  message. Never ask one question now and save others for later.
- Keep the combined question under 80 words.
- If nothing is missing, reply with exactly: PROCEED"""

_CLARIFY_SYSTEM = _build_clarify_system()


def _check_ambiguity(user_message: str, history: list) -> str | None:
    try:
        import litellm
        api_key = os.environ.get("OPENAI_API_KEY", "")
        model   = os.environ.get("OPENAI_MODEL", "gpt-4o")
        messages = [{"role": "system", "content": _CLARIFY_SYSTEM}]
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})
        resp = litellm.completion(model=model, messages=messages, api_key=api_key,
                                  max_tokens=150, temperature=0)
        answer = resp.choices[0].message.content.strip()
        return None if answer.upper().startswith("PROCEED") else answer
    except Exception:
        return None


# ── Chart gallery helpers ─────────────────────────────────────────────────────
def _collect_charts(since: float = 0.0) -> list[str]:
    pngs = glob.glob(os.path.join(OUTPUT_DIR, "**", "*.png"), recursive=True)
    if since:
        pngs = [p for p in pngs if os.path.getmtime(p) >= since]
    return sorted(pngs, key=os.path.getmtime, reverse=True)


def _session_charts() -> list[str]:
    return _collect_charts(since=_session_start_time)


# ── Confirmation detection ────────────────────────────────────────────────────
_CONFIRM_WORDS = {"yes", "y", "yes.", "yes!", "yep", "proceed", "go", "go ahead",
                  "ok", "okay", "sure", "correct", "confirmed", "confirm", "do it"}


def _is_confirmation(msg: str) -> bool:
    return msg.strip().lower().rstrip(".!") in _CONFIRM_WORDS


def _agent_already_engaged(history: list) -> bool:
    for m in history:
        if m["role"] != "assistant":
            continue
        c = m["content"]
        if (len(c) > 200
                or "Here is my plan" in c
                or "Shall I proceed" in c
                or "Data source:" in c
                or "Chart saved" in c):
            return True
    return False


def _build_task(user_message: str, history: list) -> str:
    prior = history[:-1]
    if not prior:
        return user_message
    recent = prior[-6:]
    lines = ["Conversation so far (use this as full context for the request below):"]
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"  {role}: {msg['content']}")
    lines.append(f"\nUser's latest message: {user_message}")

    if _is_confirmation(user_message):
        plan_text = ""
        for m in reversed(prior):
            if m["role"] == "assistant" and "Here is my plan" in m.get("content", ""):
                plan_text = m["content"]
                break

        cfg     = _load_config()
        targets = cfg.get("target_variables", {})
        col_info = []
        for tname, tcols in targets.items():
            col_info.append(
                f"  {tname}: actual_column='{tcols.get('actual','')}', "
                f"predicted_column='{tcols.get('predicted','')}', "
                f"metric_type='{tcols.get('metric_type','flow')}'"
            )
        col_block = "\n".join(col_info) if col_info else "  (see config.yaml)"

        return (
            f"The user confirmed the plan below. Execute it NOW using the provided tools.\n\n"
            f"{plan_text}\n\n"
            f"EXACT COLUMN NAMES (from config.yaml — use these, do not guess):\n"
            f"{col_block}\n\n"
            f"MANDATORY EXECUTION STEPS — write these exact tool calls:\n"
            f"Step 1 (code block 1):\n"
            f"  result = aggregate_credit_card(\n"
            f"      file_path=<dataset path from plan>,\n"
            f"      actual_column=<actual_column from above>,\n"
            f"      predicted_column=<predicted_column from above>,\n"
            f"      metric_type=<metric_type from above>,\n"
            f"      dimension=<dimension from plan>,\n"
            f"      dataset_name='agg_result',\n"
            f"      level=<level from plan>,\n"
            f"      row_filter=<filter from plan or empty string>,\n"
            f"  )\n"
            f"  print(result)\n\n"
            f"Step 2 (code block 2):\n"
            f"  chart_path = plot_trend(\n"
            f"      x_column=<dimension column from plan>,\n"
            f"      y_columns='plot_actual,plot_predicted',\n"
            f"      dataset_name='agg_result',\n"
            f"      title=<descriptive title>,\n"
            f"      y_unit='millions',\n"
            f"  )\n"
            f"  print(chart_path)\n\n"
            f"Step 3: final_answer('Data source: <path>\\nChart saved to: <chart_path>')\n\n"
            f"RULES:\n"
            f"- Use ONLY aggregate_credit_card and plot_trend — do NOT write raw pandas or matplotlib code.\n"
            f"- Do NOT re-present the plan. Do NOT ask 'Shall I proceed?' again.\n"
            f"- Do NOT call final_answer() before Steps 1 and 2 are complete."
        )
    else:
        lines.append(
            "\nComplete the user's latest request using all context above. "
            "Do not ask again for clarifications already answered in the conversation. "
            "However, you MUST still follow the MANDATORY WORKFLOW: present the full "
            "execution plan and ask 'Shall I proceed? (yes/no)' before calling any tool, "
            "even if all information is already clear from context."
        )
    return "\n".join(lines)


# ── Agent runner ──────────────────────────────────────────────────────────────
def run_agent(user_message: str, history: list, dataset_name: str) -> tuple:
    """
    Returns: (history, status, gallery_value, gallery_visible, summary_visible)
    After the first request, the summary chart is hidden and the gallery is shown.
    """
    if not user_message.strip():
        return history, "Enter a request above.", _session_charts(), gr.update(), gr.update()

    history = history + [{"role": "user", "content": user_message}]

    # Pre-flight clarification (only before agent is first engaged)
    if not _agent_already_engaged(history[:-1]):
        clarifying_question = _check_ambiguity(user_message, history[:-1])
        if clarifying_question:
            history = history + [{"role": "assistant", "content": clarifying_question}]
            return (history, "Waiting for clarification…",
                    _session_charts(), gr.update(), gr.update())

    charts_before = set(_collect_charts())
    task = _build_task(user_message, history)

    if _is_confirmation(user_message):
        try:
            agent.memory.reset()
        except Exception:
            pass

    try:
        result = agent.run(task)
    except Exception as e:
        result = f"ERROR: {e}"

    response  = str(result)
    new_charts = [c for c in _collect_charts() if c not in charts_before]
    if new_charts:
        new_names  = ", ".join(os.path.basename(c) for c in new_charts)
        response  += f"\n\n📊 New chart(s) generated: {new_names}"

    history = history + [{"role": "assistant", "content": response}]
    status  = (f"✅ Done — {len(new_charts)} new chart(s) generated."
               if new_charts else "✅ Done — no new charts.")

    # Switch right column: hide summary, show gallery
    return (history, status, _session_charts(),
            gr.update(visible=True),   # gallery
            gr.update(visible=False))  # summary_plot


def clear_all(dataset_name: str) -> tuple:
    """
    Returns: (history, status, gallery_value, gallery_visible, summary_fig, summary_visible)
    Resets session and restores summary chart.
    """
    global _session_start_time
    _session_start_time = time.time()
    try:
        agent.memory.reset()
    except Exception:
        pass
    summary_fig = generate_summary_charts(dataset_name)
    return ([], "Ready.",
            [],                                         # gallery value (empty)
            gr.update(visible=False),                   # gallery hidden
            gr.update(value=summary_fig, visible=True)) # summary shown


# ── Build the Gradio UI ───────────────────────────────────────────────────────
_dataset_choices = _get_dataset_choices()
_default_dataset = _dataset_choices[0] if _dataset_choices else None

with gr.Blocks(title="CCAR Backtesting Agent",
               theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:

    gr.Markdown("# CCAR Backtesting Analysis Agent")

    # ── Dataset selector ──────────────────────────────────────────────────────
    with gr.Row():
        dataset_dropdown = gr.Dropdown(
            choices=_dataset_choices,
            value=_default_dataset,
            label="Dataset",
            info="Select a dataset — summary charts will load automatically.",
            scale=1,
            min_width=200,
        )

    # ── Main layout ───────────────────────────────────────────────────────────
    with gr.Row():

        # Left: chatbox
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=520, type="messages")
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="e.g. Show me Payment MPE by statement month at portfolio level…",
                    label="Your request", scale=5, lines=2, autofocus=True)
                with gr.Column(scale=1, min_width=120):
                    submit_btn = gr.Button("▶ Run", variant="primary")
                    clear_btn  = gr.Button("🗑 Clear")
            status_box = gr.Textbox(label="Status", interactive=False, lines=1)

            gr.Markdown("### Quick-start examples")
            for ex in EXAMPLES:
                gr.Button(ex[:90] + "…", size="sm").click(
                    fn=lambda e=ex: e, outputs=msg_box)

        # Right: summary (shown on load) / gallery (shown after first request)
        with gr.Column(scale=2):

            # Summary plot — visible by default
            summary_plot = gr.Plot(
                label="Dataset Summary",
                visible=True,
            )

            # Agent gallery — hidden until first request
            gallery = gr.Gallery(
                label="Generated Charts (newest first)",
                columns=2,
                height=560,
                object_fit="contain",
                show_label=True,
                visible=False,
            )
            refresh_btn = gr.Button("🔄 Refresh Gallery", size="sm", visible=False)

    with gr.Accordion("📁 Sample data file paths", open=False):
        gr.Markdown(f"""
Copy these paths into your requests:

| Round | Path |
|---|---|
| CCAR Round 1 | `{ROUND1_PATH}` |
| CCAR Round 2 | `{ROUND2_PATH}` |

Output charts saved to: `{os.path.abspath(OUTPUT_DIR)}`
""")

    # ── Events ────────────────────────────────────────────────────────────────

    # Dataset change → regenerate summary, restore summary view
    def on_dataset_change(dataset_name):
        fig = generate_summary_charts(dataset_name)
        return (gr.update(value=fig, visible=True),   # summary shown
                gr.update(visible=False),              # gallery hidden
                gr.update(visible=False))              # refresh btn hidden

    dataset_dropdown.change(
        fn=on_dataset_change,
        inputs=[dataset_dropdown],
        outputs=[summary_plot, gallery, refresh_btn],
    )

    # Submit / Enter → run agent, switch to gallery
    submit_btn.click(
        fn=run_agent,
        inputs=[msg_box, chatbot, dataset_dropdown],
        outputs=[chatbot, status_box, gallery, gallery, summary_plot],
    ).then(fn=lambda: "", outputs=msg_box)

    msg_box.submit(
        fn=run_agent,
        inputs=[msg_box, chatbot, dataset_dropdown],
        outputs=[chatbot, status_box, gallery, gallery, summary_plot],
    ).then(fn=lambda: "", outputs=msg_box)

    # Clear → reset, restore summary
    clear_btn.click(
        fn=clear_all,
        inputs=[dataset_dropdown],
        outputs=[chatbot, status_box, gallery, gallery, summary_plot],
    )

    # Refresh gallery (only meaningful when gallery is visible)
    refresh_btn.click(fn=_session_charts, outputs=gallery)

    # Auto-load summary on page open
    demo.load(
        fn=generate_summary_charts,
        inputs=[dataset_dropdown],
        outputs=[summary_plot],
    )


if __name__ == "__main__":
    import signal, subprocess, platform

    # Kill any process already using port 7860
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(["lsof", "-ti", "tcp:7860"], capture_output=True, text=True)
            for pid in result.stdout.strip().splitlines():
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except Exception:
                    pass
        else:
            subprocess.run(["fuser", "-k", "7860/tcp"], capture_output=True)
    except Exception:
        pass

    # JupyterHub proxy support
    _jupyter_prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "").rstrip("/")
    _root_path = f"{_jupyter_prefix}/proxy/7860" if _jupyter_prefix else ""

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        root_path=_root_path,
        share=False,
        inbrowser=False,
    )
