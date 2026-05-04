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

    cfg         = _load_config()
    datasets    = cfg.get("datasets", {})
    targets     = cfg.get("target_variables", {})
    time_cols   = cfg.get("time_columns", {})
    stmt_col    = time_cols.get("statement_month", "statement_month")
    horizon_col = time_cols.get("horizon", "horizon")
    perf_col    = time_cols.get("performance_month", "performance_month")
    row_filter  = cfg.get("default_row_filter", "")

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
    fig, axes = plt.subplots(n, 1, figsize=(10, 5 * n))
    if n == 1:
        axes = [axes]

    MAX_HORIZONS = 28
    BAR_COLOR = "#4c72b0"

    for ax, (target_name, cols) in zip(axes, targets.items()):
        actual_col  = cols.get("actual", "")
        pred_col    = cols.get("predicted", "")
        metric_type = cols.get("metric_type", "flow")
        agg_fn      = "sum" if metric_type == "flow" else "mean"
        agg_label   = "sum" if metric_type == "flow" else "avg"

        try:
            needed = list({stmt_col, horizon_col, perf_col, actual_col, pred_col})
            lf = pl.scan_parquet(path).select(needed)

            # Apply default row filter from config.yaml
            if row_filter:
                df_raw = lf.collect().to_pandas()
                try:
                    df_raw = df_raw.query(row_filter)
                except Exception:
                    pass
                lf = pl.from_pandas(df_raw).lazy()

            # Step 1: portfolio sum per (statement_month, horizon)
            agg1 = (
                lf.group_by([stmt_col, horizon_col])
                  .agg([pl.col(actual_col).sum().alias("act"),
                        pl.col(pred_col).sum().alias("pred")])
                  .collect()
                  .to_pandas()
            )

            # Track max horizon per statement_month to detect incomplete months
            max_h = (
                agg1.groupby(stmt_col, as_index=False)[horizon_col]
                    .max()
                    .rename(columns={horizon_col: "max_h"})
            )

            # Step 2: aggregate across horizons per statement_month
            agg2 = (
                agg1.groupby(stmt_col, as_index=False)
                    .agg({"act": agg_fn, "pred": agg_fn})
                    .merge(max_h, on=stmt_col)
                    .sort_values(stmt_col)
                    .reset_index(drop=True)
            )

            # Compute MPE (%)
            agg2["mpe"] = np.where(
                agg2["act"] != 0,
                (agg2["pred"] - agg2["act"]) / agg2["act"] * 100,
                float("nan"),
            )

            complete   = agg2["max_h"] >= MAX_HORIZONS
            incomplete = ~complete

            labels = [str(d)[:7] for d in agg2[stmt_col]]
            x      = np.arange(len(labels))
            width  = 0.6

            # Complete months — solid bars
            if complete.any():
                ax.bar(x[complete], agg2.loc[complete, "mpe"], width,
                       color=BAR_COLOR, alpha=0.85, label="Complete (28 horizons)")

            # Incomplete months — hatched bars
            if incomplete.any():
                ax.bar(x[incomplete], agg2.loc[incomplete, "mpe"], width,
                       color=BAR_COLOR, alpha=0.5, hatch="//",
                       label="Incomplete (< 28 horizons)")

            ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")

            # Thin out x-tick labels so they don't overlap
            step = max(1, len(labels) // 12)
            ax.set_xticks(x[::step])
            ax.set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("MPE (%)")
            ax.set_title(
                f"{target_name} MPE by Statement Month\n"
                f"(Portfolio level, {agg_label} across horizons)",
                fontsize=10,
            )
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, axis="y")

        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading {target_name}:\n{e}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="red")
            ax.set_title(target_name)

    fig.suptitle(
        f"Dataset: {dataset_name} — Portfolio MPE Summary by Statement Month",
        fontsize=13, y=1.01,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig


# ── MPE summary for chatbox opening message ───────────────────────────────────
def _compute_mpe_summary(dataset_name: str) -> str:
    """Compute mean MPE and AMPE per target variable across all statement months
    using the same avg-across-horizons logic as the summary charts."""
    if not dataset_name:
        return ""
    try:
        import polars as pl
    except ImportError:
        return ""

    cfg         = _load_config()
    datasets    = cfg.get("datasets", {})
    targets     = cfg.get("target_variables", {})
    time_cols   = cfg.get("time_columns", {})
    stmt_col    = time_cols.get("statement_month", "statement_month")
    horizon_col = time_cols.get("horizon", "horizon")
    perf_col    = time_cols.get("performance_month", "performance_month")
    row_filter  = cfg.get("default_row_filter", "")

    path = datasets.get(dataset_name, "")
    if not path or not os.path.exists(path):
        return f"Dataset '{dataset_name}' not found at: {path}"

    header = (
        f"**Dataset: {dataset_name}** — Portfolio MPE Summary "
        f"(avg across horizons, all statement months)\n\n"
        f"| Target | MPE | AMPE |\n"
        f"|---|---:|---:|"
    )
    rows = []

    for target_name, cols in targets.items():
        actual_col  = cols.get("actual", "")
        pred_col    = cols.get("predicted", "")
        metric_type = cols.get("metric_type", "flow")
        agg_fn = "sum" if metric_type == "flow" else "mean"
        try:
            needed = list({stmt_col, horizon_col, perf_col, actual_col, pred_col})
            lf = pl.scan_parquet(path).select(needed)

            # Apply default row filter from config.yaml
            if row_filter:
                df_raw = lf.collect().to_pandas()
                try:
                    df_raw = df_raw.query(row_filter)
                except Exception:
                    pass
                lf = pl.from_pandas(df_raw).lazy()

            # Step 1: portfolio sum per (statement_month, horizon)
            agg1 = (
                lf.group_by([stmt_col, horizon_col])
                  .agg([pl.col(actual_col).sum().alias("act"),
                        pl.col(pred_col).sum().alias("pred")])
                  .collect()
                  .to_pandas()
            )

            # Step 2: sum (flow) or average (stock) across horizons per stmt_month
            agg2 = (
                agg1.groupby(stmt_col, as_index=False)
                    .agg({"act": agg_fn, "pred": agg_fn})
            )

            # Step 3: MPE per statement month → mean across all months
            mpe = np.where(
                agg2["act"] != 0,
                (agg2["pred"] - agg2["act"]) / agg2["act"] * 100,
                float("nan"),
            )
            mean_mpe  = float(np.nanmean(mpe))
            mean_ampe = float(np.nanmean(np.abs(mpe)))
            rows.append(f"| {target_name} | {mean_mpe:+.2f}% | {mean_ampe:.2f}% |")
        except Exception as e:
            rows.append(f"| {target_name} | Error | {e} |")

    footer = "\n\nThe charts on the left show the full trend by statement month. Send a request below to start your analysis."
    return header + "\n" + "\n".join(rows) + footer


def _load_summary(dataset_name: str) -> tuple:
    """Return (summary_fig, initial_chatbot_history) for a given dataset."""
    fig     = generate_summary_charts(dataset_name)
    msg     = _compute_mpe_summary(dataset_name)
    history = [{"role": "assistant", "content": msg}] if msg else []
    return fig, history


# ── Example prompts ───────────────────────────────────────────────────────────
EXAMPLES = [
    f"Inspect the file at '{ROUND1_PATH}' and summarise its schema.",
    f"Show me the Payment MPE trend by statement month at portfolio level using '{ROUND1_PATH}'.",
    f"Show me EOS actual vs predicted by horizon at portfolio level using '{ROUND1_PATH}'.",
    f"Show me PurchaseVolume AMPE by statement month at account level using '{ROUND1_PATH}'.",
    f"Compare Round 1 ('{ROUND1_PATH}') and Round 2 ('{ROUND2_PATH}') Payment MPE by horizon.",
]



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
    Returns: (chatbot, status, gallery_value, gallery_visible, summary_fig, summary_visible)
    Resets session, restores summary chart and opening MPE message.
    """
    global _session_start_time
    _session_start_time = time.time()
    try:
        agent.memory.reset()
    except Exception:
        pass
    summary_fig, opening_history = _load_summary(dataset_name)
    return (opening_history,                            # chatbot with opening message
            "Ready.",
            [],                                         # gallery value (empty)
            gr.update(visible=False),                   # gallery hidden
            gr.update(value=summary_fig, visible=True)) # summary shown


# ── Build the Gradio UI ───────────────────────────────────────────────────────
_dataset_choices = _get_dataset_choices()
_default_dataset = _dataset_choices[0] if _dataset_choices else None

with gr.Blocks(title="CCAR Backtesting Agent") as demo:

    gr.Markdown("# CCAR Backtesting Analysis Agent")

    # ── Dataset selector ──────────────────────────────────────────────────────
    with gr.Row():
        dataset_dropdown = gr.Dropdown(
            choices=_dataset_choices,
            value=_default_dataset,
            label="Dataset",
            scale=0,
            min_width=160,
        )

    # ── Main layout ───────────────────────────────────────────────────────────
    with gr.Row():

        # Left: summary (shown on load) / gallery (shown after first request)
        with gr.Column(scale=3):

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

        # Right: chatbox
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", height=520)
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

    # Dataset change → regenerate summary + opening message, restore summary view
    def on_dataset_change(dataset_name):
        fig, opening_history = _load_summary(dataset_name)
        return (gr.update(value=fig, visible=True),   # summary shown
                gr.update(visible=False),              # gallery hidden
                gr.update(visible=False),              # refresh btn hidden
                opening_history)                       # chatbot reset with opening message

    dataset_dropdown.change(
        fn=on_dataset_change,
        inputs=[dataset_dropdown],
        outputs=[summary_plot, gallery, refresh_btn, chatbot],
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

    # Auto-load summary + opening message on page open
    demo.load(
        fn=_load_summary,
        inputs=[dataset_dropdown],
        outputs=[summary_plot, chatbot],
    )


if __name__ == "__main__":
    import signal, subprocess, platform

    # Single place to change the port — override with GRADIO_PORT env var if needed
    _PORT = int(os.environ.get("GRADIO_PORT", 7860))

    # Kill any process already using that port
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(["lsof", "-ti", f"tcp:{_PORT}"], capture_output=True, text=True)
            for pid in result.stdout.strip().splitlines():
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except Exception:
                    pass
        else:
            subprocess.run(["fuser", "-k", f"{_PORT}/tcp"], capture_output=True)
    except Exception:
        pass

    # JupyterHub proxy support
    _jupyter_prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "").rstrip("/")
    _root_path = f"{_jupyter_prefix}/proxy/{_PORT}" if _jupyter_prefix else ""

    demo.launch(
        server_port=_PORT,
        root_path=_root_path,
        share=False,
        inbrowser=False,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    )
