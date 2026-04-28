"""
CCAR Backtesting Agent — Gradio Web UI
=======================================
Two-tab layout:
  Tab 1 — Portfolio Dashboard : auto-computed KPIs, performance summary, MPE trend
  Tab 2 — Analysis Chat       : interactive agent chatbox

Run with:
    python backtesting_agent/app.py
"""
from __future__ import annotations

import os
import sys
import glob
import yaml
import numpy as np
import pandas as pd
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
OUTPUT_DIR  = "./backtesting_output"
CONFIG_FILE = Path(__file__).parent / "config.yaml"

# ── Build agent once at startup ───────────────────────────────────────────────
_PATH_HINT = f"""
Available sample data files (tell the user these paths if they ask):
  Round 1: {ROUND1_PATH}
  Round 2: {ROUND2_PATH}

IMPORTANT: Never call user_input() or input(). If a file path is not
specified by the user, ask them which dataset they want to use.
"""
agent = build_agent(extra_instructions=_PATH_HINT)


# ── Dashboard helpers ─────────────────────────────────────────────────────────

def _scan_path(path: str) -> str:
    """Return glob pattern for partitioned parquet folders, else the path itself."""
    return str(Path(path) / "*.parquet") if Path(path).is_dir() else path


def _dashboard_overview(cfg: dict) -> pd.DataFrame:
    """Fast schema + date-range overview using polars lazy scan."""
    try:
        import polars as pl
    except ImportError:
        return pd.DataFrame([{"Error": "polars not installed"}])

    datasets = cfg.get("datasets", {})
    time_cols = cfg.get("time_columns", {})
    stmt_col  = time_cols.get("statement_month", "statement_month")

    if not datasets:
        return pd.DataFrame([{"Dataset": "—", "Rows": "—", "Columns": "—",
                               "Stmt Month Range": "No datasets in config.yaml"}])

    rows = []
    for name, path in datasets.items():
        if not os.path.exists(path):
            rows.append({"Dataset": name, "Rows": "—", "Columns": "—",
                         "Stmt Month Range": f"NOT FOUND: {path}"})
            continue
        try:
            sp  = _scan_path(path)
            lf  = pl.scan_parquet(sp)
            schema  = lf.collect_schema()
            n_rows  = lf.select(pl.len()).collect().item()
            date_df = lf.select([
                pl.col(stmt_col).min().alias("min_s"),
                pl.col(stmt_col).max().alias("max_s"),
            ]).collect().to_pandas()
            date_range = (
                f"{date_df['min_s'].iloc[0]} → {date_df['max_s'].iloc[0]}"
                if stmt_col in schema.names() else "N/A"
            )
            rows.append({
                "Dataset":         name,
                "Rows":            f"{n_rows:,}",
                "Columns":         len(schema),
                "Stmt Month Range": date_range,
            })
        except Exception as e:
            rows.append({"Dataset": name, "Rows": "ERROR", "Columns": "—",
                         "Stmt Month Range": str(e)[:80]})

    return pd.DataFrame(rows)


def _dashboard_performance(cfg: dict) -> pd.DataFrame:
    """Compute mean MPE / AMPE for each target variable across all statement months."""
    try:
        import polars as pl
    except ImportError:
        return pd.DataFrame([{"Error": "polars not installed"}])

    datasets   = cfg.get("datasets", {})
    targets    = cfg.get("target_variables", {})
    time_cols  = cfg.get("time_columns", {})
    row_filter = cfg.get("default_row_filter", "")
    stmt_col   = time_cols.get("statement_month", "statement_month")
    horizon_col = time_cols.get("horizon", "horizon")
    perf_col   = time_cols.get("performance_month", "performance_month")

    if not datasets or not targets:
        return pd.DataFrame([{"Info": "No datasets or targets configured in config.yaml"}])

    primary_path = next(iter(datasets.values()))
    if not os.path.exists(primary_path):
        return pd.DataFrame([{"Error": f"Primary dataset not found: {primary_path}"}])

    sp = _scan_path(primary_path)
    results = []

    for target_name, cols in targets.items():
        actual_col = cols.get("actual", "")
        pred_col   = cols.get("predicted", "")
        mtype      = cols.get("metric_type", "flow")

        try:
            needed = list({actual_col, pred_col, stmt_col, horizon_col, perf_col})
            lf = pl.scan_parquet(sp).select(needed)

            # Apply default row filter if present
            if row_filter.strip():
                df_pd = lf.collect().to_pandas()
                df_pd = df_pd.query(row_filter)
                lf = pl.from_pandas(df_pd).lazy()

            # EOS: keep only max horizon per statement_month
            if mtype == "stock":
                max_h = lf.group_by(stmt_col).agg(
                    pl.col(horizon_col).max().alias("_max_h"))
                lf = (lf.join(max_h, on=stmt_col, how="left")
                        .filter(pl.col(horizon_col) == pl.col("_max_h"))
                        .drop("_max_h"))

            # Aggregate by statement_month
            lf2 = lf.with_columns(
                pl.col(stmt_col).cast(pl.Date).dt.truncate("1mo").alias("_grp"))
            agg = (lf2.group_by("_grp")
                       .agg([pl.col(actual_col).sum().alias("act"),
                             pl.col(pred_col).sum().alias("pred")])
                       .sort("_grp")
                       .collect()
                       .to_pandas())

            agg["MPE"] = np.where(
                agg["act"] != 0,
                (agg["pred"] - agg["act"]) / agg["act"],
                float("nan"))
            agg["AMPE"] = agg["MPE"].abs()

            results.append({
                "Target":     target_name,
                "Type":       mtype,
                "Stmt Months": len(agg),
                "Mean MPE":   f"{agg['MPE'].mean() * 100:.2f}%",
                "Mean AMPE":  f"{agg['AMPE'].mean() * 100:.2f}%",
                "Min MPE":    f"{agg['MPE'].min() * 100:.2f}%",
                "Max MPE":    f"{agg['MPE'].max() * 100:.2f}%",
            })

        except Exception as e:
            results.append({
                "Target":      target_name,
                "Type":        mtype,
                "Stmt Months": "—",
                "Mean MPE":    f"ERROR: {str(e)[:60]}",
                "Mean AMPE":   "—",
                "Min MPE":     "—",
                "Max MPE":     "—",
            })

    return pd.DataFrame(results) if results else pd.DataFrame(
        [{"Info": "No results computed"}])


def _dashboard_trend_chart(cfg: dict):
    """MPE trend over statement months for all targets — returns a matplotlib Figure."""
    try:
        import polars as pl
    except ImportError:
        return None

    datasets    = cfg.get("datasets", {})
    targets     = cfg.get("target_variables", {})
    time_cols   = cfg.get("time_columns", {})
    row_filter  = cfg.get("default_row_filter", "")
    stmt_col    = time_cols.get("statement_month", "statement_month")
    horizon_col = time_cols.get("horizon", "horizon")
    perf_col    = time_cols.get("performance_month", "performance_month")

    if not datasets or not targets:
        return None

    primary_path = next(iter(datasets.values()))
    if not os.path.exists(primary_path):
        return None

    sp  = _scan_path(primary_path)
    fig, ax = plt.subplots(figsize=(13, 4))

    for target_name, cols in targets.items():
        actual_col = cols.get("actual", "")
        pred_col   = cols.get("predicted", "")
        mtype      = cols.get("metric_type", "flow")

        try:
            needed = list({actual_col, pred_col, stmt_col, horizon_col, perf_col})
            lf = pl.scan_parquet(sp).select(needed)

            if row_filter.strip():
                df_pd = lf.collect().to_pandas()
                df_pd = df_pd.query(row_filter)
                lf = pl.from_pandas(df_pd).lazy()

            if mtype == "stock":
                max_h = lf.group_by(stmt_col).agg(
                    pl.col(horizon_col).max().alias("_max_h"))
                lf = (lf.join(max_h, on=stmt_col, how="left")
                        .filter(pl.col(horizon_col) == pl.col("_max_h"))
                        .drop("_max_h"))

            lf2 = lf.with_columns(
                pl.col(stmt_col).cast(pl.Date).dt.truncate("1mo").alias("_grp"))
            agg = (lf2.group_by("_grp")
                       .agg([pl.col(actual_col).sum().alias("act"),
                             pl.col(pred_col).sum().alias("pred")])
                       .sort("_grp")
                       .collect()
                       .to_pandas())

            mpe_pct = np.where(
                agg["act"] != 0,
                (agg["pred"] - agg["act"]) / agg["act"] * 100,
                float("nan"))

            ax.plot(agg["_grp"], mpe_pct, marker="o", markersize=3,
                    linewidth=1.5, label=target_name)

        except Exception:
            continue

    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Statement Month")
    ax.set_ylabel("MPE (%)")
    ax.set_title("Portfolio-Level MPE by Statement Month (all targets)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Thin out crowded x-axis labels
    fig.canvas.draw()
    labels = ax.get_xticklabels()
    n = len(labels)
    if n > 18:
        step = max(1, n // 18)
        for i, lbl in enumerate(labels):
            lbl.set_visible(i % step == 0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    return fig


def load_dashboard():
    """Called when the Load/Refresh button is clicked on the Dashboard tab."""
    cfg  = _load_config()
    ov   = _dashboard_overview(cfg)
    perf = _dashboard_performance(cfg)
    fig  = _dashboard_trend_chart(cfg)
    status = "✅ Dashboard loaded."
    return ov, perf, fig, status


# ── Example prompts ───────────────────────────────────────────────────────────
EXAMPLES = [
    f"Inspect the file at '{ROUND1_PATH}' and summarise its schema.",
    f"Aggregate '{ROUND1_PATH}' by statement_month (actual_pd vs predicted_pd), "
    f"store as 'r1_stmt'. Calculate metrics at portfolio and account level.",
    f"Aggregate '{ROUND1_PATH}' by horizon (actual_pd vs predicted_pd), store as "
    f"'r1_horizon'. Plot the trend of mean_actual_pd vs mean_predicted_pd over horizon_months.",
    f"Aggregate '{ROUND1_PATH}' by product_type and risk_segment (actual_pd vs predicted_pd), "
    f"store as 'r1_seg'. Generate a bar chart of mean_actual_pd by product_type.",
    f"Compare Round 1 ('{ROUND1_PATH}') and Round 2 ('{ROUND2_PATH}'): aggregate both by "
    f"horizon (actual_pd vs predicted_pd), then compare them side by side.",
]

# ── Ambiguity pre-flight check ────────────────────────────────────────────────
_CLARIFY_SYSTEM = """You are a request parser for a backtesting analysis tool.
Decide if the user's request is missing critical information needed to proceed.

The available target variables are: Payment, PurchaseVolume, EOS.

Critical missing information means ONE of the following:
1. No dataset/file path is provided and none can be inferred from context.
2. The target variable is ambiguous — the user has not said which one to use
   (Payment, PurchaseVolume, or EOS) and the request does not make it obvious.
   This applies equally to MPE and AMPE requests: "plot MPE" without naming the
   target variable is ambiguous and MUST be clarified.
3. The analysis level is not specified (portfolio or account?) and it changes the output.
4. A plot of raw values is requested and it is unclear whether to show actual,
   predicted, or both — BUT skip this question if the user is asking for MPE or AMPE
   (those are computed automatically and do not require a choice of actual vs predicted).

IMPORTANT DISTINCTIONS:
- "Which target variable?" (Payment / PurchaseVolume / EOS) → ALWAYS ask if missing.
- "Actual or predicted?" → NEVER ask when the user requests MPE or AMPE.

If any critical information is missing, reply with ONLY a short clarifying question
(1-2 sentences, ask only the single most important missing piece).
If the request is clear enough to proceed, reply with exactly: PROCEED
Do not explain your reasoning."""


def _check_ambiguity(user_message: str, history: list) -> str | None:
    try:
        import litellm
        api_key = os.environ.get("OPENAI_API_KEY", "")
        model   = os.environ.get("OPENAI_MODEL", "gpt-4o")
        messages = [{"role": "system", "content": _CLARIFY_SYSTEM}]
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})
        resp = litellm.completion(
            model=model, messages=messages, api_key=api_key,
            max_tokens=120, temperature=0)
        answer = resp.choices[0].message.content.strip()
        return None if answer.upper().startswith("PROCEED") else answer
    except Exception:
        return None


def _collect_charts() -> list[str]:
    pngs = glob.glob(os.path.join(OUTPUT_DIR, "**", "*.png"), recursive=True)
    return sorted(pngs, key=os.path.getmtime, reverse=True)


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
    lines.append(
        "\nComplete the user's latest request using all context above. "
        "Do not ask again for clarifications already answered in the conversation. "
        "However, you MUST still follow the MANDATORY WORKFLOW: present the full "
        "execution plan and ask 'Shall I proceed? (yes/no)' before calling any tool, "
        "even if all information is already clear from context."
    )
    return "\n".join(lines)


def run_agent(user_message: str, history: list) -> tuple:
    if not user_message.strip():
        return history, "Enter a request above.", _collect_charts()

    history = history + [{"role": "user", "content": user_message}]

    clarifying_question = _check_ambiguity(user_message, history[:-1])
    if clarifying_question:
        history = history + [{"role": "assistant", "content": clarifying_question}]
        return history, "Waiting for clarification…", _collect_charts()

    charts_before = set(_collect_charts())
    task = _build_task(user_message, history)
    try:
        result = agent.run(task)
    except Exception as e:
        result = f"ERROR: {e}"

    response = str(result)
    charts_after = _collect_charts()
    new_charts = [c for c in charts_after if c not in charts_before]
    if new_charts:
        new_names = ", ".join(os.path.basename(c) for c in new_charts)
        response += f"\n\n📊 New chart(s) generated: {new_names}"

    history = history + [{"role": "assistant", "content": response}]
    status = (
        f"✅ Done — {len(new_charts)} new chart(s) generated."
        if new_charts else "✅ Done — no new charts."
    )
    return history, status, charts_after


def clear_all():
    """Reset Gradio UI and agent memory so the next request starts fully fresh."""
    try:
        agent.memory.reset()
    except Exception:
        pass
    return [], "Ready.", []


# ── Build the Gradio UI ───────────────────────────────────────────────────────
with gr.Blocks(title="CCAR Backtesting Agent",
               theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:

    gr.Markdown("# CCAR Backtesting Analysis Agent")

    with gr.Tabs():

        # ── Tab 1: Portfolio Dashboard ────────────────────────────────────────
        with gr.Tab("📊 Portfolio Dashboard"):
            gr.Markdown(
                "Overview of your configured datasets and portfolio-level "
                "backtesting performance. Click **Load Dashboard** to compute."
            )

            dash_status = gr.Textbox(
                label="Status", value="Click 'Load Dashboard' to begin.",
                interactive=False, lines=1)

            load_btn = gr.Button("🔄 Load / Refresh Dashboard", variant="primary")

            gr.Markdown("### Dataset Overview")
            overview_table = gr.Dataframe(
                label="Datasets",
                interactive=False,
                wrap=True,
            )

            gr.Markdown("### Performance Summary by Target Variable")
            gr.Markdown(
                "*Portfolio-level MPE / AMPE aggregated across all statement months. "
                "Values shown as percentages.*"
            )
            perf_table = gr.Dataframe(
                label="Performance Statistics",
                interactive=False,
                wrap=True,
            )

            gr.Markdown("### MPE Trend by Statement Month")
            trend_chart = gr.Plot(label="MPE Trend")

            load_btn.click(
                fn=load_dashboard,
                outputs=[overview_table, perf_table, trend_chart, dash_status],
            )

        # ── Tab 2: Analysis Chat ──────────────────────────────────────────────
        with gr.Tab("💬 Analysis Chat"):
            gr.Markdown(
                "Ask questions about your backtesting data in plain English. "
                "The agent will aggregate your parquet file, compute metrics, "
                "and generate charts."
            )

            with gr.Row():
                # Left: chat
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="Conversation", height=520)

                    with gr.Row():
                        msg_box = gr.Textbox(
                            placeholder="e.g. Aggregate by horizon and show PD trend…",
                            label="Your request",
                            scale=5,
                            lines=2,
                            autofocus=True,
                        )
                        with gr.Column(scale=1, min_width=120):
                            submit_btn = gr.Button("▶ Run", variant="primary")
                            clear_btn  = gr.Button("🗑 Clear")

                    status_box = gr.Textbox(
                        label="Status", interactive=False, lines=1)

                    gr.Markdown("### Quick-start examples")
                    for ex in EXAMPLES:
                        gr.Button(ex[:90] + "…", size="sm").click(
                            fn=lambda e=ex: e, outputs=msg_box)

                # Right: chart gallery
                with gr.Column(scale=2):
                    gr.Markdown("### Generated Charts")
                    gallery = gr.Gallery(
                        label="Charts (newest first)",
                        columns=2,
                        height=560,
                        object_fit="contain",
                        show_label=False,
                    )
                    refresh_btn = gr.Button("🔄 Refresh Gallery", size="sm")

            # ── File info accordion ───────────────────────────────────────────
            with gr.Accordion("📁 Sample data file paths", open=False):
                gr.Markdown(
                    f"""
Copy these paths into your requests:

| Round | Path |
|---|---|
| CCAR Round 1 | `{ROUND1_PATH}` |
| CCAR Round 2 | `{ROUND2_PATH}` |

Generate them with:
```bash
python backtesting_agent/generate_sample_data.py
```
Output charts are saved to: `{os.path.abspath(OUTPUT_DIR)}`
"""
                )

            # ── Event wiring ──────────────────────────────────────────────────
            submit_btn.click(
                fn=run_agent,
                inputs=[msg_box, chatbot],
                outputs=[chatbot, status_box, gallery],
            ).then(fn=lambda: "", outputs=msg_box)

            msg_box.submit(
                fn=run_agent,
                inputs=[msg_box, chatbot],
                outputs=[chatbot, status_box, gallery],
            ).then(fn=lambda: "", outputs=msg_box)

            clear_btn.click(fn=clear_all, outputs=[chatbot, status_box, gallery])
            refresh_btn.click(fn=_collect_charts, outputs=gallery)


if __name__ == "__main__":
    import signal, subprocess
    result = subprocess.run(["lsof", "-ti", "tcp:7860"], capture_output=True, text=True)
    for pid in result.stdout.strip().splitlines():
        try:
            os.kill(int(pid), signal.SIGKILL)
        except Exception:
            pass

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
