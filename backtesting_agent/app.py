"""
CCAR Backtesting Agent — Gradio Web UI
=======================================
Run with:
    python backtesting_agent/app.py

Then open http://localhost:7860 in your browser.
"""
from __future__ import annotations

import os
import sys
import glob
from pathlib import Path

_repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / "src"))   # editable install: smolagents lives in src/

import gradio as gr
from backtesting_agent.agent import build_agent

# ── Paths ─────────────────────────────────────────────────────────────────────
SAMPLE_DIR  = Path(__file__).parent / "sample_data"
ROUND1_PATH = str(SAMPLE_DIR / "ccar_round1.parquet")
ROUND2_PATH = str(SAMPLE_DIR / "ccar_round2.parquet")
OUTPUT_DIR  = "./backtesting_output"

# ── Build agent once at startup ───────────────────────────────────────────────
_PATH_HINT = f"""
Available sample data files (tell the user these paths if they ask):
  Round 1: {ROUND1_PATH}
  Round 2: {ROUND2_PATH}

IMPORTANT: Never call user_input() or input(). If a file path is not
specified by the user, ask them which dataset they want to use.
"""
agent = build_agent(extra_instructions=_PATH_HINT)

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
    """
    Returns a clarifying question string if the request is ambiguous,
    or None if the agent should proceed.
    """
    try:
        import litellm
        api_key = os.environ.get("OPENAI_API_KEY", "")
        model   = os.environ.get("OPENAI_MODEL", "gpt-4o")

        # Build a short conversation context for the check
        messages = [{"role": "system", "content": _CLARIFY_SYSTEM}]
        # Include last 2 turns of history for context
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})

        resp = litellm.completion(
            model=model,
            messages=messages,
            api_key=api_key,
            max_tokens=120,
            temperature=0,
        )
        answer = resp.choices[0].message.content.strip()
        if answer.upper().startswith("PROCEED"):
            return None
        return answer
    except Exception:
        # If the check itself fails, let the agent proceed
        return None


def _collect_charts() -> list[str]:
    """Return all PNG paths in the output directory, sorted newest-first."""
    pngs = glob.glob(os.path.join(OUTPUT_DIR, "**", "*.png"), recursive=True)
    return sorted(pngs, key=os.path.getmtime, reverse=True)


def _build_task(user_message: str, history: list) -> str:
    """
    Reconstruct a complete task string for the agent by prepending relevant
    conversation history.  This ensures that short follow-up answers like
    "PurchaseVolume" or "portfolio" carry the full context of the original request.

    We include at most the last 3 prior exchanges (6 messages) to avoid token bloat.
    """
    # history already includes the current user message at the end
    prior = history[:-1]  # everything before the current turn
    if not prior:
        return user_message

    # Keep last 6 messages (3 exchanges) of prior history
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
    """
    Called each time the user submits a message.
    Returns (updated_history, status_text, chart_list).
    """
    if not user_message.strip():
        return history, "Enter a request above.", _collect_charts()

    # Append user turn (Gradio 6 messages format)
    history = history + [{"role": "user", "content": user_message}]

    # ── Pre-flight: check for ambiguity before running the full agent ──────────
    clarifying_question = _check_ambiguity(user_message, history[:-1])
    if clarifying_question:
        history = history + [{"role": "assistant", "content": clarifying_question}]
        return history, "Waiting for clarification…", _collect_charts()

    # ── Run the full agent with conversation context ───────────────────────────
    charts_before = set(_collect_charts())
    task = _build_task(user_message, history)
    try:
        result = agent.run(task)
    except Exception as e:
        result = f"ERROR: {e}"

    response = str(result)

    # Detect charts that were actually newly created during this agent run
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
with gr.Blocks(title="CCAR Backtesting Agent") as demo:

    gr.Markdown(
        """
        # CCAR Backtesting Analysis Agent
        Ask questions about your backtesting data in plain English.
        The agent will aggregate your parquet file, compute metrics, and generate charts.
        """
    )

    with gr.Row():
        # ── Left column: chat ────────────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=520,
            )

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

            status_box = gr.Textbox(label="Status", interactive=False, lines=1)

            gr.Markdown("### Quick-start examples")
            for ex in EXAMPLES:
                gr.Button(ex[:90] + "…", size="sm").click(
                    fn=lambda e=ex: e,
                    outputs=msg_box,
                )

        # ── Right column: chart gallery ──────────────────────────────────────
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

    # ── File info accordion ──────────────────────────────────────────────────
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

    # ── Event wiring ─────────────────────────────────────────────────────────
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
    # Free port 7860 if another instance is still running
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
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    )
