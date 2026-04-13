"""
CCAR Backtesting Analysis Agent
================================
Creates a CodeAgent equipped with domain-specific tools for:
  - Loading account-level backtesting datasets
  - Aggregating data across dimensions (product type, vintage, geography, …)
  - Generating visualizations
  - Comparing two CCAR rounds

Usage
-----
    from backtesting_agent.agent import build_agent
    agent = build_agent(provider="openai")   # or "anthropic", "azure", "hf", "groq"
    agent.run("Load round1.csv and show me RMSE by product type.")

Alternatively run from the command line:
    python -m backtesting_agent.agent
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Load .env from the same directory as this file ────────────────────────────
_ENV_FILE = Path(__file__).parent / ".env"
if _ENV_FILE.exists():
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE)

# ── Load column config (config.yaml) ──────────────────────────────────────────
_CONFIG_FILE = Path(__file__).parent / "config.yaml"

def _load_config() -> dict:
    if not _CONFIG_FILE.exists():
        return {}
    import yaml
    with open(_CONFIG_FILE) as f:
        return yaml.safe_load(f) or {}

def _build_column_context(cfg: dict) -> str:
    """Turn config.yaml into a plain-English block injected into system instructions."""
    targets = cfg.get("target_variables", {})
    groups = cfg.get("default_group_by_columns", [])
    output_dir = cfg.get("output_dir", "./backtesting_output")

    if not targets and not groups:
        return ""

    lines = ["\nProject-specific column mappings (from config.yaml)\n" + "-" * 52]
    if targets:
        lines.append("Target variables — use these names when the user mentions a model target:")
        for name, cols in targets.items():
            lines.append(f"  {name}: actual_column='{cols['actual']}', predicted_column='{cols['predicted']}'")
    if groups:
        lines.append(f"\nDefault segmentation columns: {', '.join(groups)}")
    lines.append(f"Default output directory: {output_dir}")
    return "\n".join(lines)

# ── smolagents imports ─────────────────────────────────────────────────────────
from smolagents import CodeAgent, UserInputTool

# ── domain tools ──────────────────────────────────────────────────────────────
from backtesting_agent.tools import (
    # data
    inspect_parquet,
    load_dataset,
    get_dataset_info,
    list_loaded_datasets,
    # aggregation
    aggregate_parquet,
    aggregate_by_statement_month,
    aggregate_by_vintage,
    aggregate_by_feature_bins,
    # metrics & viz
    calculate_backtesting_metrics,
    generate_chart,
    generate_full_report,
    compare_ccar_rounds,
)

TOOLS = [
    # data loading
    inspect_parquet,
    load_dataset,
    get_dataset_info,
    list_loaded_datasets,
    # aggregation (run these BEFORE metrics/viz on large parquet files)
    aggregate_parquet,
    aggregate_by_statement_month,
    aggregate_by_vintage,
    aggregate_by_feature_bins,
    # metrics & visualization
    calculate_backtesting_metrics,
    generate_chart,
    generate_full_report,
    compare_ccar_rounds,
    UserInputTool(),
]

# Libraries the agent is allowed to import inside its generated code
AUTHORIZED_IMPORTS = [
    "pandas", "numpy", "matplotlib", "matplotlib.pyplot",
    "seaborn", "scipy", "scipy.stats", "sklearn",
    "sklearn.metrics", "os", "json", "re", "math",
]

SYSTEM_INSTRUCTIONS = """\
You are a quantitative model-risk analyst specializing in CCAR (Comprehensive
Capital Analysis and Review) model backtesting.

Your job is to help users:
1. Inspect and load account-level backtesting datasets (CSV, Excel, or large Parquet).
2. Aggregate raw account-level data by statement month, vintage, feature bins, or
   any categorical dimension before running metrics or charts.
3. Calculate standard backtesting metrics: RMSE, MAE, Bias, MAPE, R², Gini, KS.
4. Generate publication-quality charts (scatter, bar, line, box, residual, heatmap).
5. Compare two CCAR rounds side-by-side and highlight improvements / regressions.

Workflow for large Parquet files (ALWAYS follow this order)
-----------------------------------------------------------
Step 1 — inspect_parquet        : read the schema without loading the file
Step 2 — aggregate_by_*         : aggregate to a compact summary dataset
          • aggregate_by_statement_month  → time-series view
          • aggregate_by_vintage          → cohort / vintage view
          • aggregate_by_feature_bins     → score-band / LTV-tier view
          • aggregate_parquet             → any other custom group-by
Step 3 — calculate_backtesting_metrics  : run metrics on the aggregated dataset
Step 4 — generate_chart / generate_full_report : visualize

Guidelines
----------
- For parquet input, never call load_dataset directly on a multi-million-row file —
  use inspect_parquet first, then the appropriate aggregate_by_* tool.
- Use generate_full_report for a quick comprehensive view; use generate_chart for
  targeted single charts.
- When comparing two rounds, remind the user to load/aggregate BOTH datasets first.
- Interpret results quantitatively: explain what the metric values mean for model
  quality and regulatory acceptability.
- Save all charts to ./backtesting_output/ unless the user specifies otherwise.

Clarification rules — use the user_input tool to ask before proceeding when:
- The user mentions a target variable (e.g. "PD", "loss") but it is NOT listed in
  the column mappings above and no dataset is loaded yet to inspect.
- The user asks to compare two rounds but only one dataset is loaded.
- The user requests a chart type or segmentation column that cannot be inferred
  from what has been loaded (e.g. "by region" when no region column exists).
- The request is genuinely ambiguous about WHICH target variable to use
  (e.g. "analyze the model" when multiple targets are defined).
Ask ONE focused question at a time. Never ask for information that can be
determined by inspecting the already-loaded dataset.
"""


def _build_model(provider: str):
    """Build the LLM model object for the given provider."""
    provider = provider.lower().strip()

    if provider == "openai":
        from smolagents import LiteLLMModel
        return LiteLLMModel(
            model_id="gpt-4o",
            api_key=os.environ["OPENAI_API_KEY"],
        )

    elif provider == "anthropic":
        from smolagents import LiteLLMModel
        return LiteLLMModel(
            model_id="anthropic/claude-3-5-sonnet-latest",
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )

    elif provider == "azure":
        from smolagents import AzureOpenAIModel
        return AzureOpenAIModel(
            model_id=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("OPENAI_API_VERSION", "2024-02-01"),
        )

    elif provider in ("hf", "huggingface"):
        from smolagents import InferenceClientModel
        return InferenceClientModel(
            model_id="Qwen/Qwen2.5-72B-Instruct",
            token=os.environ.get("HF_TOKEN"),
        )

    elif provider == "groq":
        from smolagents import LiteLLMModel
        return LiteLLMModel(
            model_id="groq/llama-3.3-70b-versatile",
            api_key=os.environ["GROQ_API_KEY"],
        )

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            "Choose from: openai, anthropic, azure, hf, groq"
        )


def build_agent(provider: str = "openai", **agent_kwargs) -> CodeAgent:
    """Build and return the backtesting analysis agent.

    Args:
        provider: LLM provider. One of: openai, anthropic, azure, hf, groq.
        **agent_kwargs: Extra keyword arguments passed to CodeAgent (e.g. max_steps).
    """
    model = _build_model(provider)

    cfg = _load_config()
    column_context = _build_column_context(cfg)
    instructions = SYSTEM_INSTRUCTIONS + column_context

    agent = CodeAgent(
        tools=TOOLS,
        model=model,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        instructions=instructions,
        max_steps=agent_kwargs.pop("max_steps", 25),
        verbosity_level=agent_kwargs.pop("verbosity_level", 1),
        stream_outputs=agent_kwargs.pop("stream_outputs", True),
        **agent_kwargs,
    )
    return agent


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CCAR Backtesting Analysis Agent")
    parser.add_argument(
        "--provider", "-p",
        default="openai",
        choices=["openai", "anthropic", "azure", "hf", "groq"],
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--task", "-t",
        default=None,
        help="Task to run. If omitted, enters interactive mode.",
    )
    args = parser.parse_args()

    agent = build_agent(provider=args.provider)

    if args.task:
        result = agent.run(args.task)
        print("\n=== RESULT ===")
        print(result)
    else:
        # Interactive loop
        print("\nCCAR Backtesting Agent — interactive mode. Type 'quit' to exit.\n")
        while True:
            try:
                task = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if task.lower() in ("quit", "exit", "q"):
                break
            if not task:
                continue
            result = agent.run(task)
            print(f"\nAgent: {result}\n")
