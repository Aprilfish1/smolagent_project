"""
End-to-end example: run the CCAR backtesting agent on synthetic data.

Before running:
  1. Copy .env.example to .env and fill in your API key.
  2. Generate sample data:
       python backtesting_agent/generate_sample_data.py
  3. Run this script:
       python backtesting_agent/run_example.py --provider openai

Charts will be saved to ./backtesting_output/
"""
import argparse
import os
import sys

# Make sure the repo root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtesting_agent.agent import build_agent

ROUND1_PATH = os.path.join(os.path.dirname(__file__), "sample_data", "ccar_round1.csv")
ROUND2_PATH = os.path.join(os.path.dirname(__file__), "sample_data", "ccar_round2.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider", "-p",
        default="openai",
        choices=["openai", "anthropic", "azure", "hf", "groq"],
        help="Which LLM provider to use",
    )
    parser.add_argument(
        "--demo",
        choices=["single", "compare", "all"],
        default="all",
        help="Which demo to run (default: all)",
    )
    args = parser.parse_args()

    agent = build_agent(provider=args.provider)

    # ── DEMO 1: single dataset, multiple perspectives ─────────────────────────
    if args.demo in ("single", "all"):
        print("\n" + "=" * 70)
        print("DEMO 1: Analyze one CCAR round from multiple perspectives")
        print("=" * 70)

        task1 = f"""
Please analyze the CCAR Round 1 backtesting data:

1. Load the file at '{ROUND1_PATH}' and register it as 'round1'.
2. Show me a summary of the dataset (columns, shape, a few rows).
3. Calculate portfolio-level backtesting metrics.
4. Calculate metrics broken out by 'product_type' and separately by 'risk_segment'.
5. Generate a full report (scatter, residual, bar charts by product_type and risk_segment).
6. Summarize which segments have the worst model performance and why.
"""
        result1 = agent.run(task1)
        print("\n[AGENT RESULT]\n", result1)

    # ── DEMO 2: compare two CCAR rounds ──────────────────────────────────────
    if args.demo in ("compare", "all"):
        print("\n" + "=" * 70)
        print("DEMO 2: Compare CCAR Round 1 vs Round 2")
        print("=" * 70)

        task2 = f"""
Please compare the two CCAR backtesting rounds:

1. Load '{ROUND1_PATH}' as 'round1' and '{ROUND2_PATH}' as 'round2'.
2. Compare both rounds at the portfolio level and by 'product_type'.
3. Also compare by 'geography' and 'risk_segment'.
4. Generate side-by-side charts for RMSE and Gini by product_type.
5. Generate a scatter overlay chart.
6. Provide a concise executive summary: what improved, what got worse, and
   which segment needs the most attention heading into the next CCAR cycle.
"""
        result2 = agent.run(task2)
        print("\n[AGENT RESULT]\n", result2)


if __name__ == "__main__":
    main()
