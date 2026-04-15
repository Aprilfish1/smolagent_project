"""
End-to-end example: run the CCAR backtesting agent on synthetic parquet data.

Before running:
  1. Ensure .env contains your OPENAI_API_KEY.
  2. Generate sample parquet files:
       python backtesting_agent/generate_sample_data.py
  3. Run this script:
       python backtesting_agent/run_example.py

Charts will be saved to ./backtesting_output/
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtesting_agent.agent import build_agent

ROUND1 = os.path.join(os.path.dirname(__file__), "sample_data", "ccar_round1.parquet")
ROUND2 = os.path.join(os.path.dirname(__file__), "sample_data", "ccar_round2.parquet")


def main():
    agent = build_agent()

    # ── DEMO 1: Single round — time-series and horizon analysis ──────────────
    print("\n" + "=" * 70)
    print("DEMO 1: Analyze CCAR Round 1 across multiple perspectives")
    print("=" * 70)

    task1 = f"""
Analyze the CCAR Round 1 backtesting data at '{ROUND1}':

1. Inspect the file to confirm the schema.
2. Aggregate by statement_month (use actual_pd/predicted_pd and actual_loss/predicted_loss,
   exposure weighting by exposure column), store as 'r1_stmt'.
3. Aggregate by horizon (statement_month → performance_month),
   targets: actual_pd/predicted_pd, store as 'r1_horizon'.
4. Aggregate by product_type × risk_segment (use aggregate_parquet),
   targets: actual_pd/predicted_pd, store as 'r1_segment'.
5. Calculate metrics (AMPE, MPE, dollar errors at both account and portfolio level)
   for target pair actual_pd:predicted_pd on 'r1_stmt', using n_accounts column.
6. Plot the actual vs predicted PD trend over statement_month from 'r1_stmt'.
7. Plot the actual vs predicted PD trend over horizon_months from 'r1_horizon'
   — this shows whether model accuracy degrades at longer forecast horizons.
8. Generate a bar chart of mean_actual_pd vs mean_predicted_pd by segment
   from 'r1_segment'.
9. Summarize the key findings: which statement months, horizons, and segments
   show the largest errors.
"""
    result1 = agent.run(task1)
    print("\n[AGENT RESULT]\n", result1)

    # ── DEMO 2: Compare two CCAR rounds ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("DEMO 2: Compare CCAR Round 1 vs Round 2")
    print("=" * 70)

    task2 = f"""
Compare CCAR Round 1 ('{ROUND1}') against Round 2 ('{ROUND2}'):

1. For Round 1: aggregate by horizon (actual_pd:predicted_pd), store as 'r1_horizon'.
   For Round 2: aggregate by horizon (actual_pd:predicted_pd), store as 'r2_horizon'.
2. Compare both rounds using compare_ccar_rounds on those two datasets,
   comparing by horizon_months and product_type.
3. Also aggregate both rounds by product_type using aggregate_parquet,
   store as 'r1_prod' and 'r2_prod', then compare.
4. Provide an executive summary: which targets improved round-over-round,
   which horizons regressed, and which product segments need attention.
"""
    result2 = agent.run(task2)
    print("\n[AGENT RESULT]\n", result2)


if __name__ == "__main__":
    main()
