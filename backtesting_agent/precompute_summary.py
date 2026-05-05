# %% [markdown]
# # CCAR Backtesting — Pre-compute Summary Metrics
#
# Run this script once (or whenever a dataset changes) to generate a small
# per-statement-month summary CSV used by the Gradio UI for fast dashboard loading.
#
# Usage (from repo root):
#     python backtesting_agent/precompute_summary.py
#
# Config structure expected in config.yaml:
#     datasets:
#       <MarketName>:
#         <SegmentName>: path/to/parquet
#
# Output (one CSV per Market+Segment pair):
#     backtesting_agent/data_input/{Market}_{Segment}_summary.csv
#   (spaces in Market/Segment names are replaced with underscores)
#
# CSV columns per statement month:
#   statement_month, max_horizon,
#   {Target}_mpe, {Target}_ampe   — for every target in config
#   {Target}_pred_negative         — for stock targets (e.g. EOS)

# %%
import yaml
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

# ── Locate config relative to this file ──────────────────────────────────────
_HERE       = Path(__file__).parent
CONFIG_PATH = _HERE / "config.yaml"
OUTPUT_DIR  = _HERE / "data_input"
OUTPUT_DIR.mkdir(exist_ok=True)

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

datasets    = cfg.get("datasets", {})
targets     = cfg.get("target_variables", {})
time_cols   = cfg.get("time_columns", {})
stmt_col    = time_cols.get("statement_month", "statement_month")
horizon_col = "horizon"   # always present in parquet as a column
perf_col    = time_cols.get("performance_month", "performance_month")
row_filter  = cfg.get("default_row_filter", "")


def _csv_key(market: str, segment: str) -> str:
    """Return the filesystem-safe key used for the summary CSV filename."""
    sanitize = lambda s: s.strip().replace(" ", "_")
    return f"{sanitize(market)}_{sanitize(segment)}"


print("=" * 60)
print("CCAR Backtesting — Pre-compute Summary Metrics")
print("=" * 60)
print(f"Targets    : {list(targets.keys())}")
print(f"Row filter : {row_filter or '(none)'}")
print(f"Output dir : {OUTPUT_DIR}")
print()

# Enumerate all (market, segment, path) triples from nested config
dataset_triples: list[tuple[str, str, str]] = []
for market, segments in datasets.items():
    if isinstance(segments, dict):
        for segment, parquet_path in segments.items():
            dataset_triples.append((market, segment, parquet_path))
    else:
        # Flat entry (legacy): treat market name as both market and segment
        dataset_triples.append((market, market, segments))

print(f"Datasets   : {[(m, s) for m, s, _ in dataset_triples]}")
print()


# %%
def compute_summary(label: str, parquet_path: str) -> pd.DataFrame:
    """
    For each statement month compute:
      - max_horizon             (detect completeness; 28 = full)
      - {Target}_mpe            MPE in % using flow (sum) or stock (avg) logic
      - {Target}_ampe           AMPE in %
      - {Target}_pred_negative  True if portfolio predicted value < 0 (stock only)
    """
    print(f"Processing '{label}' → {parquet_path} ...")

    # Collect all needed columns up front
    all_act  = [v["actual"]    for v in targets.values()]
    all_pred = [v["predicted"] for v in targets.values()]
    needed   = list({stmt_col, horizon_col, perf_col} | set(all_act) | set(all_pred))

    lf = pl.scan_parquet(parquet_path).select(needed)

    # Apply default row filter
    if row_filter:
        df_raw = lf.collect().to_pandas()
        try:
            df_raw = df_raw.query(row_filter)
            print(f"  Filter applied — {len(df_raw):,} rows remain.")
        except Exception as e:
            print(f"  WARNING: filter failed ({e}), using unfiltered data.")
        lf = pl.from_pandas(df_raw).lazy()

    # Max horizon per statement_month
    max_h_df = (
        lf.group_by(stmt_col)
          .agg(pl.col(horizon_col).max().alias("max_horizon"))
          .collect()
          .to_pandas()
    )
    result = max_h_df.copy()

    for target_name, tcols in targets.items():
        act_col  = tcols["actual"]
        pred_col = tcols["predicted"]
        mtype    = tcols.get("metric_type", "flow")
        agg_fn   = "sum" if mtype == "flow" else "mean"

        # Step 1: portfolio sum per (statement_month, horizon)
        agg1 = (
            lf.group_by([stmt_col, horizon_col])
              .agg([pl.col(act_col).sum().alias("act"),
                    pl.col(pred_col).sum().alias("pred")])
              .collect()
              .to_pandas()
        )

        # Step 2: sum (flow) or average (stock) across horizons per stmt_month
        agg2 = (
            agg1.groupby(stmt_col, as_index=False)
                .agg({"act": agg_fn, "pred": agg_fn})
        )

        agg2[f"{target_name}_mpe"] = np.where(
            agg2["act"] != 0,
            (agg2["pred"] - agg2["act"]) / agg2["act"] * 100,
            float("nan"),
        )
        agg2[f"{target_name}_ampe"] = agg2[f"{target_name}_mpe"].abs()

        keep = [stmt_col, f"{target_name}_mpe", f"{target_name}_ampe"]

        # For stock targets: flag months where portfolio predicted value < 0
        if mtype == "stock":
            agg2[f"{target_name}_pred_negative"] = agg2["pred"] < 0
            keep.append(f"{target_name}_pred_negative")

        result = result.merge(agg2[keep], on=stmt_col, how="left")

    result = result.sort_values(stmt_col).reset_index(drop=True)
    return result


# %%
for market, segment, parquet_path in dataset_triples:
    label = f"{market} / {segment}"
    key   = _csv_key(market, segment)
    try:
        df       = compute_summary(label, parquet_path)
        out_path = OUTPUT_DIR / f"{key}_summary.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved → {out_path}  ({len(df)} rows)")
        print(f"  Columns: {list(df.columns)}")
        print(df.head(3).to_string(index=False))
        print()
    except Exception as e:
        print(f"  ERROR processing '{label}': {e}")
        print()

print("Done!")
