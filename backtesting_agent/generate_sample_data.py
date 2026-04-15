"""
Generate synthetic credit card balance sheet backtesting data.

Schema — one row per (account × statement_month × horizon):

  Identifiers
  -----------
  account_id          : unique account identifier (e.g. ACC000001)
  statement_month     : month the forecast was made
  performance_month   : calendar month the outcome was observed
  horizon             : performance_month − statement_month in months (1–28)

  Feature columns  (static per account — same value across all horizons
                    for a given account and statement_month)
  -----------------------------------------------------------------------
  product_type        : Classic / Gold / Platinum / Business
  risk_segment        : Prime / Near Prime / Subprime
  geography           : Northeast / Southeast / Midwest / West
  credit_score        : FICO score (580–850)
  credit_limit        : credit line in USD
  months_on_book      : account age in months at statement_month
  origination_date    : date the account was opened

  Target variables  (vary by horizon — actual outcomes vs model predictions)
  --------------------------------------------------------------------------
  actual_payment           / predicted_payment        : monthly payment amount (flow)
  actual_purchase_volume   / predicted_purchase_volume: monthly purchase spend (flow)
  actual_eos               / predicted_eos            : end-of-statement balance (stock)

Domain rules reflected in the data
-----------------------------------
- Payment and Purchase Volume are FLOW metrics: each horizon contributes
  an independent monthly amount.
- EOS is a STOCK metric: the balance at horizon h is the running balance
  from the previous horizon  (initial_eos + cumulative purchases − cumulative payments).
- Feature columns are identical for all 28 horizons within a
  (account, statement_month) pair.
- Round 1 has a larger prediction noise / systematic bias than Round 2.
"""

import os
import numpy as np
import polars as pl
from datetime import date

# ── Parameters ────────────────────────────────────────────────────────────────
N_ACCOUNTS    = 2_000   # unique accounts
N_STMT_MONTHS = 12      # Jan–Dec 2023
N_HORIZONS    = 28      # performance months per statement month
FIRST_STMT    = date(2023, 1, 1)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sample_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Credit card product parameters ───────────────────────────────────────────
# Monthly purchase rate = fraction of credit limit spent per month
PRODUCT_CONFIG = {
    #              purchase_rate  payment_rate  credit_limit_range
    "Classic":    (0.04,          0.28,          (1_000,  5_000)),
    "Gold":       (0.06,          0.33,          (3_000,  10_000)),
    "Platinum":   (0.09,          0.42,          (7_000,  20_000)),
    "Business":   (0.12,          0.35,          (5_000,  25_000)),
}
PRODUCT_NAMES = list(PRODUCT_CONFIG.keys())
PRODUCT_PROBS = [0.40, 0.30, 0.20, 0.10]

RISK_PAYMENT_MOD = {"Prime": 1.25, "Near Prime": 1.00, "Subprime": 0.60}
RISK_PROBS       = [0.50, 0.30, 0.20]
RISK_NAMES       = ["Prime", "Near Prime", "Subprime"]

GEO_NAMES  = ["Northeast", "Southeast", "Midwest", "West"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _add_months(d: date, months: int) -> date:
    m = d.month - 1 + months
    return date(d.year + m // 12, m % 12 + 1, 1)


def _generate_accounts(rng: np.random.Generator) -> dict:
    """
    Generate static account-level features for N_ACCOUNTS accounts.
    These stay constant across all statement months and horizons.
    """
    products      = rng.choice(PRODUCT_NAMES, size=N_ACCOUNTS, p=PRODUCT_PROBS)
    risk_segments = rng.choice(RISK_NAMES,    size=N_ACCOUNTS, p=RISK_PROBS)
    geographies   = rng.choice(GEO_NAMES,     size=N_ACCOUNTS)
    credit_scores = rng.integers(580, 851, size=N_ACCOUNTS)

    # Credit limit drawn from product-specific range
    credit_limits = np.array([
        rng.integers(*PRODUCT_CONFIG[p][2])
        for p in products
    ], dtype=float)

    # Months on book: 6–144 months (6 months to 12 years)
    mob = rng.integers(6, 145, size=N_ACCOUNTS)

    # Origination date = FIRST_STMT_MONTH − months_on_book
    orig_dates = [_add_months(FIRST_STMT, -int(m)) for m in mob]

    # Per-account behavioral rates (drawn once, fixed across all time)
    purchase_rates = np.array([PRODUCT_CONFIG[p][0] for p in products])
    base_pay_rates = np.array([PRODUCT_CONFIG[p][1] for p in products])
    risk_mods      = np.array([RISK_PAYMENT_MOD[r] for r in risk_segments])
    payment_rates  = np.clip(base_pay_rates * risk_mods, 0.10, 0.95)

    # Add account-level heterogeneity
    purchase_rates = np.clip(purchase_rates * rng.lognormal(0, 0.20, N_ACCOUNTS), 0.01, 0.40)
    payment_rates  = np.clip(payment_rates  * rng.lognormal(0, 0.15, N_ACCOUNTS), 0.05, 0.95)

    # Initial EOS at start of first statement month (20–70% utilization)
    init_utilization = rng.uniform(0.20, 0.70, N_ACCOUNTS)
    init_eos         = credit_limits * init_utilization

    return dict(
        account_id     = [f"ACC{i:06d}" for i in range(N_ACCOUNTS)],
        product_type   = products,
        risk_segment   = risk_segments,
        geography      = geographies,
        credit_score   = credit_scores,
        credit_limit   = credit_limits,
        months_on_book = mob,
        origination_date = orig_dates,
        purchase_rates = purchase_rates,   # internal use only
        payment_rates  = payment_rates,    # internal use only
        init_eos       = init_eos,         # internal use only
    )


def _generate_round(
    accounts: dict,
    rng: np.random.Generator,
    model_pay_bias: float   = 0.0,
    model_purch_bias: float = 0.0,
    model_eos_bias: float   = 0.0,
    noise_scale: float      = 0.15,
) -> pl.DataFrame:
    """
    Build one full round of backtesting data.

    Returns a Polars DataFrame with
      N_ACCOUNTS × N_STMT_MONTHS × N_HORIZONS rows.
    """
    n = N_ACCOUNTS
    purchase_rates = accounts["purchase_rates"]
    payment_rates  = accounts["payment_rates"]
    credit_limits  = accounts["credit_limit"]

    # Pre-allocate flat output arrays
    total = n * N_STMT_MONTHS * N_HORIZONS
    out_account_id     = []
    out_stmt_month     = []
    out_perf_month     = []
    out_horizon        = np.empty(total, dtype=np.int32)

    # Feature columns (tiled — same value for all horizons per account×stmt_month)
    out_product_type   = []
    out_risk_segment   = []
    out_geography      = []
    out_credit_score   = np.empty(total, dtype=np.int32)
    out_credit_limit   = np.empty(total, dtype=np.float64)
    out_mob            = np.empty(total, dtype=np.int32)
    out_orig_date      = []

    # Target variables
    out_act_pay   = np.empty(total, dtype=np.float64)
    out_pred_pay  = np.empty(total, dtype=np.float64)
    out_act_purch = np.empty(total, dtype=np.float64)
    out_pred_purch= np.empty(total, dtype=np.float64)
    out_act_eos   = np.empty(total, dtype=np.float64)
    out_pred_eos  = np.empty(total, dtype=np.float64)

    idx = 0

    for s in range(N_STMT_MONTHS):
        stmt_month = _add_months(FIRST_STMT, s)

        # EOS at start of this statement month drifts slightly over time
        drift = 1.0 + s * 0.004 + rng.normal(0, 0.01, n)
        eos   = accounts["init_eos"] * np.maximum(drift, 0.5)

        for h in range(1, N_HORIZONS + 1):
            perf_month = _add_months(stmt_month, h)
            slc = slice(idx, idx + n)

            # ── Actual purchase volume (flow) ─────────────────────────────────
            # Seasonal factor: spending peaks in Nov/Dec
            seasonal = 1.0 + 0.20 * np.sin(2 * np.pi * (perf_month.month - 6) / 12)
            act_purch = (
                credit_limits * purchase_rates * seasonal
                * rng.lognormal(0, 0.15, n)
            ).clip(0)

            # ── Actual payment (flow) ─────────────────────────────────────────
            act_pay = (
                eos * payment_rates
                * rng.lognormal(0, 0.18, n)
            ).clip(0)
            act_pay = np.minimum(act_pay, eos + act_purch)  # can't overpay

            # ── Actual EOS (stock) ────────────────────────────────────────────
            act_eos = np.maximum(eos + act_purch - act_pay, 0)

            # ── Predicted values (actual × exp(bias + noise)) ─────────────────
            pred_purch = (act_purch * np.exp(
                rng.normal(model_purch_bias, noise_scale, n)
            )).clip(0)

            pred_pay = (act_pay * np.exp(
                rng.normal(model_pay_bias, noise_scale * 0.9, n)
            )).clip(0)

            pred_eos = np.maximum(eos + pred_purch - pred_pay, 0)
            # Add a small direct EOS bias on top
            pred_eos = (pred_eos * np.exp(
                rng.normal(model_eos_bias, noise_scale * 0.5, n)
            )).clip(0)

            # ── Write to output arrays ────────────────────────────────────────
            out_account_id .extend(accounts["account_id"])
            out_stmt_month .extend([stmt_month]  * n)
            out_perf_month .extend([perf_month]  * n)
            out_horizon[slc]      = h
            out_product_type .extend(accounts["product_type"])
            out_risk_segment .extend(accounts["risk_segment"])
            out_geography    .extend(accounts["geography"])
            out_credit_score[slc] = accounts["credit_score"]
            out_credit_limit[slc] = accounts["credit_limit"]
            out_mob[slc]          = accounts["months_on_book"]
            out_orig_date    .extend(accounts["origination_date"])

            out_act_purch[slc]  = act_purch.round(2)
            out_pred_purch[slc] = pred_purch.round(2)
            out_act_pay[slc]    = act_pay.round(2)
            out_pred_pay[slc]   = pred_pay.round(2)
            out_act_eos[slc]    = act_eos.round(2)
            out_pred_eos[slc]   = pred_eos.round(2)

            eos = act_eos   # carry forward actual EOS as base for next horizon
            idx += n

    return pl.DataFrame({
        "account_id"            : out_account_id,
        "statement_month"       : out_stmt_month,
        "performance_month"     : out_perf_month,
        "horizon"               : out_horizon,
        # Feature columns (static per account across all 28 horizons)
        "product_type"          : out_product_type,
        "risk_segment"          : out_risk_segment,
        "geography"             : out_geography,
        "credit_score"          : out_credit_score,
        "credit_limit"          : out_credit_limit,
        "months_on_book"        : out_mob,
        "origination_date"      : out_orig_date,
        # Target variables
        "actual_payment"        : out_act_pay,
        "predicted_payment"     : out_pred_pay,
        "actual_purchase_volume": out_act_purch,
        "predicted_purchase_volume": out_pred_purch,
        "actual_eos"            : out_act_eos,
        "predicted_eos"         : out_pred_eos,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    print(f"Generating account features ({N_ACCOUNTS:,} accounts)…")
    accounts = _generate_accounts(rng)

    # Round 1 — larger model noise, slight over-prediction bias
    print(f"Generating Round 1 ({N_ACCOUNTS:,} accts × {N_STMT_MONTHS} stmt months × {N_HORIZONS} horizons)…")
    df1 = _generate_round(
        accounts, rng,
        model_pay_bias   =  0.05,   # slight over-prediction of payments
        model_purch_bias =  0.08,   # over-prediction of purchases
        model_eos_bias   =  0.03,
        noise_scale      =  0.18,
    )
    path1 = os.path.join(OUTPUT_DIR, "ccar_round1.parquet")
    df1.write_parquet(path1)
    print(f"  Saved: {path1}")
    print(f"  Shape: {df1.shape[0]:,} rows × {df1.shape[1]} columns")

    # Round 2 — smaller noise, less bias (model improvements)
    print(f"Generating Round 2 ({N_ACCOUNTS:,} accts × {N_STMT_MONTHS} stmt months × {N_HORIZONS} horizons)…")
    df2 = _generate_round(
        accounts, rng,
        model_pay_bias   =  0.02,
        model_purch_bias =  0.03,
        model_eos_bias   =  0.01,
        noise_scale      =  0.10,
    )
    path2 = os.path.join(OUTPUT_DIR, "ccar_round2.parquet")
    df2.write_parquet(path2)
    print(f"  Saved: {path2}")
    print(f"  Shape: {df2.shape[0]:,} rows × {df2.shape[1]} columns")

    print("\nSchema:")
    for name, dtype in df1.schema.items():
        print(f"  {name:<30} {dtype}")

    print(f"\nSample (first 3 rows of Round 1):\n{df1.head(3)}")

    # Sanity check: features should be identical across all horizons for one account
    acct0 = df1.filter(
        (pl.col("account_id") == "ACC000000") &
        (pl.col("statement_month") == df1["statement_month"].min())
    )
    assert acct0["credit_score"].n_unique() == 1, "credit_score should be static across horizons"
    assert acct0["product_type"].n_unique() == 1, "product_type should be static across horizons"
    print("\nSanity check passed: feature columns are static across all horizons for each account.")
