"""
Generate synthetic CCAR backtesting datasets for testing.

Produces two CSV files:
  sample_data/ccar_round1.csv
  sample_data/ccar_round2.csv

Each row is an account with:
  - account_id
  - product_type   (Mortgage, Auto, Credit Card, Commercial)
  - geography      (Northeast, Southeast, Midwest, West)
  - vintage_year   (2019–2023)
  - risk_segment   (Low, Medium, High)
  - actual_loss    (observed loss rate)
  - predicted_loss (model-predicted loss rate)
  - exposure       (EAD in USD thousands)

Round 2 has slightly better predictions than Round 1 for most segments,
but worse for 'Credit Card' to make the comparison interesting.
"""
import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 5_000
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sample_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _make_dataset(noise_scale: float, cc_bias: float = 0.0) -> pd.DataFrame:
    product_type = RNG.choice(
        ["Mortgage", "Auto", "Credit Card", "Commercial"], size=N,
        p=[0.4, 0.25, 0.2, 0.15]
    )
    geography = RNG.choice(
        ["Northeast", "Southeast", "Midwest", "West"], size=N
    )
    vintage_year = RNG.integers(2019, 2024, size=N)
    risk_segment = RNG.choice(["Low", "Medium", "High"], size=N, p=[0.5, 0.35, 0.15])

    # True loss rates vary by product and risk
    base_loss = {
        "Mortgage": 0.01, "Auto": 0.025, "Credit Card": 0.06, "Commercial": 0.015
    }
    risk_mult = {"Low": 0.5, "Medium": 1.0, "High": 2.5}

    actual_loss = np.array([
        base_loss[p] * risk_mult[r] * (1 + 0.05 * (2023 - vy))
        for p, r, vy in zip(product_type, risk_segment, vintage_year)
    ])
    actual_loss += RNG.normal(0, 0.003, size=N)
    actual_loss = np.clip(actual_loss, 0.001, 0.30)

    # Predicted = actual + noise (simulate model errors)
    noise = RNG.normal(0, noise_scale, size=N)
    # Add systematic bias for Credit Card in round 1
    cc_mask = product_type == "Credit Card"
    predicted_loss = actual_loss + noise
    predicted_loss[cc_mask] += cc_bias
    predicted_loss = np.clip(predicted_loss, 0.001, 0.35)

    exposure = RNG.lognormal(mean=5.0, sigma=1.2, size=N)  # in $K

    return pd.DataFrame({
        "account_id": [f"ACC{i:06d}" for i in range(N)],
        "product_type": product_type,
        "geography": geography,
        "vintage_year": vintage_year,
        "risk_segment": risk_segment,
        "actual_loss": actual_loss.round(5),
        "predicted_loss": predicted_loss.round(5),
        "exposure_k": exposure.round(2),
    })


if __name__ == "__main__":
    round1 = _make_dataset(noise_scale=0.008, cc_bias=0.015)  # higher error, CC biased
    round2 = _make_dataset(noise_scale=0.004, cc_bias=0.030)  # better overall, CC worse

    p1 = os.path.join(OUTPUT_DIR, "ccar_round1.csv")
    p2 = os.path.join(OUTPUT_DIR, "ccar_round2.csv")
    round1.to_csv(p1, index=False)
    round2.to_csv(p2, index=False)

    print(f"Generated:\n  {p1}\n  {p2}")
    print(f"\nRound 1 sample:\n{round1.head(3).to_string(index=False)}")
    print(f"\nRound 2 sample:\n{round2.head(3).to_string(index=False)}")
