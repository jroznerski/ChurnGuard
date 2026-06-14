"""Generate synthetic customer churn dataset (Telco-inspired, ~7 000 rows)."""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RANDOM_STATE = 42
N_CUSTOMERS = 7_043


def generate_customers(n: int = N_CUSTOMERS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_id = [f"CUST-{i:06d}" for i in range(1, n + 1)]
    gender = rng.choice(["Male", "Female"], n)
    senior_citizen = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])
    tenure = rng.integers(0, 73, n)

    phone_service = rng.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], n),
    )

    internet_service = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    no_internet = internet_service == "No"

    def internet_feature(prob_yes: float = 0.45) -> np.ndarray:
        vals = rng.choice(["Yes", "No"], n, p=[prob_yes, 1 - prob_yes])
        return np.where(no_internet, "No internet service", vals)

    online_security = internet_feature(0.29)
    online_backup = internet_feature(0.34)
    device_protection = internet_feature(0.34)
    tech_support = internet_feature(0.29)
    streaming_tv = internet_feature(0.38)
    streaming_movies = internet_feature(0.39)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        n,
        p=[0.55, 0.21, 0.24],
    )
    paperless_billing = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        n,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    # Monthly charges correlated with services
    base_charge = 20.0 + rng.normal(0, 3, n)
    base_charge += np.where(internet_service == "Fiber optic", 30, 0)
    base_charge += np.where(internet_service == "DSL", 15, 0)
    base_charge += np.where(multiple_lines == "Yes", 10, 0)
    base_charge += np.where(streaming_tv == "Yes", 8, 0)
    base_charge += np.where(streaming_movies == "Yes", 8, 0)
    base_charge += np.where(online_security == "Yes", 6, 0)
    base_charge += np.where(tech_support == "Yes", 6, 0)
    monthly_charges = np.clip(base_charge, 18.0, 120.0).round(2)
    total_charges = (monthly_charges * tenure + rng.normal(0, 5, n)).clip(0).round(2)

    # Churn probability (logistic-like)
    log_odds = (
        -2.5
        + 0.03 * (monthly_charges - 65)
        - 0.04 * tenure
        + np.where(contract == "Month-to-month", 1.2, 0)
        + np.where(contract == "One year", 0.3, 0)
        + np.where(internet_service == "Fiber optic", 0.6, 0)
        + np.where(payment_method == "Electronic check", 0.4, 0)
        + np.where(senior_citizen == 1, 0.3, 0)
        + np.where(partner == "No", 0.2, 0)
        + np.where(tech_support == "No", 0.25, 0)
        + np.where(online_security == "No", 0.25, 0)
        + rng.normal(0, 0.5, n)
    )
    churn_prob = 1 / (1 + np.exp(-log_odds))
    churn = (rng.random(n) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure": tenure,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "contract": contract,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "churn": churn,
        }
    )

    # Inject ~1% missing values in total_charges to simulate real data
    mask = rng.random(n) < 0.01
    df.loc[mask, "total_charges"] = np.nan

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic churn dataset")
    parser.add_argument("--output", default="data/raw/customers.csv")
    parser.add_argument("--n", type=int, default=N_CUSTOMERS)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = generate_customers(n=args.n, seed=args.seed)
    df.to_csv(args.output, index=False)

    churn_rate = df["churn"].mean() * 100
    print(f"Generated {len(df):,} customers — churn rate: {churn_rate:.1f}%")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
