
from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_SEED = 42

FEATURE_COLUMNS = [
    "income",
    "debt_to_income_ratio",
    "credit_history_length_months",
    "prior_late_payments",
    "existing_klarna_balance",
    "purchase_amount",
    "merchant_category_risk",
    "thin_file_flag",
    "device_consistency_score",
    "repayment_history_score",
]

REASON_CODE_LIBRARY = {
    "income": "INSUFFICIENT_INCOME",
    "debt_to_income_ratio": "HIGH_DEBT_TO_INCOME",
    "credit_history_length_months": "LIMITED_CREDIT_HISTORY",
    "prior_late_payments": "PRIOR_LATE_PAYMENTS",
    "existing_klarna_balance": "HIGH_EXISTING_BALANCE",
    "purchase_amount": "PURCHASE_AMOUNT_TOO_HIGH",
    "merchant_category_risk": "ELEVATED_MERCHANT_RISK",
    "thin_file_flag": "THIN_CREDIT_FILE",
    "device_consistency_score": "DEVICE_INCONSISTENCY",
    "repayment_history_score": "WEAK_REPAYMENT_HISTORY",
    "legacy_bureau_score": "LOW_LEGACY_BUREAU_SCORE",
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_data(
    n_us: int = 10_000,
    n_eu: int = 10_000,
    random_seed: int = RANDOM_SEED,
    scenario: str = "baseline",
) -> pd.DataFrame:
    """
    Generate synthetic BNPL application records for US and EU jurisdictions.

    Parameters
    ----------
    n_us, n_eu : int
        Sample sizes per jurisdiction.
    random_seed : int
        Random seed for reproducibility.
    scenario : str
        Data generation scenario: baseline, economic_stress, distribution_shift.
    """
    rng = np.random.default_rng(random_seed)
    rows: list[pd.DataFrame] = []

    for jurisdiction, n in [("US", n_us), ("EU", n_eu)]:
        applicant_ids = [
            f"{jurisdiction}-{i:05d}" for i in range(1, n + 1)
        ]
        audit_group = rng.choice(
            ["reference_group", "protected_or_vulnerable_group"],
            size=n,
            p=[0.55, 0.45],
        )

        is_protected = audit_group == "protected_or_vulnerable_group"

        income = rng.lognormal(mean=10.5, sigma=0.45, size=n)
        if scenario == "economic_stress":
            income = income * 0.78

        thin_file_prob = np.where(is_protected, 0.42, 0.16)
        if scenario == "distribution_shift":
            thin_file_prob = np.where(is_protected, 0.58, 0.16)

        thin_file_flag = rng.binomial(1, thin_file_prob).astype(float)

        credit_history_base = rng.gamma(shape=2.5, scale=18.0, size=n)
        credit_history_length_months = np.where(
            is_protected,
            credit_history_base * 0.75,
            credit_history_base,
        )
        if scenario == "distribution_shift":
            credit_history_length_months = np.where(
                is_protected,
                credit_history_length_months * 0.70,
                credit_history_length_months,
            )

        debt_to_income_ratio = np.clip(
            rng.normal(
                loc=np.where(is_protected, 0.42, 0.34),
                scale=0.10,
                size=n,
            ),
            0.05,
            0.95,
        )
        if scenario == "economic_stress":
            debt_to_income_ratio = np.clip(debt_to_income_ratio + 0.08, 0.05, 0.95)

        prior_late_payments = rng.poisson(
            lam=np.where(is_protected, 1.1, 0.7),
            size=n,
        ).astype(float)
        if scenario == "economic_stress":
            prior_late_payments = prior_late_payments + rng.binomial(
                1, 0.22, size=n
            ).astype(float)

        existing_klarna_balance = np.clip(
            rng.gamma(shape=2.0, scale=350.0, size=n),
            0,
            5000,
        )
        if scenario == "economic_stress":
            existing_klarna_balance = np.clip(existing_klarna_balance * 1.20, 0, 5000)
        purchase_amount = np.clip(
            rng.lognormal(mean=5.2, sigma=0.55, size=n),
            20,
            3000,
        )
        merchant_category_risk = rng.uniform(0.0, 1.0, size=n)
        device_consistency_score = np.clip(
            rng.normal(
                loc=np.where(is_protected, 0.62, 0.74),
                scale=0.12,
                size=n,
            ),
            0.0,
            1.0,
        )
        repayment_history_score = np.clip(
            rng.normal(
                loc=np.where(is_protected, 0.58, 0.70),
                scale=0.14,
                size=n,
            ),
            0.0,
            1.0,
        )

        # Legacy bureau score: used in residual policy rules but NOT in the ML model
        legacy_bureau_score = np.clip(
            rng.normal(
                loc=np.where(is_protected, 0.42, 0.62),
                scale=0.18,
                size=n,
            ),
            0.0,
            1.0,
        )

        logit = (
            -2.10
            + 0.55 * thin_file_flag
            + 0.85 * debt_to_income_ratio
            - 0.004 * credit_history_length_months
            + 0.35 * prior_late_payments
            + 0.00015 * existing_klarna_balance
            + 0.00008 * purchase_amount
            + 0.45 * merchant_category_risk
            - 0.55 * device_consistency_score
            - 0.90 * repayment_history_score
            - 0.000008 * income
            + rng.normal(0, 0.35, size=n)
        )
        if scenario == "economic_stress":
            logit = logit + 0.35

        true_pd = _sigmoid(logit)
        actual_default_90d = rng.binomial(1, true_pd).astype(int)

        df = pd.DataFrame(
            {
                "applicant_id": applicant_ids,
                "jurisdiction": jurisdiction,
                "audit_group": audit_group,
                "income": income,
                "debt_to_income_ratio": debt_to_income_ratio,
                "credit_history_length_months": credit_history_length_months,
                "prior_late_payments": prior_late_payments,
                "existing_klarna_balance": existing_klarna_balance,
                "purchase_amount": purchase_amount,
                "merchant_category_risk": merchant_category_risk,
                "thin_file_flag": thin_file_flag,
                "device_consistency_score": device_consistency_score,
                "repayment_history_score": repayment_history_score,
                "legacy_bureau_score": legacy_bureau_score,
                "true_pd": true_pd,
                "actual_default_90d": actual_default_90d,
            }
        )
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


def apply_missing_audit_group(
    df: pd.DataFrame,
    missing_rate: float,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Simulate missing audit_group values for data governance testing."""
    out = df.copy()
    rng = np.random.default_rng(random_seed)
    n_missing = int(len(out) * missing_rate)
    if n_missing == 0:
        return out
    missing_idx = rng.choice(out.index, size=n_missing, replace=False)
    out.loc[missing_idx, "audit_group"] = np.nan
    return out
