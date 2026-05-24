
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from generate_synthetic_data import FEATURE_COLUMNS, REASON_CODE_LIBRARY


POLICY_REASON_RULES = [
    ("legacy_bureau_score", "LOW_LEGACY_BUREAU_SCORE", lambda x: x < 0.45),
    ("thin_file_flag", "THIN_CREDIT_FILE", lambda x: x == 1),
    ("device_consistency_score", "DEVICE_INCONSISTENCY", lambda x: x < 0.55),
]


def train_credit_model(
    df: pd.DataFrame,
    random_seed: int = 42,
) -> tuple[LogisticRegression, StandardScaler, pd.DataFrame]:
    """
    Train a logistic regression credit risk model.

    Model inputs exclude audit_group, jurisdiction, approved, and denial reasons.
    The model is intentionally simple and interpretable for a compliance prototype.
    """
    model_df = df.copy()
    X = model_df[FEATURE_COLUMNS].values
    y = model_df["actual_default_90d"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=random_seed)
    model.fit(X_scaled, y)

    model_df["predicted_pd"] = model.predict_proba(X_scaled)[:, 1]
    return model, scaler, model_df


def score_with_existing_model(
    df: pd.DataFrame,
    model: LogisticRegression,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Score a current-period dataset using the already-approved baseline model."""
    out = df.copy()
    X_scaled = scaler.transform(out[FEATURE_COLUMNS].values)
    out["predicted_pd"] = model.predict_proba(X_scaled)[:, 1]
    return out


def apply_approval_decision(
    df: pd.DataFrame,
    cutoff: float = 0.18,
) -> pd.DataFrame:
    """
    Apply business approval cutoff on model score plus legacy policy adjustments.

    The policy layer uses legacy_bureau_score (not in the ML model) plus proxy
    features to simulate residual non-model rules still present in production BNPL
    decisioning. RAAD then detects whether approval disparities persist at equal
    predicted_pd.
    """
    out = df.copy()
    policy_adjustment = (
        0.050 * (1.0 - out["legacy_bureau_score"])
        + 0.012 * out["thin_file_flag"]
        + 0.008 * (1.0 - out["device_consistency_score"])
    )
    out["policy_adjustment"] = policy_adjustment
    out["effective_pd"] = (out["predicted_pd"] + policy_adjustment).clip(0, 1)
    out["approved"] = (out["effective_pd"] <= cutoff).astype(int)
    return out


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            output.append(item)
            seen.add(item)
    return output


def _policy_layer_reasons(row: pd.Series) -> list[str]:
    """Return decision reasons from residual policy-layer variables."""
    reasons: list[str] = []
    for feature, code, predicate in POLICY_REASON_RULES:
        if feature in row.index and predicate(row[feature]):
            reasons.append(code)
    return reasons


def generate_denial_reasons(
    df: pd.DataFrame,
    model: LogisticRegression,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Map denied applications to top reason codes.

    The mapping covers both:
    1. policy-layer factors used in the actual approval decision; and
    2. positive model-feature contributions that increased predicted default risk.

    This supports US adverse-action review and EU traceability/auditability.
    The implementation assigns reason columns in arrays rather than repeatedly
    mutating a DataFrame row-by-row, which keeps the pipeline fast and stable.
    """
    out = df.copy(deep=False)
    n_rows = len(out)
    reason_1 = np.full(n_rows, None, dtype=object)
    reason_2 = np.full(n_rows, None, dtype=object)
    reason_3 = np.full(n_rows, None, dtype=object)

    approved = out["approved"].to_numpy()
    denied_positions = np.flatnonzero(approved == 0)
    if len(denied_positions) == 0:
        out = out.assign(
            denial_reason_1=reason_1,
            denial_reason_2=reason_2,
            denial_reason_3=reason_3,
        )
        return out

    coef = model.coef_[0]
    feature_impact = {
        FEATURE_COLUMNS[i]: abs(coef[i]) for i in range(len(FEATURE_COLUMNS))
    }
    ranked_features = sorted(
        feature_impact, key=feature_impact.get, reverse=True  # type: ignore[arg-type]
    )

    X_denied = out.iloc[denied_positions][FEATURE_COLUMNS].values
    X_denied_scaled = scaler.transform(X_denied)
    contribution_matrix = X_denied_scaled * coef

    legacy_scores = out.iloc[denied_positions]["legacy_bureau_score"].to_numpy()
    thin_file = out.iloc[denied_positions]["thin_file_flag"].to_numpy()
    device_scores = out.iloc[denied_positions]["device_consistency_score"].to_numpy()

    for local_i, pos in enumerate(denied_positions):
        reason_codes: list[str] = []

        if legacy_scores[local_i] < 0.45:
            reason_codes.append("LOW_LEGACY_BUREAU_SCORE")
        if thin_file[local_i] == 1:
            reason_codes.append("THIN_CREDIT_FILE")
        if device_scores[local_i] < 0.55:
            reason_codes.append("DEVICE_INCONSISTENCY")

        contributions = contribution_matrix[local_i]
        ordered_features = np.argsort(contributions)[::-1]
        added_model_reason = False
        for feature_idx in ordered_features:
            if contributions[feature_idx] <= 0:
                continue
            feature = FEATURE_COLUMNS[feature_idx]
            code = REASON_CODE_LIBRARY.get(feature, feature.upper())
            reason_codes.append(code)
            added_model_reason = True
            if len(_dedupe_preserve_order(reason_codes)) >= 3:
                break

        if not added_model_reason:
            for feature in ranked_features[:3]:
                reason_codes.append(REASON_CODE_LIBRARY.get(feature, feature.upper()))

        reason_codes = _dedupe_preserve_order(reason_codes)
        while len(reason_codes) < 3:
            reason_codes.append("ADDITIONAL_RISK_FACTORS")

        reason_1[pos] = reason_codes[0]
        reason_2[pos] = reason_codes[1]
        reason_3[pos] = reason_codes[2]

    out = out.assign(
        denial_reason_1=reason_1,
        denial_reason_2=reason_2,
        denial_reason_3=reason_3,
    )
    return out


def audit_denial_reason_coverage(df: pd.DataFrame) -> dict[str, Any]:
    """Check whether denied applicants have mapped, non-vague denial reasons."""
    denied = df[df["approved"] == 0]
    if len(denied) == 0:
        return {"coverage_rate": 1.0, "vague_reason_count": 0}

    has_reason = denied["denial_reason_1"].notna()
    coverage_rate = float(has_reason.mean())
    vague = denied["denial_reason_1"].isin(
        ["ADDITIONAL_RISK_FACTORS", np.nan]
    ).sum()
    return {
        "coverage_rate": coverage_rate,
        "vague_reason_count": int(vague),
    }
