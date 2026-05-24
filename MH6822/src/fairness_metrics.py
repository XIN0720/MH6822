
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

REFERENCE_GROUP = "reference_group"
PROTECTED_GROUP = "protected_or_vulnerable_group"
MIN_BAND_SAMPLES = 30


@dataclass
class RAADResult:
    raad: float
    abs_raad: float
    affected_group: str
    ci_lower: float
    ci_upper: float
    band_results: list[dict[str, Any]]
    n_bands_used: int


def compute_psi(
    baseline: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    """
    Population Stability Index comparing current vs baseline distribution.

    Uses numpy histogram rather than pandas interval bins to avoid interval-index
    instability and to ensure out-of-range current values are counted.
    """
    baseline_arr = baseline.dropna().to_numpy(dtype=float)
    current_arr = current.dropna().to_numpy(dtype=float)
    if len(baseline_arr) == 0 or len(current_arr) == 0:
        return np.nan

    cutpoints = np.unique(np.quantile(baseline_arr, np.linspace(0, 1, bins + 1)))
    if len(cutpoints) < 3:
        return 0.0
    cutpoints = cutpoints.astype(float)
    cutpoints[0] = -np.inf
    cutpoints[-1] = np.inf

    baseline_counts, _ = np.histogram(baseline_arr, bins=cutpoints)
    current_counts, _ = np.histogram(current_arr, bins=cutpoints)

    baseline_pct = baseline_counts / max(baseline_counts.sum(), 1)
    current_pct = current_counts / max(current_counts.sum(), 1)

    baseline_pct = np.where(baseline_pct == 0, 1e-6, baseline_pct)
    current_pct = np.where(current_pct == 0, 1e-6, current_pct)

    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(psi)


def _affected_group_from_raad(raad: float) -> str:
    if np.isnan(raad):
        return "insufficient_data"
    if np.isclose(raad, 0.0, atol=1e-12):
        return "no_directional_disparity"
    return PROTECTED_GROUP if raad > 0 else REFERENCE_GROUP


def _raad_components(
    scores: np.ndarray,
    approved: np.ndarray,
    groups: np.ndarray,
    min_band_samples: int,
) -> tuple[float, list[dict[str, Any]]]:
    """Compute RAAD and valid-band details from numpy arrays."""
    if len(scores) == 0:
        return np.nan, []

    edges = np.unique(np.quantile(scores, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return np.nan, []

    internal_edges = edges[1:-1]
    band_ids = np.digitize(scores, internal_edges, right=True)
    n_bands = len(edges) - 1
    total_n = len(scores)
    band_results: list[dict[str, Any]] = []

    for band_id in range(n_bands):
        mask = band_ids == band_id
        if not np.any(mask):
            continue
        ref_mask = mask & (groups == REFERENCE_GROUP)
        vul_mask = mask & (groups == PROTECTED_GROUP)
        n_ref = int(ref_mask.sum())
        n_vul = int(vul_mask.sum())
        if n_ref < min_band_samples or n_vul < min_band_samples:
            continue
        ref_rate = float(approved[ref_mask].mean())
        vul_rate = float(approved[vul_mask].mean())
        gap = ref_rate - vul_rate
        weight = float(mask.sum() / total_n)
        band_results.append(
            {
                "risk_band": f"band_{band_id + 1}",
                "reference_approval_rate": ref_rate,
                "protected_approval_rate": vul_rate,
                "gap": float(gap),
                "weight": weight,
                "n_reference": n_ref,
                "n_protected": n_vul,
            }
        )

    if not band_results:
        return np.nan, []

    weight_sum = sum(item["weight"] for item in band_results)
    if weight_sum == 0:
        return np.nan, band_results
    raad = sum(item["gap"] * item["weight"] for item in band_results) / weight_sum
    return float(raad), band_results


def compute_raad(
    df: pd.DataFrame,
    n_bootstrap: int = 50,
    random_seed: int = 42,
    min_band_samples: int = MIN_BAND_SAMPLES,
) -> RAADResult:
    """
    Compute Risk-Adjusted Approval Disparity (RAAD).

    RAAD = weighted average over risk bands of
    approval_rate(reference group) - approval_rate(protected/vulnerable group).

    Alerting must use |RAAD|. The sign is retained only to identify which group
    is adversely affected. Both directions are monitored.
    """
    work = df.dropna(subset=["predicted_pd", "approved", "audit_group"]).copy()
    if work.empty:
        return RAADResult(np.nan, np.nan, "insufficient_data", np.nan, np.nan, [], 0)

    scores = work["predicted_pd"].to_numpy(dtype=float)
    approved = work["approved"].to_numpy(dtype=float)
    groups = work["audit_group"].to_numpy(dtype=object)

    raad, band_results = _raad_components(scores, approved, groups, min_band_samples)
    if not band_results or np.isnan(raad):
        return RAADResult(np.nan, np.nan, "insufficient_data", np.nan, np.nan, [], 0)

    rng = np.random.default_rng(random_seed)
    bootstrap_estimates: list[float] = []
    n = len(scores)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        estimate, _ = _raad_components(
            scores[idx], approved[idx], groups[idx], min_band_samples
        )
        if not np.isnan(estimate):
            bootstrap_estimates.append(estimate)

    if bootstrap_estimates:
        ci_lower = float(np.percentile(bootstrap_estimates, 2.5))
        ci_upper = float(np.percentile(bootstrap_estimates, 97.5))
    else:
        ci_lower = np.nan
        ci_upper = np.nan

    return RAADResult(
        raad=float(raad),
        abs_raad=float(abs(raad)),
        affected_group=_affected_group_from_raad(raad),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        band_results=band_results,
        n_bands_used=len(band_results),
    )


def compute_air(df: pd.DataFrame) -> float:
    """Approval rate ratio: protected/vulnerable group approval rate divided by reference group approval rate."""
    ref = df[df["audit_group"] == REFERENCE_GROUP]
    vul = df[df["audit_group"] == PROTECTED_GROUP]
    if len(ref) == 0 or len(vul) == 0:
        return np.nan
    ref_rate = ref["approved"].mean()
    vul_rate = vul["approved"].mean()
    if ref_rate == 0:
        return np.nan
    return float(vul_rate / ref_rate)


def compute_fnr_gap(df: pd.DataFrame) -> float:
    """
    Gap in false negative rate among non-defaulters.

    FNR = P(rejected | no default). Gap = reference FNR - protected FNR.
    """
    non_defaulters = df[df["actual_default_90d"] == 0].copy()
    ref = non_defaulters[non_defaulters["audit_group"] == REFERENCE_GROUP]
    vul = non_defaulters[non_defaulters["audit_group"] == PROTECTED_GROUP]
    if len(ref) == 0 or len(vul) == 0:
        return np.nan
    ref_fnr = 1.0 - ref["approved"].mean()
    vul_fnr = 1.0 - vul["approved"].mean()
    return float(ref_fnr - vul_fnr)


def compute_drift_metrics(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    score_col: str = "predicted_pd",
) -> dict[str, float]:
    """
    Compute overall PSI and subgroup PSI gap.

    PSI is a data-governance trigger, not a fairness proof. EU-mode interpretation
    should decompose drift by audit group and then reassess RAAD/FNR effects.
    """
    overall_psi = compute_psi(baseline_df[score_col], current_df[score_col])

    ref_base = baseline_df[baseline_df["audit_group"] == REFERENCE_GROUP][score_col]
    ref_curr = current_df[current_df["audit_group"] == REFERENCE_GROUP][score_col]
    vul_base = baseline_df[baseline_df["audit_group"] == PROTECTED_GROUP][score_col]
    vul_curr = current_df[current_df["audit_group"] == PROTECTED_GROUP][score_col]

    ref_psi = compute_psi(ref_base, ref_curr)
    vul_psi = compute_psi(vul_base, vul_curr)
    subgroup_psi_gap = abs(ref_psi - vul_psi) if not (
        np.isnan(ref_psi) or np.isnan(vul_psi)
    ) else np.nan

    return {
        "overall_psi": overall_psi,
        "reference_subgroup_psi": ref_psi,
        "protected_subgroup_psi": vul_psi,
        "subgroup_psi_gap": subgroup_psi_gap,
    }


def compute_audit_group_missingness(df: pd.DataFrame) -> float:
    """Share of records with missing audit_group."""
    return float(df["audit_group"].isna().mean())
