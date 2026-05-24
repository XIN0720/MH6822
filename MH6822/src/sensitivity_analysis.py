
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from fairness_metrics import (
    compute_air,
    compute_audit_group_missingness,
    compute_drift_metrics,
    compute_fnr_gap,
    compute_raad,
)
from generate_synthetic_data import apply_missing_audit_group, generate_synthetic_data
from jurisdiction_engine import apply_jurisdiction_rules, load_jurisdiction_rules
from train_credit_model import (
    apply_approval_decision,
    audit_denial_reason_coverage,
    generate_denial_reasons,
    score_with_existing_model,
    train_credit_model,
)


def evaluate_subset(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    jurisdiction: str,
    rules: dict[str, Any],
    scenario: str,
    model: LogisticRegression,
    scaler: StandardScaler,
    config_applied: str | None = None,
    cutoff: float = 0.18,
) -> dict[str, Any]:
    """Run metrics and rule engine on a jurisdiction subset using the fixed baseline model."""
    modeled_full = score_with_existing_model(df, model, scaler)
    modeled_full = apply_approval_decision(modeled_full, cutoff=cutoff)
    modeled_full = generate_denial_reasons(modeled_full, model, scaler)

    subset = modeled_full[modeled_full["jurisdiction"] == jurisdiction].copy()
    if len(subset) == 0:
        return {}

    baseline_sub = baseline_df[baseline_df["jurisdiction"] == jurisdiction]
    drift = compute_drift_metrics(baseline_sub, subset)
    raad_result = compute_raad(subset)
    air = compute_air(subset)
    fnr_gap = compute_fnr_gap(subset)
    missingness = compute_audit_group_missingness(subset)
    reason_audit = audit_denial_reason_coverage(subset)

    rule_output = apply_jurisdiction_rules(
        jurisdiction=jurisdiction,
        raad_result=raad_result,
        air=air,
        overall_psi=drift["overall_psi"],
        subgroup_psi_gap=drift["subgroup_psi_gap"],
        audit_group_missingness=missingness,
        reason_coverage_rate=reason_audit["coverage_rate"],
        rules=rules,
        config_applied=config_applied,
    )

    return {
        "scenario": scenario,
        "jurisdiction": jurisdiction,
        "config_applied": config_applied or jurisdiction,
        "cutoff": cutoff,
        "n_applications": len(subset),
        "raad": raad_result.raad,
        "abs_raad": raad_result.abs_raad,
        "ci_lower": raad_result.ci_lower,
        "ci_upper": raad_result.ci_upper,
        "ci_95": f"[{raad_result.ci_lower:.3f}, {raad_result.ci_upper:.3f}]",
        "air": air,
        "fnr_gap": fnr_gap,
        "overall_psi": drift["overall_psi"],
        "reference_subgroup_psi": drift["reference_subgroup_psi"],
        "protected_subgroup_psi": drift["protected_subgroup_psi"],
        "subgroup_psi_gap": drift["subgroup_psi_gap"],
        "affected_group": raad_result.affected_group,
        "audit_group_missingness": missingness,
        "reason_coverage_rate": reason_audit["coverage_rate"],
        "status": rule_output["status"],
        "tool_output": rule_output["tool_output"],
        "regulatory_interpretation": rule_output["regulatory_interpretation"],
        "human_review_required": rule_output["human_review_required"],
        "triggers": rule_output["triggers"],
        "governance_actions": rule_output["governance_actions"],
    }


def run_all_scenarios(
    rules_path: str | Path,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute baseline and sensitivity scenarios.

    Returns
    -------
    baseline_processed, baseline_results, sensitivity_results, jurisdiction_flags
    """
    rules = load_jurisdiction_rules(rules_path)

    baseline_raw = generate_synthetic_data(
        scenario="baseline", random_seed=random_seed
    )
    model, scaler, baseline_processed = train_credit_model(
        baseline_raw, random_seed
    )
    baseline_processed = apply_approval_decision(baseline_processed, cutoff=0.18)
    baseline_processed = generate_denial_reasons(
        baseline_processed, model, scaler
    )

    results: list[dict[str, Any]] = []

    for jurisdiction in ["US", "EU"]:
        results.append(
            evaluate_subset(
                baseline_raw,
                baseline_processed,
                jurisdiction,
                rules,
                scenario="baseline",
                model=model,
                scaler=scaler,
                config_applied=jurisdiction,
            )
        )

    baseline_df = pd.DataFrame(results)

    sensitivity_scenarios: list[tuple[str, pd.DataFrame, float]] = []

    stress_raw = generate_synthetic_data(
        scenario="economic_stress", random_seed=random_seed + 1
    )
    sensitivity_scenarios.append(("economic_stress", stress_raw, 0.18))

    shift_raw = generate_synthetic_data(
        scenario="distribution_shift", random_seed=random_seed + 2
    )
    sensitivity_scenarios.append(("distribution_shift", shift_raw, 0.18))

    tighten_raw = generate_synthetic_data(
        scenario="baseline", random_seed=random_seed
    )
    sensitivity_scenarios.append(("credit_tightening", tighten_raw, 0.14))

    for missing_rate in [0.10, 0.20, 0.30]:
        missing_raw = apply_missing_audit_group(
            generate_synthetic_data(scenario="baseline", random_seed=random_seed),
            missing_rate=missing_rate,
            random_seed=random_seed + int(missing_rate * 100),
        )
        sensitivity_scenarios.append(
            (f"missing_audit_{int(missing_rate * 100)}pct", missing_raw, 0.18)
        )

    misconfig_raw = generate_synthetic_data(
        scenario="baseline", random_seed=random_seed
    )
    sensitivity_scenarios.append(("misconfig_eu_data_us_rules", misconfig_raw, 0.18))
    sensitivity_scenarios.append(("misconfig_us_data_eu_rules", misconfig_raw, 0.18))

    sensitivity_results: list[dict[str, Any]] = []

    for scenario_name, data, cutoff in sensitivity_scenarios:
        for jurisdiction in ["US", "EU"]:
            config_applied = jurisdiction

            if scenario_name == "misconfig_eu_data_us_rules" and jurisdiction == "EU":
                config_applied = "US"
            elif scenario_name == "misconfig_us_data_eu_rules" and jurisdiction == "US":
                config_applied = "EU"

            row = evaluate_subset(
                data,
                baseline_processed,
                jurisdiction,
                rules,
                scenario=scenario_name,
                model=model,
                scaler=scaler,
                config_applied=config_applied,
                cutoff=cutoff,
            )
            sensitivity_results.append(row)

    sensitivity_df = pd.DataFrame(sensitivity_results)
    flags = pd.concat([baseline_df, sensitivity_df], ignore_index=True)

    return baseline_processed, baseline_df, sensitivity_df, flags
