
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fairness_metrics import RAADResult

STATUS_PRIORITY = {"red": 3, "amber": 2, "watch": 1, "green": 0}


def load_jurisdiction_rules(config_path: str | Path) -> dict[str, Any]:
    """Load jurisdiction_rules.yaml."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_thresholds(config: dict[str, Any]) -> dict[str, Any]:
    """Extract PSI and missingness thresholds from flat or nested config."""
    if "data_governance_metrics" in config:
        dg = config["data_governance_metrics"]
        return {
            "overall_psi_amber": dg["overall_psi_amber_threshold"],
            "overall_psi_red": dg["overall_psi_red_threshold"],
            "subgroup_psi_gap_amber": dg["subgroup_psi_gap_amber_threshold"],
            "subgroup_missingness_max": dg["subgroup_missingness_max"],
            "subgroup_psi_required": dg.get("subgroup_psi_required", True),
        }
    return {
        "overall_psi_amber": config.get("overall_psi_amber_threshold", 0.10),
        "overall_psi_red": config.get("overall_psi_red_threshold", 0.20),
        "subgroup_psi_gap_amber": config.get(
            "subgroup_psi_gap_amber_threshold", 0.05
        ),
        "subgroup_missingness_max": config.get("subgroup_missingness_max", 0.05),
        "subgroup_psi_required": config.get("subgroup_psi_required", False),
    }


def _is_number(x: float) -> bool:
    return x is not None and not np.isnan(x)


def _max_status(current: str, candidate: str) -> str:
    """Return the more severe of two statuses."""
    return max(current, candidate, key=lambda s: STATUS_PRIORITY[s])


def classify_raad_status(
    abs_raad: float,
    ci_lower: float,
    ci_upper: float,
    amber_threshold: float,
    red_threshold: float,
) -> str:
    """
    Classify RAAD alert level using |RAAD| and bootstrap CI.

    Red: |RAAD| >= red threshold and CI does not cross zero.
    Amber: |RAAD| >= amber threshold, or |RAAD| >= red but CI crosses zero.
    Watch: close to amber threshold.
    """
    if not _is_number(abs_raad):
        return "amber"  # insufficient data is a governance concern, not a pass

    if abs_raad >= red_threshold:
        if _is_number(ci_lower) and _is_number(ci_upper) and ci_lower <= 0 <= ci_upper:
            return "amber"
        return "red"
    if abs_raad >= amber_threshold:
        return "amber"
    if abs_raad >= amber_threshold * 0.85:
        return "watch"
    return "green"


def apply_jurisdiction_rules(
    jurisdiction: str,
    raad_result: RAADResult,
    air: float,
    overall_psi: float,
    subgroup_psi_gap: float,
    audit_group_missingness: float,
    reason_coverage_rate: float,
    rules: dict[str, Any],
    config_applied: str | None = None,
) -> dict[str, Any]:
    """
    Apply jurisdiction-specific thresholds and generate compliance output.

    config_applied may differ from jurisdiction in misconfiguration scenarios.
    """
    config_key = config_applied or jurisdiction
    if config_key not in rules:
        raise ValueError(f"Unknown jurisdiction config: {config_key}")

    config = rules[config_key]
    thresholds = _get_thresholds(config)
    output_tone = config.get("output_tone", {})

    triggers: list[str] = []
    governance_actions: list[str] = []

    raad_status = classify_raad_status(
        raad_result.abs_raad,
        raad_result.ci_lower,
        raad_result.ci_upper,
        config["raad_amber_threshold"],
        config["raad_red_threshold"],
    )
    final_status = raad_status

    if _is_number(raad_result.abs_raad) and raad_result.abs_raad >= config["raad_amber_threshold"]:
        triggers.append(
            f"|RAAD|={raad_result.abs_raad:.3f} exceeds amber threshold "
            f"({config['raad_amber_threshold']})"
        )
        governance_actions.append(
            f"Review adversely affected group: {raad_result.affected_group}"
        )

    if _is_number(raad_result.abs_raad) and raad_result.abs_raad >= config["raad_red_threshold"]:
        triggers.append(
            f"|RAAD|={raad_result.abs_raad:.3f} exceeds red threshold "
            f"({config['raad_red_threshold']})"
        )

    air_threshold = config.get("air_screening_threshold", 0.80)
    if _is_number(air) and air < air_threshold:
        triggers.append(f"AIR={air:.3f} below screening threshold ({air_threshold})")
        governance_actions.append(
            "Supplementary outcome disparity review: AIR is a screening metric, not a legal conclusion"
        )
        final_status = _max_status(final_status, "watch")

    if audit_group_missingness > thresholds["subgroup_missingness_max"]:
        triggers.append(
            f"Audit group missingness {audit_group_missingness:.1%} exceeds limit "
            f"({thresholds['subgroup_missingness_max']:.1%})"
        )
        governance_actions.append("Insufficient evidence — data governance review")
        final_status = _max_status(final_status, "amber")

    if _is_number(overall_psi) and overall_psi >= thresholds["overall_psi_amber"]:
        triggers.append(f"Overall PSI={overall_psi:.3f} exceeds amber threshold")
        governance_actions.append("Investigate data drift before relying on fairness conclusion")
        final_status = _max_status(final_status, "watch")
        if thresholds["subgroup_psi_required"]:
            governance_actions.append(
                "Decompose drift by audit group and reassess RAAD / FNR gap"
            )

    if _is_number(overall_psi) and overall_psi >= thresholds["overall_psi_red"]:
        triggers.append(f"Overall PSI={overall_psi:.3f} exceeds red threshold")
        final_status = _max_status(final_status, "red")

    if (
        thresholds["subgroup_psi_required"]
        and _is_number(subgroup_psi_gap)
        and subgroup_psi_gap >= thresholds["subgroup_psi_gap_amber"]
    ):
        triggers.append(
            f"Subgroup PSI gap={subgroup_psi_gap:.3f} exceeds threshold "
            f"({thresholds['subgroup_psi_gap_amber']})"
        )
        governance_actions.append("Escalate: drift concentrated in audit subgroup")
        final_status = _max_status(final_status, "amber")

    reason_required_us = config.get("reason_mapping_required", False) or config.get(
        "adverse_action_reason_coverage_required", None
    ) is not None
    reason_required_eu = config.get("explainability_and_traceability", {}).get(
        "reason_mapping_required", False
    )
    if (reason_required_us or reason_required_eu) and reason_coverage_rate < 1.0:
        if config_key == "US":
            triggers.append("Incomplete adverse action reason coverage")
            governance_actions.append("Validate denial reasons against actual model and policy drivers")
        else:
            triggers.append("Reason mapping incomplete for EU traceability")
            governance_actions.append("Complete explainability logs before governance sign-off")
        final_status = _max_status(final_status, "amber")

    if config_applied and config_applied != jurisdiction:
        triggers.append(
            f"Jurisdictional misconfiguration: data={jurisdiction}, config={config_applied}"
        )
        governance_actions.append("Correct jurisdiction configuration and rerun all monitoring outputs")
        final_status = _max_status(final_status, "amber")

    tool_output = output_tone.get(final_status, "Continue monitoring")

    if config_key == "US":
        legal_boundary = config.get(
            "legal_interpretation_boundary",
            "Not an automatic ECOA violation",
        )
        regulatory_interpretation = (
            f"US fair lending monitoring — {legal_boundary}. {tool_output}"
        )
    else:
        regulatory_interpretation = f"EU high-risk AI governance — {tool_output}"

    if config_applied and config_applied != jurisdiction:
        regulatory_interpretation = (
            f"MISCONFIGURATION: {jurisdiction} data evaluated under "
            f"{config_applied} rules. {regulatory_interpretation}"
        )

    human_review_required = final_status in ("watch", "amber", "red") or len(triggers) > 0

    return {
        "jurisdiction": jurisdiction,
        "config_applied": config_applied or jurisdiction,
        "raad": raad_result.raad,
        "abs_raad": raad_result.abs_raad,
        "ci_lower": raad_result.ci_lower,
        "ci_upper": raad_result.ci_upper,
        "affected_group": raad_result.affected_group,
        "air": air,
        "overall_psi": overall_psi,
        "subgroup_psi_gap": subgroup_psi_gap,
        "audit_group_missingness": audit_group_missingness,
        "reason_coverage_rate": reason_coverage_rate,
        "status": final_status,
        "tool_output": tool_output,
        "regulatory_interpretation": regulatory_interpretation,
        "triggers": "; ".join(triggers) if triggers else "None",
        "governance_actions": "; ".join(governance_actions) if governance_actions else "None",
        "human_review_required": human_review_required,
        "raad_amber_threshold": config["raad_amber_threshold"],
        "raad_red_threshold": config["raad_red_threshold"],
    }
