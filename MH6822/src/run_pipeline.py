
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from generate_synthetic_data import generate_synthetic_data
from sensitivity_analysis import run_all_scenarios
from train_credit_model import (
    apply_approval_decision,
    generate_denial_reasons,
    train_credit_model,
)


def _plot_baseline_abs_raad(baseline_results: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = baseline_results["jurisdiction"].tolist()
    values = baseline_results["abs_raad"].tolist()
    ax.bar(labels, values)

    # Jurisdiction-specific alert thresholds.
    us_amber = baseline_results.loc[baseline_results["jurisdiction"] == "US", "abs_raad"].index
    ax.axhline(0.05, linestyle="--", label="US amber |RAAD| = 0.05")
    ax.axhline(0.08, linestyle=":", label="US red |RAAD| = 0.08")
    ax.axhline(0.03, linestyle="--", label="EU amber |RAAD| = 0.03")
    ax.axhline(0.05, linestyle=":", label="EU red |RAAD| = 0.05")
    ax.set_ylabel("|RAAD|")
    ax.set_title("Baseline |RAAD| with Jurisdiction-Specific Thresholds")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "baseline_abs_raad_thresholds.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_baseline_air(baseline_results: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = baseline_results["jurisdiction"].tolist()
    values = baseline_results["air"].tolist()
    ax.bar(labels, values)
    ax.axhline(0.80, linestyle="--", label="AIR screening threshold = 0.80")
    ax.set_ylabel("AIR: protected approval rate / reference approval rate")
    ax.set_title("Baseline AIR Screening Metric")
    ax.set_ylim(0, max(1.0, max(values) + 0.1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "baseline_air_screening.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_baseline_psi(baseline_results: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(baseline_results))
    labels = baseline_results["jurisdiction"].tolist()
    ax.bar([i - 0.15 for i in x], baseline_results["overall_psi"], width=0.3, label="Overall PSI")
    ax.bar([i + 0.15 for i in x], baseline_results["subgroup_psi_gap"], width=0.3, label="Subgroup PSI gap")
    ax.axhline(0.10, linestyle="--", label="PSI amber = 0.10")
    ax.axhline(0.20, linestyle=":", label="PSI red = 0.20")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("PSI value")
    ax.set_title("Baseline Drift Metrics")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "baseline_psi_drift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_sensitivity_abs_raad(sensitivity_results: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    pivot = sensitivity_results.pivot_table(
        index="scenario",
        columns="jurisdiction",
        values="abs_raad",
        aggfunc="first",
    )
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0.03, linestyle="--", label="EU amber |RAAD| = 0.03")
    ax.axhline(0.05, linestyle="--", label="US amber / EU red |RAAD| = 0.05")
    ax.axhline(0.08, linestyle=":", label="US red |RAAD| = 0.08")
    ax.set_ylabel("|RAAD|")
    ax.set_title("Sensitivity Analysis: |RAAD| by Scenario and Jurisdiction")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "sensitivity_abs_raad.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rules_path = PROJECT_ROOT / "configs" / "jurisdiction_rules.yaml"
    data_dir = PROJECT_ROOT / "data"
    outputs_dir = PROJECT_ROOT / "outputs"
    figures_dir = outputs_dir / "figures"

    for d in [data_dir, outputs_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic data and running fixed-model sensitivity analysis...")
    baseline_data, baseline_results, sensitivity_results, flags = run_all_scenarios(
        rules_path
    )

    # Export the baseline processed dataset used by the pipeline.
    raw = generate_synthetic_data(scenario="baseline")
    model, scaler, full_df = train_credit_model(raw)
    full_df = apply_approval_decision(full_df, cutoff=0.18)
    full_df = generate_denial_reasons(full_df, model, scaler)

    csv_path = data_dir / "synthetic_bnpl_applications.csv"
    full_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path} ({len(full_df)} rows)")

    baseline_results.to_csv(outputs_dir / "baseline_results.csv", index=False)
    sensitivity_results.to_csv(outputs_dir / "sensitivity_results.csv", index=False)
    flags.to_csv(outputs_dir / "jurisdiction_flags.csv", index=False)
    print(f"Saved baseline_results.csv ({len(baseline_results)} rows)")
    print(f"Saved sensitivity_results.csv ({len(sensitivity_results)} rows)")

    # Remove the earlier mixed-scale chart if it exists.
    old_chart = figures_dir / "baseline_metrics_comparison.png"
    if old_chart.exists():
        old_chart.unlink()

    _plot_baseline_abs_raad(baseline_results, figures_dir)
    _plot_baseline_air(baseline_results, figures_dir)
    _plot_baseline_psi(baseline_results, figures_dir)
    _plot_sensitivity_abs_raad(sensitivity_results, figures_dir)

    print("Saved figures:")
    for fig in sorted(figures_dir.glob("*.png")):
        print(f"  - {fig}")

    print("\n--- Baseline Results ---")
    print(baseline_results.to_string(index=False))
    print("\n--- Pipeline complete ---")


if __name__ == "__main__":
    main()
