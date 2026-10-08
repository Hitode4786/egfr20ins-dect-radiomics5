from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from egfr20ins_paper_common import compute_binary_metrics_detailed, summarize_metric_series
from luad_tabular_utils import build_logger, save_json
from egfr20ins_runtime_guard import assert_frozen_runtime


def require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    import matplotlib.pyplot as plt

    return plt


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def find_row(rows: Sequence[Dict[str, str]], key: str, value: str) -> Dict[str, str]:
    for row in rows:
        if str(row.get(key, "")) == str(value):
            return row
    raise KeyError(f"Missing row where {key} == {value!r}")


def load_main_branch_data(results_dir: Path) -> Tuple[str, float, np.ndarray, np.ndarray, Dict[str, float]]:
    fixed_dir = results_dir / "fixed_bootstrap"
    prediction_rows = read_csv_rows(fixed_dir / "fixed_test_predictions.csv")
    summary_rows = read_csv_rows(fixed_dir / "fixed_test_summary.csv")
    main_summary_row = find_row(summary_rows, "branch_name", "fusion_rfe_xgboost_calibrated")

    y_true = np.asarray([int(row["y_true"]) for row in prediction_rows], dtype=np.int64)
    y_score = np.asarray(
        [float(row["fusion_rfe_xgboost_calibrated_score"]) for row in prediction_rows],
        dtype=np.float64,
    )
    threshold = float(main_summary_row["threshold"])
    observed = {
        "auc": float(main_summary_row["auc"]),
        "specificity": float(main_summary_row["specificity"]),
        "brier": float(main_summary_row["brier"]),
    }
    return "fusion_rfe_xgboost_calibrated", threshold, y_true, y_score, observed


def bootstrap_distributions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    *,
    n_bootstrap: int,
    random_state: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(random_state)
    auc_values: List[float] = []
    specificity_values: List[float] = []
    brier_values: List[float] = []
    attempts = 0
    max_attempts = max(1000, n_bootstrap * 50)
    while len(auc_values) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        sample_indices = rng.integers(0, len(y_true), size=len(y_true))
        sample_true = y_true[sample_indices]
        if np.unique(sample_true).size < 2:
            continue
        sample_score = y_score[sample_indices]
        metrics = compute_binary_metrics_detailed(sample_true, sample_score, threshold)
        auc_values.append(float(metrics["auc"]))
        specificity_values.append(float(metrics["specificity"]))
        brier_values.append(float(metrics["brier"]))

    return {
        "requested": int(n_bootstrap),
        "valid": int(len(auc_values)),
        "attempts": int(attempts),
        "auc": auc_values,
        "specificity": specificity_values,
        "brier": brier_values,
        "auc_summary": summarize_metric_series(auc_values),
        "specificity_summary": summarize_metric_series(specificity_values),
        "brier_summary": summarize_metric_series(brier_values),
    }


def save_samples_csv(path: Path, distributions: Dict[str, object]) -> None:
    auc_values = distributions["auc"]  # type: ignore[assignment]
    specificity_values = distributions["specificity"]  # type: ignore[assignment]
    brier_values = distributions["brier"]  # type: ignore[assignment]
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["sample_index", "auc", "specificity", "brier"])
        writer.writeheader()
        for idx, (auc, spec, brier) in enumerate(zip(auc_values, specificity_values, brier_values), start=1):
            writer.writerow(
                {
                    "sample_index": idx,
                    "auc": f"{float(auc):.10f}",
                    "specificity": f"{float(spec):.10f}",
                    "brier": f"{float(brier):.10f}",
                }
            )


def plot_density_panel(ax, values: Sequence[float], *, title: str, color: str, observed: float) -> None:
    arr = np.asarray([float(value) for value in values if np.isfinite(value)], dtype=np.float64)
    if arr.size == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No valid bootstrap samples", ha="center", va="center", transform=ax.transAxes)
        return

    xmin = float(np.min(arr))
    xmax = float(np.max(arr))
    if np.isclose(xmin, xmax):
        pad = 0.05 if abs(xmin) < 1e-6 else abs(xmin) * 0.05
        xmin -= pad
        xmax += pad
    else:
        pad = 0.06 * (xmax - xmin)
        xmin -= pad
        xmax += pad

    ax.hist(arr, bins=28, density=True, color=color, alpha=0.22, edgecolor="none")
    if arr.size > 1 and np.std(arr) > 0:
        grid = np.linspace(xmin, xmax, 400)
        kde = gaussian_kde(arr)
        ax.plot(grid, kde(grid), color=color, linewidth=2.0)
    ax.axvline(observed, color="#111111", linestyle="--", linewidth=1.2)
    ax.set_title(title)
    ax.grid(True, alpha=0.18)
    ax.text(
        0.98,
        0.95,
        f"obs={observed:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
    )


def plot_zero_specificity_panel(ax, values: Sequence[float]) -> None:
    arr = np.asarray([float(value) for value in values if np.isfinite(value)], dtype=np.float64)
    y_values = np.arange(1, arr.size + 1)
    ax.scatter(arr, y_values, s=4, color="#56B4E9", alpha=0.60, edgecolors="none", rasterized=True)
    ax.axvline(0.0, color="#111111", linestyle="--", linewidth=1.0)
    ax.set_title("Specificity at locked threshold", fontsize=9)
    ax.set_xlabel("Specificity")
    ax.set_ylabel("Bootstrap resample")
    ax.set_xlim(-0.05, 0.05)
    ax.set_ylim(0, max(1, arr.size + 10))
    ax.set_yticks([1, max(1, arr.size // 2), max(1, arr.size)])
    ax.grid(True, axis="y", alpha=0.18)
    ax.text(
        0.0,
        0.05,
        f"Specificity = 0.000\n{arr.size}/{arr.size} resamples",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6,
        color="#3b4a54",
    )


def make_figure(output_path: Path, distributions: Dict[str, object], observed: Dict[str, float], threshold: float, main_branch_name: str) -> None:
    plt = require_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.9), dpi=600)

    plot_density_panel(
        axes[0],
        distributions["auc"],  # type: ignore[arg-type]
        title="AUC-ROC",
        color="#E69F00",
        observed=float(observed["auc"]),
    )
    plot_zero_specificity_panel(axes[1], distributions["specificity"])  # type: ignore[arg-type]
    plot_density_panel(
        axes[2],
        distributions["brier"],  # type: ignore[arg-type]
        title="Brier score",
        color="#009E73",
        observed=float(observed["brier"]),
    )

    axes[0].set_xlabel("AUC-ROC")
    axes[0].set_ylabel("Density")
    axes[2].set_xlabel("Brier score")
    axes[2].set_ylabel("")
    for panel_label, ax in zip(["A", "B", "C"], axes):
        ax.text(-0.18, 1.05, panel_label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.20, top=0.88, wspace=0.42)
    for suffix, dpi in ((".png", 600), (".pdf", None), (".svg", None), (".tiff", 600)):
        export_path = output_path.with_suffix(suffix)
        save_kwargs = {"bbox_inches": "tight"}
        if dpi is not None:
            save_kwargs["dpi"] = dpi
        fig.savefig(export_path, **save_kwargs)
    plt.close(fig)


def main() -> None:
    assert_frozen_runtime()
    results_dir = Path(
        os.environ.get(
            "LUAD_EGFR20INS_PAPER_PACK_RESULTS_DIR",
            str(PROJECT_ROOT / "results" / "egfr20ins_paper_pack_main_sigmoid_final"),
        )
    )
    fixed_dir = results_dir / "fixed_bootstrap"
    figure_dir = results_dir / "paper_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    logger = build_logger(
        "egfr20ins_bootstrap_stability",
        figure_dir / "egfr20ins_bootstrap_stability.log",
        enable_file_logging=True,
        enable_stream_logging=True,
    )

    n_bootstrap = int(os.environ.get("LUAD_EGFR20INS_BOOTSTRAP_STABILITY_SAMPLES", "1000"))
    seed = int(os.environ.get("LUAD_EGFR20INS_BOOTSTRAP_STABILITY_SEED", "42"))
    main_branch_name, threshold, y_true, y_score, observed = load_main_branch_data(results_dir)
    distributions = bootstrap_distributions(
        y_true,
        y_score,
        threshold,
        n_bootstrap=n_bootstrap,
        random_state=seed,
    )

    save_samples_csv(figure_dir / "bootstrap_stability_samples.csv", distributions)
    save_json(
        {
            "main_branch_name": main_branch_name,
            "threshold": threshold,
            "observed": observed,
            "requested": distributions["requested"],
            "valid": distributions["valid"],
            "attempts": distributions["attempts"],
            "auc": distributions["auc_summary"],
            "specificity": distributions["specificity_summary"],
            "brier": distributions["brier_summary"],
        },
        figure_dir / "bootstrap_stability_summary.json",
    )
    make_figure(
        figure_dir / "figure_bootstrap_stability.png",
        distributions,
        observed,
        threshold,
        main_branch_name,
    )

    logger.info(
        "Bootstrap stability done | branch=%s | requested=%d | valid=%d | attempts=%d",
        main_branch_name,
        int(distributions["requested"]),
        int(distributions["valid"]),
        int(distributions["attempts"]),
    )


if __name__ == "__main__":
    main()
