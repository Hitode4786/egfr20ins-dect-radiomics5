from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import chi2
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from egfr20ins_paper_common import (
    branch_matrix_and_names,
    default_branch_specs,
    fixed_split_indices,
    get_fixed_eval_seed,
    load_bundle_and_feature_blocks,
    search_branch_threshold,
    fit_branch_scores,
    write_rows_csv,
)
from luad_tabular_utils import build_logger, save_json
from egfr20ins_runtime_guard import assert_frozen_runtime


def try_import_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def load_prediction_csv(path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig")))
    y_true = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int64)
    score_map: Dict[str, np.ndarray] = {}
    if not rows:
        return y_true, score_map
    for key in rows[0].keys():
        if key.endswith("_score"):
            score_map[key[: -len("_score")]] = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return y_true, score_map


def ensure_fixed_predictions(logger) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    fixed_results_dir = Path(
        os.environ.get(
            "LUAD_EGFR20INS_FIXED_BOOTSTRAP_RESULTS_DIR",
            str(PROJECT_ROOT / "results" / "egfr20ins_fixed_bootstrap"),
        )
    )
    predictions_path = fixed_results_dir / "fixed_test_predictions.csv"
    if predictions_path.is_file():
        logger.info("Reusing fixed test predictions | path=%s", predictions_path)
        return load_prediction_csv(predictions_path)

    logger.info("Fixed prediction file missing; rebuilding fixed predictions in-process.")
    (
        bundle,
        x_radiomics,
        x_clinical,
        x_combined,
        radiomics_feature_names,
        clinical_feature_names,
        combined_feature_names,
    ) = load_bundle_and_feature_blocks()
    y = np.asarray(bundle.y, dtype=np.int64)
    dev_indices, fixed_test_indices = fixed_split_indices(bundle)
    y_fixed = y[fixed_test_indices]

    score_map: Dict[str, np.ndarray] = {}
    for branch in default_branch_specs():
        fit_seed = get_fixed_eval_seed(branch.seed)
        x_all, feature_names = branch_matrix_and_names(
            branch,
            x_radiomics=x_radiomics,
            x_clinical=x_clinical,
            x_combined=x_combined,
            radiomics_feature_names=radiomics_feature_names,
            clinical_feature_names=clinical_feature_names,
            combined_feature_names=combined_feature_names,
        )
        x_dev = x_all[dev_indices]
        x_fixed = x_all[fixed_test_indices]
        threshold = search_branch_threshold(
            branch=branch,
            x_train=x_dev,
            y_train=y[dev_indices],
            feature_names=feature_names,
            seed=fit_seed,
        )
        scores, _ = fit_branch_scores(
            branch=branch,
            x_train=x_dev,
            y_train=y[dev_indices],
            x_eval=x_fixed,
            feature_names=feature_names,
            seed=fit_seed,
        )
        score_map[branch.name] = np.asarray(scores, dtype=np.float64)
        logger.info("Rebuilt fixed predictions | branch=%s | fit_seed=%d | threshold=%.5f", branch.name, fit_seed, threshold)
    return y_fixed, score_map


def compute_dca(y_true: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray) -> Dict[str, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    n = max(1, y_true.size)
    prevalence = float(np.mean(y_true))
    model_net_benefit = np.zeros_like(thresholds, dtype=np.float64)
    treat_all_net_benefit = np.zeros_like(thresholds, dtype=np.float64)
    treat_none_net_benefit = np.zeros_like(thresholds, dtype=np.float64)
    for idx, threshold in enumerate(thresholds):
        predictions = (probabilities >= threshold).astype(np.int64)
        tp = int(np.sum((predictions == 1) & (y_true == 1)))
        fp = int(np.sum((predictions == 1) & (y_true == 0)))
        odds = threshold / max(1e-8, 1.0 - threshold)
        model_net_benefit[idx] = tp / n - fp / n * odds
        treat_all_net_benefit[idx] = prevalence - (1.0 - prevalence) * odds
        treat_none_net_benefit[idx] = 0.0
    return {
        "thresholds": thresholds,
        "model_net_benefit": model_net_benefit,
        "treat_all_net_benefit": treat_all_net_benefit,
        "treat_none_net_benefit": treat_none_net_benefit,
    }


def hosmer_lemeshow_test(y_true: np.ndarray, probabilities: np.ndarray, groups: int) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y_true.size == 0 or probabilities.size != y_true.size:
        return {"statistic": float("nan"), "df": float("nan"), "p_value": float("nan"), "groups_used": 0.0}

    requested_groups = max(2, int(groups))
    quantile_edges = np.quantile(probabilities, np.linspace(0.0, 1.0, requested_groups + 1))
    unique_edges = np.unique(np.asarray(quantile_edges, dtype=np.float64))
    if unique_edges.size < 3:
        return {"statistic": float("nan"), "df": float("nan"), "p_value": float("nan"), "groups_used": 0.0}

    bin_indices = np.digitize(probabilities, unique_edges[1:-1], right=True)
    hl_statistic = 0.0
    valid_groups = 0
    for group_idx in range(int(bin_indices.max()) + 1):
        group_mask = bin_indices == group_idx
        if not np.any(group_mask):
            continue
        group_true = y_true[group_mask]
        group_prob = probabilities[group_mask]
        observed = float(group_true.sum())
        expected = float(group_prob.sum())
        n_group = int(group_true.size)
        expected_negative = float(n_group - expected)
        if expected <= 0.0 or expected_negative <= 0.0:
            continue
        observed_negative = float(n_group - observed)
        hl_statistic += ((observed - expected) ** 2) / expected
        hl_statistic += ((observed_negative - expected_negative) ** 2) / expected_negative
        valid_groups += 1

    if valid_groups < 3:
        return {"statistic": float("nan"), "df": float("nan"), "p_value": float("nan"), "groups_used": float(valid_groups)}

    hl_df = float(valid_groups - 2)
    p_value = float(1.0 - chi2.cdf(hl_statistic, df=hl_df))
    return {
        "statistic": float(hl_statistic),
        "df": hl_df,
        "p_value": p_value,
        "groups_used": float(valid_groups),
    }


def plot_outputs(
    results_dir: Path,
    calibration_point_rows: List[Dict[str, object]],
    dca_rows: List[Dict[str, object]],
) -> None:
    plt = try_import_plotting()
    if plt is None:
        return

    calibration_by_branch: Dict[str, List[Dict[str, object]]] = {}
    for row in calibration_point_rows:
        calibration_by_branch.setdefault(str(row["branch_name"]), []).append(row)
    if calibration_by_branch:
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        for branch_name, rows in calibration_by_branch.items():
            rows_sorted = sorted(rows, key=lambda item: float(item["prob_pred"]))
            ax.plot(
                [float(row["prob_pred"]) for row in rows_sorted],
                [float(row["prob_true"]) for row in rows_sorted],
                marker="o",
                linewidth=2,
                label=branch_name,
            )
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Frequency")
        ax.set_title("EGFR20ins Calibration")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(results_dir / "calibration_curve.png", dpi=300)
        plt.close(fig)

    dca_by_branch: Dict[str, List[Dict[str, object]]] = {}
    for row in dca_rows:
        dca_by_branch.setdefault(str(row["branch_name"]), []).append(row)
    if dca_by_branch:
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        for branch_name, rows in dca_by_branch.items():
            rows_sorted = sorted(rows, key=lambda item: float(item["threshold"]))
            ax.plot(
                [float(row["threshold"]) for row in rows_sorted],
                [float(row["model_net_benefit"]) for row in rows_sorted],
                linewidth=2,
                label=branch_name,
            )
        first_rows = sorted(next(iter(dca_by_branch.values())), key=lambda item: float(item["threshold"]))
        ax.plot(
            [float(row["threshold"]) for row in first_rows],
            [float(row["treat_all_net_benefit"]) for row in first_rows],
            linestyle="--",
            linewidth=1.5,
            color="gray",
            label="Treat All",
        )
        ax.plot(
            [float(row["threshold"]) for row in first_rows],
            [0.0 for _ in first_rows],
            linestyle=":",
            linewidth=1.2,
            color="black",
            label="Treat None",
        )
        ax.set_xlabel("Threshold Probability")
        ax.set_ylabel("Net Benefit")
        ax.set_title("EGFR20ins Decision Curve Analysis")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(results_dir / "decision_curve.png", dpi=300)
        plt.close(fig)


def main() -> None:
    assert_frozen_runtime()
    results_dir = Path(
        os.environ.get(
            "LUAD_EGFR20INS_CALIBRATION_DCA_RESULTS_DIR",
            str(PROJECT_ROOT / "results" / "egfr20ins_calibration_dca"),
        )
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(
        "egfr20ins_calibration_dca",
        results_dir / "egfr20ins_calibration_dca.log",
        enable_file_logging=True,
        enable_stream_logging=True,
    )

    y_true, score_map = ensure_fixed_predictions(logger)
    bin_count = int(os.environ.get("LUAD_EGFR20INS_CALIBRATION_BINS", "12"))
    hl_groups = int(os.environ.get("LUAD_EGFR20INS_HL_GROUPS", "10"))
    threshold_grid = np.linspace(
        float(os.environ.get("LUAD_EGFR20INS_DCA_THRESHOLD_MIN", "0.01")),
        float(os.environ.get("LUAD_EGFR20INS_DCA_THRESHOLD_MAX", "0.50")),
        int(os.environ.get("LUAD_EGFR20INS_DCA_THRESHOLD_POINTS", "50")),
    )

    calibration_summary_rows: List[Dict[str, object]] = []
    calibration_point_rows: List[Dict[str, object]] = []
    dca_rows: List[Dict[str, object]] = []
    detail_payload: Dict[str, object] = {"calibration": {}, "dca": {}}

    for branch_name, probabilities in score_map.items():
        auc = float(roc_auc_score(y_true, probabilities)) if np.unique(y_true).size >= 2 else float("nan")
        pr_auc = float(average_precision_score(y_true, probabilities)) if np.unique(y_true).size >= 2 else float("nan")
        brier = float(brier_score_loss(y_true, probabilities))
        hl_result = hosmer_lemeshow_test(y_true, probabilities, hl_groups)
        prob_true, prob_pred = calibration_curve(y_true, probabilities, n_bins=bin_count, strategy="quantile")
        calibration_summary_rows.append(
            {
                "branch_name": branch_name,
                "auc": auc,
                "pr_auc": pr_auc,
                "brier": brier,
                "bin_count": int(prob_true.size),
                "hl_statistic": float(hl_result["statistic"]),
                "hl_df": float(hl_result["df"]),
                "hl_p_value": float(hl_result["p_value"]),
                "hl_groups_used": int(hl_result["groups_used"]),
            }
        )
        for point_idx, (true_value, pred_value) in enumerate(zip(prob_true.tolist(), prob_pred.tolist())):
            calibration_point_rows.append(
                {
                    "branch_name": branch_name,
                    "bin_idx": int(point_idx),
                    "prob_true": float(true_value),
                    "prob_pred": float(pred_value),
                }
            )
        dca = compute_dca(y_true, probabilities, threshold_grid)
        for row_idx, threshold in enumerate(dca["thresholds"].tolist()):
            dca_rows.append(
                {
                    "branch_name": branch_name,
                    "threshold": float(threshold),
                    "model_net_benefit": float(dca["model_net_benefit"][row_idx]),
                    "treat_all_net_benefit": float(dca["treat_all_net_benefit"][row_idx]),
                    "treat_none_net_benefit": float(dca["treat_none_net_benefit"][row_idx]),
                }
            )
        detail_payload["calibration"][branch_name] = {
            "auc": auc,
            "pr_auc": pr_auc,
            "brier": brier,
            "hosmer_lemeshow": hl_result,
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        }
        detail_payload["dca"][branch_name] = {
            "thresholds": dca["thresholds"].tolist(),
            "model_net_benefit": dca["model_net_benefit"].tolist(),
            "treat_all_net_benefit": dca["treat_all_net_benefit"].tolist(),
            "treat_none_net_benefit": dca["treat_none_net_benefit"].tolist(),
        }
        logger.info(
            "Calibration/DCA | branch=%s | auc=%.5f | pr_auc=%.5f | brier=%.5f",
            branch_name,
            auc,
            pr_auc,
            brier,
        )

    write_rows_csv(
        results_dir / "calibration_summary.csv",
        calibration_summary_rows,
        fieldnames=[
            "branch_name",
            "auc",
            "pr_auc",
            "brier",
            "bin_count",
            "hl_statistic",
            "hl_df",
            "hl_p_value",
            "hl_groups_used",
        ],
    )
    write_rows_csv(
        results_dir / "calibration_curve_points.csv",
        calibration_point_rows,
        fieldnames=["branch_name", "bin_idx", "prob_true", "prob_pred"],
    )
    write_rows_csv(
        results_dir / "dca_summary.csv",
        dca_rows,
        fieldnames=["branch_name", "threshold", "model_net_benefit", "treat_all_net_benefit", "treat_none_net_benefit"],
    )
    save_json(detail_payload, results_dir / "calibration_dca_detail.json")
    plot_outputs(results_dir, calibration_point_rows, dca_rows)
    logger.info("EGFR20ins calibration/DCA finished | summary=%s", results_dir / "calibration_summary.csv")


if __name__ == "__main__":
    main()
