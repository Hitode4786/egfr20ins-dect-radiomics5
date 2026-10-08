from __future__ import annotations

from collections import Counter
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from egfr20ins_paper_common import (
    branch_matrix_and_names,
    compute_binary_metrics_detailed,
    default_branch_specs,
    fixed_split_indices,
    load_bundle_and_feature_blocks,
    search_branch_threshold,
    fit_branch_scores,
    summarize_metric_series,
    write_rows_csv,
)
from luad_tabular_utils import build_logger, save_json
from egfr20ins_runtime_guard import assert_frozen_runtime


def parse_repeat_seeds() -> List[int]:
    raw_value = os.environ.get("LUAD_EGFR20INS_REPEATED_CV_SEEDS", "42,52,62,72,82")
    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]


def main() -> None:
    assert_frozen_runtime()
    results_dir = Path(
        os.environ.get("LUAD_EGFR20INS_REPEATED_CV_RESULTS_DIR", str(PROJECT_ROOT / "results" / "egfr20ins_repeated_cv"))
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(
        "egfr20ins_repeated_cv",
        results_dir / "egfr20ins_repeated_cv.log",
        enable_file_logging=True,
        enable_stream_logging=True,
    )

    n_splits = int(os.environ.get("LUAD_EGFR20INS_REPEATED_CV_SPLITS", "5"))
    repeat_seeds = parse_repeat_seeds()
    branch_specs = default_branch_specs()

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
    y_dev = y[dev_indices]

    logger.info(
        "Starting EGFR20ins repeated CV | branch_count=%d | dev_samples=%d | fixed_test=%d | splits=%d | repeat_seeds=%s",
        len(branch_specs),
        int(dev_indices.size),
        int(fixed_test_indices.size),
        n_splits,
        repeat_seeds,
    )

    fold_rows: List[Dict[str, object]] = []
    selected_feature_rows: List[Dict[str, object]] = []
    feature_stability_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    detail_payload: Dict[str, object] = {
        "n_splits": n_splits,
        "repeat_seeds": repeat_seeds,
        "results": {},
    }

    for branch in branch_specs:
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

        branch_metric_values: Dict[str, List[float]] = {
            "auc": [],
            "pr_auc": [],
            "sensitivity": [],
            "specificity": [],
            "f1": [],
            "brier": [],
            "threshold": [],
        }
        branch_rows: List[Dict[str, object]] = []
        branch_feature_counts: Counter[str] = Counter()

        for repeat_idx, repeat_seed in enumerate(repeat_seeds):
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(repeat_seed))
            for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(splitter.split(x_dev, y_dev)):
                threshold = search_branch_threshold(
                    branch=branch,
                    x_train=x_dev[fold_train_idx],
                    y_train=y_dev[fold_train_idx],
                    feature_names=feature_names,
                    seed=int(repeat_seed) + fold_idx,
                )
                scores, importance_payload = fit_branch_scores(
                    branch=branch,
                    x_train=x_dev[fold_train_idx],
                    y_train=y_dev[fold_train_idx],
                    x_eval=x_dev[fold_val_idx],
                    feature_names=feature_names,
                    seed=int(repeat_seed) + fold_idx,
                )
                metrics = compute_binary_metrics_detailed(y_dev[fold_val_idx], scores, threshold)
                row = {
                    "branch_name": branch.name,
                    "display_name": branch.display_name,
                    "repeat": int(repeat_idx),
                    "repeat_seed": int(repeat_seed),
                    "fold": int(fold_idx),
                    "auc": float(metrics["auc"]),
                    "pr_auc": float(metrics["pr_auc"]),
                    "sensitivity": float(metrics["sensitivity"]),
                    "specificity": float(metrics["specificity"]),
                    "f1": float(metrics["f1"]),
                    "brier": float(metrics["brier"]),
                    "threshold": float(metrics["threshold"]),
                }
                fold_rows.append(row)
                branch_rows.append(row)
                for metric_name in branch_metric_values:
                    branch_metric_values[metric_name].append(float(row[metric_name]))
                if isinstance(importance_payload, dict):
                    selected_names = importance_payload.get("selected_feature_names", [])
                    if isinstance(selected_names, list):
                        for feature_name in selected_names:
                            feature_name = str(feature_name)
                            branch_feature_counts[feature_name] += 1
                            selected_feature_rows.append(
                                {
                                    "branch_name": branch.name,
                                    "display_name": branch.display_name,
                                    "repeat": int(repeat_idx),
                                    "repeat_seed": int(repeat_seed),
                                    "fold": int(fold_idx),
                                    "feature_name": feature_name,
                                }
                            )
                logger.info(
                    "Repeated CV | branch=%s | repeat=%d | fold=%d | auc=%.5f | pr_auc=%.5f | sen=%.5f | spec=%.5f | threshold=%.5f",
                    branch.name,
                    repeat_idx,
                    fold_idx,
                    row["auc"],
                    row["pr_auc"],
                    row["sensitivity"],
                    row["specificity"],
                    row["threshold"],
                )

        summary_row: Dict[str, object] = {
            "branch_name": branch.name,
            "display_name": branch.display_name,
            "fold_evaluations": len(branch_rows),
        }
        for metric_name, values in branch_metric_values.items():
            stats = summarize_metric_series(values)
            summary_row[f"{metric_name}_mean"] = stats["mean"]
            summary_row[f"{metric_name}_std"] = stats["std"]
            summary_row[f"{metric_name}_ci_lower"] = stats["ci_lower"]
            summary_row[f"{metric_name}_ci_upper"] = stats["ci_upper"]
        summary_rows.append(summary_row)
        branch_total_evaluations = max(1, len(branch_rows))
        for feature_name, selection_count in branch_feature_counts.items():
            feature_stability_rows.append(
                {
                    "branch_name": branch.name,
                    "display_name": branch.display_name,
                    "feature_name": str(feature_name),
                    "selection_count": int(selection_count),
                    "selection_rate": float(selection_count / branch_total_evaluations),
                    "total_fold_evaluations": int(branch_total_evaluations),
                }
            )
        detail_payload["results"][branch.name] = {
            "summary": summary_row,
            "fold_rows": branch_rows,
            "feature_selection_counts": dict(sorted(branch_feature_counts.items(), key=lambda item: (-item[1], item[0]))),
        }
        logger.info(
            "Repeated CV done | branch=%s | auc=%.5f | pr_auc=%.5f | sen=%.5f | spec=%.5f",
            branch.name,
            summary_row["auc_mean"],
            summary_row["pr_auc_mean"],
            summary_row["sensitivity_mean"],
            summary_row["specificity_mean"],
        )

    summary_rows = sorted(summary_rows, key=lambda row: float(row["auc_mean"]), reverse=True)
    write_rows_csv(
        results_dir / "repeated_cv_folds.csv",
        fold_rows,
        fieldnames=[
            "branch_name",
            "display_name",
            "repeat",
            "repeat_seed",
            "fold",
            "auc",
            "pr_auc",
            "sensitivity",
            "specificity",
            "f1",
            "brier",
            "threshold",
        ],
    )
    write_rows_csv(
        results_dir / "repeated_cv_summary.csv",
        summary_rows,
        fieldnames=[
            "branch_name",
            "display_name",
            "fold_evaluations",
            "auc_mean",
            "auc_std",
            "auc_ci_lower",
            "auc_ci_upper",
            "pr_auc_mean",
            "pr_auc_std",
            "pr_auc_ci_lower",
            "pr_auc_ci_upper",
            "sensitivity_mean",
            "sensitivity_std",
            "sensitivity_ci_lower",
            "sensitivity_ci_upper",
            "specificity_mean",
            "specificity_std",
            "specificity_ci_lower",
            "specificity_ci_upper",
            "f1_mean",
            "f1_std",
            "f1_ci_lower",
            "f1_ci_upper",
            "brier_mean",
            "brier_std",
            "brier_ci_lower",
            "brier_ci_upper",
            "threshold_mean",
            "threshold_std",
            "threshold_ci_lower",
            "threshold_ci_upper",
        ],
    )
    write_rows_csv(
        results_dir / "repeated_cv_selected_features.csv",
        selected_feature_rows,
        fieldnames=["branch_name", "display_name", "repeat", "repeat_seed", "fold", "feature_name"],
    )
    feature_stability_rows = sorted(
        feature_stability_rows,
        key=lambda row: (
            str(row["branch_name"]),
            -int(row["selection_count"]),
            str(row["feature_name"]),
        ),
    )
    write_rows_csv(
        results_dir / "feature_stability_summary.csv",
        feature_stability_rows,
        fieldnames=[
            "branch_name",
            "display_name",
            "feature_name",
            "selection_count",
            "selection_rate",
            "total_fold_evaluations",
        ],
    )
    save_json(detail_payload, results_dir / "repeated_cv_detail.json")
    logger.info("EGFR20ins repeated CV finished | summary=%s", results_dir / "repeated_cv_summary.csv")


if __name__ == "__main__":
    main()
