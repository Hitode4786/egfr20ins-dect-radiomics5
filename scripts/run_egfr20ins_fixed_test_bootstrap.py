from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from egfr20ins_paper_common import (
    bootstrap_binary_metrics,
    branch_matrix_and_names,
    compute_binary_metrics_detailed,
    default_branch_specs,
    delong_roc_test,
    fixed_split_indices,
    format_metric_ci,
    get_fixed_eval_seed,
    get_main_branch_spec,
    load_bundle_and_feature_blocks,
    search_branch_threshold,
    fit_branch_scores,
    write_rows_csv,
)
from luad_tabular_utils import build_logger, save_json
from egfr20ins_runtime_guard import assert_frozen_runtime


def main() -> None:
    assert_frozen_runtime()
    results_dir = Path(
        os.environ.get(
            "LUAD_EGFR20INS_FIXED_BOOTSTRAP_RESULTS_DIR",
            str(PROJECT_ROOT / "results" / "egfr20ins_fixed_bootstrap"),
        )
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(
        "egfr20ins_fixed_bootstrap",
        results_dir / "egfr20ins_fixed_bootstrap.log",
        enable_file_logging=True,
        enable_stream_logging=True,
    )

    n_bootstrap = int(os.environ.get("LUAD_EGFR20INS_FIXED_BOOTSTRAP_SAMPLES", "2000"))
    branch_specs = default_branch_specs()
    main_branch_name = get_main_branch_spec().name

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
    y_fixed = y[fixed_test_indices]

    logger.info(
        "Starting EGFR20ins fixed-test bootstrap | branch_count=%d | dev=%d | fixed_test=%d | bootstrap=%d",
        len(branch_specs),
        int(dev_indices.size),
        int(fixed_test_indices.size),
        n_bootstrap,
    )

    summary_rows: List[Dict[str, object]] = []
    delong_rows: List[Dict[str, object]] = []
    detail_payload: Dict[str, object] = {
        "main_branch_name": main_branch_name,
        "n_bootstrap": n_bootstrap,
        "results": {},
        "delong": [],
    }
    patient_rows: List[Dict[str, object]] = []
    prediction_lookup: Dict[str, np.ndarray] = {}

    fixed_patient_ids = [bundle.sample_ids[int(index)] for index in fixed_test_indices.tolist()]
    for row_idx, patient_id in enumerate(fixed_patient_ids):
        patient_rows.append(
            {
                "patient_id": patient_id,
                "y_true": int(y_fixed[row_idx]),
            }
        )

    for branch in branch_specs:
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
            y_train=y_dev,
            feature_names=feature_names,
            seed=fit_seed,
        )
        scores, importance_payload = fit_branch_scores(
            branch=branch,
            x_train=x_dev,
            y_train=y_dev,
            x_eval=x_fixed,
            feature_names=feature_names,
            seed=fit_seed,
        )
        scores = np.asarray(scores, dtype=np.float64)
        prediction_lookup[branch.name] = scores
        metrics = compute_binary_metrics_detailed(y_fixed, scores, threshold)
        bootstrap = bootstrap_binary_metrics(
            y_true=y_fixed,
            positive_scores=scores,
            threshold=threshold,
            n_bootstrap=n_bootstrap,
            random_state=branch.seed,
        )
        summary_row = {
            "branch_name": branch.name,
            "display_name": branch.display_name,
            "auc": float(metrics["auc"]),
            "auc_ci": format_metric_ci(
                float(bootstrap["auc"]["value"]),
                float(bootstrap["auc"]["ci_lower"]),
                float(bootstrap["auc"]["ci_upper"]),
            ),
            "pr_auc": float(metrics["pr_auc"]),
            "pr_auc_ci": format_metric_ci(
                float(bootstrap["pr_auc"]["value"]),
                float(bootstrap["pr_auc"]["ci_lower"]),
                float(bootstrap["pr_auc"]["ci_upper"]),
            ),
            "sensitivity": float(metrics["sensitivity"]),
            "sensitivity_ci": format_metric_ci(
                float(bootstrap["sensitivity"]["value"]),
                float(bootstrap["sensitivity"]["ci_lower"]),
                float(bootstrap["sensitivity"]["ci_upper"]),
            ),
            "specificity": float(metrics["specificity"]),
            "specificity_ci": format_metric_ci(
                float(bootstrap["specificity"]["value"]),
                float(bootstrap["specificity"]["ci_lower"]),
                float(bootstrap["specificity"]["ci_upper"]),
            ),
            "ppv": float(metrics["ppv"]),
            "ppv_ci": format_metric_ci(
                float(bootstrap["ppv"]["value"]),
                float(bootstrap["ppv"]["ci_lower"]),
                float(bootstrap["ppv"]["ci_upper"]),
            ),
            "npv": float(metrics["npv"]),
            "npv_ci": format_metric_ci(
                float(bootstrap["npv"]["value"]),
                float(bootstrap["npv"]["ci_lower"]),
                float(bootstrap["npv"]["ci_upper"]),
            ),
            "f1": float(metrics["f1"]),
            "f1_ci": format_metric_ci(
                float(bootstrap["f1"]["value"]),
                float(bootstrap["f1"]["ci_lower"]),
                float(bootstrap["f1"]["ci_upper"]),
            ),
            "brier": float(metrics["brier"]),
            "brier_ci": format_metric_ci(
                float(bootstrap["brier"]["value"]),
                float(bootstrap["brier"]["ci_lower"]),
                float(bootstrap["brier"]["ci_upper"]),
            ),
            "threshold": float(threshold),
            "positives": int(metrics["positives"]),
            "negatives": int(metrics["negatives"]),
            "tp": int(metrics["tp"]),
            "tn": int(metrics["tn"]),
            "fp": int(metrics["fp"]),
            "fn": int(metrics["fn"]),
        }
        summary_rows.append(summary_row)
        detail_payload["results"][branch.name] = {
            "display_name": branch.display_name,
            "feature_block": branch.feature_block,
            "base_seed": int(branch.seed),
            "fit_seed": int(fit_seed),
            "threshold": float(threshold),
            "metrics": metrics,
            "bootstrap": bootstrap,
            "scores": scores.tolist(),
            "importance": importance_payload,
        }
        for row_idx, patient_row in enumerate(patient_rows):
            patient_row[f"{branch.name}_score"] = float(scores[row_idx])
            patient_row[f"{branch.name}_pred"] = int(scores[row_idx] >= threshold)
        logger.info(
            "Fixed test done | branch=%s | auc=%.5f | pr_auc=%.5f | sen=%.5f | spec=%.5f | threshold=%.5f",
            branch.name,
            summary_row["auc"],
            summary_row["pr_auc"],
            summary_row["sensitivity"],
            summary_row["specificity"],
            summary_row["threshold"],
        )

    main_scores = prediction_lookup.get(main_branch_name)
    if main_scores is not None:
        for branch in branch_specs:
            if branch.name == main_branch_name:
                continue
            compare_scores = prediction_lookup[branch.name]
            test_result = delong_roc_test(y_fixed, main_scores, compare_scores)
            row = {
                "main_branch_name": main_branch_name,
                "compare_branch_name": branch.name,
                "main_auc": float(test_result["auc_one"]),
                "compare_auc": float(test_result["auc_two"]),
                "z_value": float(test_result["z"]),
                "p_value": float(test_result["p"]),
            }
            delong_rows.append(row)
            detail_payload["delong"].append(row)

    summary_rows = sorted(summary_rows, key=lambda row: float(row["auc"]), reverse=True)
    write_rows_csv(
        results_dir / "fixed_test_summary.csv",
        summary_rows,
        fieldnames=[
            "branch_name",
            "display_name",
            "auc",
            "auc_ci",
            "pr_auc",
            "pr_auc_ci",
            "sensitivity",
            "sensitivity_ci",
            "specificity",
            "specificity_ci",
            "ppv",
            "ppv_ci",
            "npv",
            "npv_ci",
            "f1",
            "f1_ci",
            "brier",
            "brier_ci",
            "threshold",
            "positives",
            "negatives",
            "tp",
            "tn",
            "fp",
            "fn",
        ],
    )
    write_rows_csv(
        results_dir / "fixed_test_predictions.csv",
        patient_rows,
        fieldnames=list(patient_rows[0].keys()) if patient_rows else ["patient_id", "y_true"],
    )
    write_rows_csv(
        results_dir / "fixed_test_delong_comparisons.csv",
        delong_rows,
        fieldnames=["main_branch_name", "compare_branch_name", "main_auc", "compare_auc", "z_value", "p_value"],
    )
    save_json(detail_payload, results_dir / "fixed_test_bootstrap_detail.json")
    logger.info("EGFR20ins fixed-test bootstrap finished | summary=%s", results_dir / "fixed_test_summary.csv")


if __name__ == "__main__":
    main()
