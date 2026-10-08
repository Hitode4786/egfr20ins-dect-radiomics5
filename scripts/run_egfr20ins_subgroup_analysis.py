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
    build_repeated_oof_predictions,
    compute_binary_metrics_detailed,
    format_metric_ci,
    get_main_branch_spec,
    load_bundle_and_feature_blocks,
    bundle_metadata_array,
    subgroup_rows_from_clinical,
    write_rows_csv,
)
from luad_tabular_utils import build_logger, save_json
from egfr20ins_runtime_guard import assert_frozen_runtime


def parse_repeat_seeds() -> List[int]:
    raw_value = os.environ.get("LUAD_EGFR20INS_SUBGROUP_OOF_SEEDS", "42,52,62,72,82")
    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]


def main() -> None:
    assert_frozen_runtime()
    results_dir = Path(
        os.environ.get("LUAD_EGFR20INS_SUBGROUP_RESULTS_DIR", str(PROJECT_ROOT / "results" / "egfr20ins_subgroups"))
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(
        "egfr20ins_subgroups",
        results_dir / "egfr20ins_subgroups.log",
        enable_file_logging=True,
        enable_stream_logging=True,
    )

    repeat_seeds = parse_repeat_seeds()
    n_splits = int(os.environ.get("LUAD_EGFR20INS_SUBGROUP_OOF_SPLITS", "5"))
    n_bootstrap = int(os.environ.get("LUAD_EGFR20INS_SUBGROUP_BOOTSTRAP_SAMPLES", "2000"))
    branch = get_main_branch_spec()

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
    x_all, feature_names = branch_matrix_and_names(
        branch,
        x_radiomics=x_radiomics,
        x_clinical=x_clinical,
        x_combined=x_combined,
        radiomics_feature_names=radiomics_feature_names,
        clinical_feature_names=clinical_feature_names,
        combined_feature_names=combined_feature_names,
    )

    logger.info(
        "Starting EGFR20ins subgroup analysis | branch=%s | samples=%d | repeat_seeds=%s | splits=%d",
        branch.name,
        int(y.shape[0]),
        repeat_seeds,
        n_splits,
    )

    oof_payload = build_repeated_oof_predictions(
        branch=branch,
        x=x_all,
        y=y,
        feature_names=feature_names,
        repeat_seeds=repeat_seeds,
        n_splits=n_splits,
        logger=logger,
    )
    scores = np.asarray(oof_payload["scores"], dtype=np.float64)
    threshold = float(oof_payload["global_threshold"])

    subgroup_rows = subgroup_rows_from_clinical(bundle)
    summary_rows: List[Dict[str, object]] = []
    patient_rows: List[Dict[str, object]] = []
    metastasis_all = bundle_metadata_array(bundle, "metastasis", dtype=np.int64)
    raw_location_all = bundle_metadata_array(bundle, "location_raw")
    for sample_idx, patient_id in enumerate(bundle.sample_ids):
        patient_rows.append(
            {
                "patient_id": patient_id,
                "y_true": int(y[sample_idx]),
                "oof_score": float(scores[sample_idx]),
                "oof_pred": int(scores[sample_idx] >= threshold) if np.isfinite(scores[sample_idx]) else "",
                "age": float(bundle.x_clinical[sample_idx, 0]),
                "sex": int(bundle.x_clinical[sample_idx, 1]),
                "smoking_history": int(bundle.x_clinical[sample_idx, 2]),
                "location_code": int(bundle.x_clinical[sample_idx, 3]),
                "location_raw": "" if raw_location_all is None else str(raw_location_all[sample_idx]),
                "metastasis": "" if metastasis_all is None else int(metastasis_all[sample_idx]),
            }
        )

    for subgroup in subgroup_rows:
        subgroup_indices = np.asarray(subgroup["indices"], dtype=np.int64)
        if subgroup_indices.size == 0:
            continue
        subgroup_true = y[subgroup_indices]
        subgroup_scores = scores[subgroup_indices]
        metrics = compute_binary_metrics_detailed(subgroup_true, subgroup_scores, threshold)
        bootstrap = bootstrap_binary_metrics(
            y_true=subgroup_true,
            positive_scores=subgroup_scores,
            threshold=threshold,
            n_bootstrap=n_bootstrap,
            random_state=int(len(summary_rows) + 1000),
        )
        row = {
            "subgroup_name": subgroup["subgroup_name"],
            "subgroup_value": subgroup["subgroup_value"],
            "n": int(subgroup_indices.size),
            "positives": int(subgroup_true.sum()),
            "negatives": int((subgroup_true == 0).sum()),
            "auc": float(metrics["auc"]),
            "auc_ci_lower": float(bootstrap["auc"]["ci_lower"]),
            "auc_ci_upper": float(bootstrap["auc"]["ci_upper"]),
            "auc_ci": format_metric_ci(
                float(bootstrap["auc"]["value"]),
                float(bootstrap["auc"]["ci_lower"]),
                float(bootstrap["auc"]["ci_upper"]),
            ),
            "pr_auc": float(metrics["pr_auc"]),
            "pr_auc_ci_lower": float(bootstrap["pr_auc"]["ci_lower"]),
            "pr_auc_ci_upper": float(bootstrap["pr_auc"]["ci_upper"]),
            "pr_auc_ci": format_metric_ci(
                float(bootstrap["pr_auc"]["value"]),
                float(bootstrap["pr_auc"]["ci_lower"]),
                float(bootstrap["pr_auc"]["ci_upper"]),
            ),
            "sensitivity": float(metrics["sensitivity"]),
            "sensitivity_ci_lower": float(bootstrap["sensitivity"]["ci_lower"]),
            "sensitivity_ci_upper": float(bootstrap["sensitivity"]["ci_upper"]),
            "sensitivity_ci": format_metric_ci(
                float(bootstrap["sensitivity"]["value"]),
                float(bootstrap["sensitivity"]["ci_lower"]),
                float(bootstrap["sensitivity"]["ci_upper"]),
            ),
            "specificity": float(metrics["specificity"]),
            "specificity_ci_lower": float(bootstrap["specificity"]["ci_lower"]),
            "specificity_ci_upper": float(bootstrap["specificity"]["ci_upper"]),
            "specificity_ci": format_metric_ci(
                float(bootstrap["specificity"]["value"]),
                float(bootstrap["specificity"]["ci_lower"]),
                float(bootstrap["specificity"]["ci_upper"]),
            ),
            "ppv": float(metrics["ppv"]),
            "ppv_ci_lower": float(bootstrap["ppv"]["ci_lower"]),
            "ppv_ci_upper": float(bootstrap["ppv"]["ci_upper"]),
            "ppv_ci": format_metric_ci(
                float(bootstrap["ppv"]["value"]),
                float(bootstrap["ppv"]["ci_lower"]),
                float(bootstrap["ppv"]["ci_upper"]),
            ),
            "npv": float(metrics["npv"]),
            "npv_ci_lower": float(bootstrap["npv"]["ci_lower"]),
            "npv_ci_upper": float(bootstrap["npv"]["ci_upper"]),
            "npv_ci": format_metric_ci(
                float(bootstrap["npv"]["value"]),
                float(bootstrap["npv"]["ci_lower"]),
                float(bootstrap["npv"]["ci_upper"]),
            ),
            "f1": float(metrics["f1"]),
            "f1_ci_lower": float(bootstrap["f1"]["ci_lower"]),
            "f1_ci_upper": float(bootstrap["f1"]["ci_upper"]),
            "f1_ci": format_metric_ci(
                float(bootstrap["f1"]["value"]),
                float(bootstrap["f1"]["ci_lower"]),
                float(bootstrap["f1"]["ci_upper"]),
            ),
            "threshold": float(threshold),
            "bootstrap_valid_samples": int(bootstrap["valid_bootstrap_samples"]),
        }
        summary_rows.append(row)
        logger.info(
            "Subgroup | %s=%s | n=%d | pos=%d | auc=%.5f | pr_auc=%.5f",
            row["subgroup_name"],
            row["subgroup_value"],
            row["n"],
            row["positives"],
            row["auc"],
            row["pr_auc"],
        )

    write_rows_csv(
        results_dir / "subgroup_summary.csv",
        summary_rows,
        fieldnames=[
            "subgroup_name",
            "subgroup_value",
            "n",
            "positives",
            "negatives",
            "auc",
            "auc_ci_lower",
            "auc_ci_upper",
            "auc_ci",
            "pr_auc",
            "pr_auc_ci_lower",
            "pr_auc_ci_upper",
            "pr_auc_ci",
            "sensitivity",
            "sensitivity_ci_lower",
            "sensitivity_ci_upper",
            "sensitivity_ci",
            "specificity",
            "specificity_ci_lower",
            "specificity_ci_upper",
            "specificity_ci",
            "ppv",
            "ppv_ci_lower",
            "ppv_ci_upper",
            "ppv_ci",
            "npv",
            "npv_ci_lower",
            "npv_ci_upper",
            "npv_ci",
            "f1",
            "f1_ci_lower",
            "f1_ci_upper",
            "f1_ci",
            "threshold",
            "bootstrap_valid_samples",
        ],
    )
    write_rows_csv(
        results_dir / "subgroup_oof_predictions.csv",
        patient_rows,
        fieldnames=[
            "patient_id",
            "y_true",
            "oof_score",
            "oof_pred",
            "age",
            "sex",
            "smoking_history",
            "location_code",
            "location_raw",
            "metastasis",
        ],
    )
    save_json(
        {
            "branch_name": branch.name,
            "branch_display_name": branch.display_name,
            "repeat_seeds": repeat_seeds,
            "n_splits": n_splits,
            "n_bootstrap": n_bootstrap,
            "global_threshold": threshold,
            "oof_payload": oof_payload,
        },
        results_dir / "subgroup_detail.json",
    )
    logger.info("EGFR20ins subgroup analysis finished | summary=%s", results_dir / "subgroup_summary.csv")


if __name__ == "__main__":
    main()
