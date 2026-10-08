from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from egfr20ins_paper_common import default_branch_specs, format_metric_ci, get_main_branch_spec, read_json, write_rows_csv
from luad_tabular_utils import build_logger, save_json
from egfr20ins_runtime_guard import assert_frozen_runtime


def read_rows_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def require_paths(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required paper-pack inputs:\n" + "\n".join(missing))


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def formatted_from_repeat_row(row: Dict[str, str], metric_name: str) -> str:
    return format_metric_ci(
        safe_float(row.get(f"{metric_name}_mean")),
        safe_float(row.get(f"{metric_name}_ci_lower")),
        safe_float(row.get(f"{metric_name}_ci_upper")),
    )


def find_nearest_net_benefit(rows: Sequence[Dict[str, str]], target_threshold: float) -> float:
    if not rows:
        return float("nan")
    nearest_row = min(rows, key=lambda row: abs(safe_float(row.get("threshold")) - target_threshold))
    return safe_float(nearest_row.get("model_net_benefit"))


def max_net_benefit(rows: Sequence[Dict[str, str]], low: float, high: float) -> float:
    values = [
        safe_float(row.get("model_net_benefit"))
        for row in rows
        if low <= safe_float(row.get("threshold")) <= high and np.isfinite(safe_float(row.get("model_net_benefit")))
    ]
    return float(np.max(values)) if values else float("nan")


def build_model_table(
    repeated_rows: Sequence[Dict[str, str]],
    fixed_rows: Sequence[Dict[str, str]],
    calibration_rows: Sequence[Dict[str, str]],
    delong_rows: Sequence[Dict[str, str]],
) -> List[Dict[str, object]]:
    repeated_map = {str(row["branch_name"]): row for row in repeated_rows}
    fixed_map = {str(row["branch_name"]): row for row in fixed_rows}
    calibration_map = {str(row["branch_name"]): row for row in calibration_rows}
    delong_map = {str(row["compare_branch_name"]): row for row in delong_rows}
    main_branch_name = get_main_branch_spec().name

    rows: List[Dict[str, object]] = []
    for branch in default_branch_specs():
        repeated_row = repeated_map.get(branch.name, {})
        fixed_row = fixed_map.get(branch.name, {})
        calibration_row = calibration_map.get(branch.name, {})
        delong_row = delong_map.get(branch.name, {})
        row = {
            "branch_name": branch.name,
            "display_name": branch.display_name,
            "feature_block": branch.feature_block,
            "model_kind": branch.model_kind,
            "is_main_model": int(branch.name == main_branch_name),
            "cv_auc": formatted_from_repeat_row(repeated_row, "auc") if repeated_row else "",
            "cv_pr_auc": formatted_from_repeat_row(repeated_row, "pr_auc") if repeated_row else "",
            "cv_sensitivity": formatted_from_repeat_row(repeated_row, "sensitivity") if repeated_row else "",
            "cv_specificity": formatted_from_repeat_row(repeated_row, "specificity") if repeated_row else "",
            "fixed_test_auc": fixed_row.get("auc_ci", ""),
            "fixed_test_pr_auc": fixed_row.get("pr_auc_ci", ""),
            "fixed_test_sensitivity": fixed_row.get("sensitivity_ci", ""),
            "fixed_test_specificity": fixed_row.get("specificity_ci", ""),
            "fixed_test_ppv": fixed_row.get("ppv_ci", ""),
            "fixed_test_npv": fixed_row.get("npv_ci", ""),
            "fixed_test_f1": fixed_row.get("f1_ci", ""),
            "fixed_test_brier": fixed_row.get("brier_ci", ""),
            "threshold": safe_float(fixed_row.get("threshold")),
            "calibration_brier": safe_float(calibration_row.get("brier")),
            "calibration_hl_statistic": safe_float(calibration_row.get("hl_statistic")),
            "calibration_hl_df": safe_float(calibration_row.get("hl_df")),
            "calibration_hl_p_value": safe_float(calibration_row.get("hl_p_value")),
            "delong_p_vs_main": "" if branch.name == main_branch_name else safe_float(delong_row.get("p_value")),
            "cv_auc_mean": safe_float(repeated_row.get("auc_mean")),
            "cv_auc_std": safe_float(repeated_row.get("auc_std")),
            "cv_pr_auc_mean": safe_float(repeated_row.get("pr_auc_mean")),
            "cv_sensitivity_mean": safe_float(repeated_row.get("sensitivity_mean")),
            "cv_specificity_mean": safe_float(repeated_row.get("specificity_mean")),
            "fixed_auc_value": safe_float(fixed_row.get("auc")),
            "fixed_pr_auc_value": safe_float(fixed_row.get("pr_auc")),
            "fixed_sensitivity_value": safe_float(fixed_row.get("sensitivity")),
            "fixed_specificity_value": safe_float(fixed_row.get("specificity")),
            "fixed_ppv_value": safe_float(fixed_row.get("ppv")),
            "fixed_npv_value": safe_float(fixed_row.get("npv")),
            "fixed_f1_value": safe_float(fixed_row.get("f1")),
        }
        rows.append(row)
    return rows


def build_calibration_dca_table(
    calibration_rows: Sequence[Dict[str, str]],
    dca_rows: Sequence[Dict[str, str]],
) -> List[Dict[str, object]]:
    dca_by_branch: Dict[str, List[Dict[str, str]]] = {}
    for row in dca_rows:
        dca_by_branch.setdefault(str(row["branch_name"]), []).append(row)

    output_rows: List[Dict[str, object]] = []
    for branch in default_branch_specs():
        calibration_row = next((row for row in calibration_rows if str(row["branch_name"]) == branch.name), None)
        if calibration_row is None:
            continue
        branch_dca_rows = dca_by_branch.get(branch.name, [])
        output_rows.append(
            {
                "branch_name": branch.name,
                "display_name": branch.display_name,
                "auc": safe_float(calibration_row.get("auc")),
                "pr_auc": safe_float(calibration_row.get("pr_auc")),
                "brier": safe_float(calibration_row.get("brier")),
                "hl_statistic": safe_float(calibration_row.get("hl_statistic")),
                "hl_df": safe_float(calibration_row.get("hl_df")),
                "hl_p_value": safe_float(calibration_row.get("hl_p_value")),
                "net_benefit_0.05": find_nearest_net_benefit(branch_dca_rows, 0.05),
                "net_benefit_0.10": find_nearest_net_benefit(branch_dca_rows, 0.10),
                "net_benefit_0.20": find_nearest_net_benefit(branch_dca_rows, 0.20),
                "net_benefit_0.30": find_nearest_net_benefit(branch_dca_rows, 0.30),
                "max_net_benefit_0.05_0.30": max_net_benefit(branch_dca_rows, 0.05, 0.30),
            }
        )
    return output_rows


def subgroup_sort_key(row: Dict[str, str]) -> tuple:
    order = {
        "overall": 0,
        "age": 1,
        "sex": 2,
        "smoking_history": 3,
        "metastasis": 4,
        "location_group": 5,
        "location_code": 6,
    }
    return (order.get(str(row.get("subgroup_name")), 99), str(row.get("subgroup_value")))


def build_top_feature_rows(detail_payload: Dict[str, object]) -> List[Dict[str, object]]:
    main_branch_name = get_main_branch_spec().name
    results_map = detail_payload.get("results", {})
    if not isinstance(results_map, dict):
        return []
    branch_payload = results_map.get(main_branch_name, {})
    if not isinstance(branch_payload, dict):
        return []
    importance_payload = branch_payload.get("importance", {})
    if not isinstance(importance_payload, dict):
        return []
    importances = importance_payload.get("xgboost_feature_importances", [])
    if not isinstance(importances, list):
        return []

    clinical_names = {"age", "sex", "smoking_history", "location_code"}
    rows: List[Dict[str, object]] = []
    for rank, item in enumerate(importances, start=1):
        if not isinstance(item, dict):
            continue
        feature_name = str(item.get("feature_name", ""))
        rows.append(
            {
                "rank": int(rank),
                "feature_name": feature_name,
                "feature_group": "clinical" if feature_name in clinical_names else "radiomics",
                "importance": safe_float(item.get("importance")),
            }
        )
    return rows


def build_feature_stability_rows(feature_stability_rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    output_rows: List[Dict[str, object]] = []
    for row in feature_stability_rows:
        output_rows.append(
            {
                "branch_name": str(row.get("branch_name", "")),
                "display_name": str(row.get("display_name", "")),
                "feature_name": str(row.get("feature_name", "")),
                "selection_count": int(safe_float(row.get("selection_count"))),
                "selection_rate": safe_float(row.get("selection_rate")),
                "total_fold_evaluations": int(safe_float(row.get("total_fold_evaluations"))),
            }
        )
    return output_rows


def build_key_results(
    model_rows: Sequence[Dict[str, object]],
    calibration_dca_rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    main_branch_name = get_main_branch_spec().name
    model_row = next((row for row in model_rows if str(row["branch_name"]) == main_branch_name), None)
    calibration_row = next((row for row in calibration_dca_rows if str(row["branch_name"]) == main_branch_name), None)
    if model_row is None:
        return {}
    payload: Dict[str, object] = {
        "main_model_name": model_row["branch_name"],
        "main_model_display_name": model_row["display_name"],
        "cv_auc": model_row["cv_auc"],
        "cv_pr_auc": model_row["cv_pr_auc"],
        "fixed_test_auc": model_row["fixed_test_auc"],
        "fixed_test_pr_auc": model_row["fixed_test_pr_auc"],
        "fixed_test_sensitivity": model_row["fixed_test_sensitivity"],
        "fixed_test_specificity": model_row["fixed_test_specificity"],
        "fixed_test_ppv": model_row["fixed_test_ppv"],
        "fixed_test_npv": model_row["fixed_test_npv"],
        "fixed_test_brier": model_row["fixed_test_brier"],
        "threshold": model_row["threshold"],
        "hosmer_lemeshow_p_value": model_row["calibration_hl_p_value"],
    }
    if calibration_row is not None:
        payload["main_model_dca"] = {
            "brier": calibration_row["brier"],
            "hl_statistic": calibration_row["hl_statistic"],
            "hl_df": calibration_row["hl_df"],
            "hl_p_value": calibration_row["hl_p_value"],
            "net_benefit_0.05": calibration_row["net_benefit_0.05"],
            "net_benefit_0.10": calibration_row["net_benefit_0.10"],
            "net_benefit_0.20": calibration_row["net_benefit_0.20"],
            "net_benefit_0.30": calibration_row["net_benefit_0.30"],
            "max_net_benefit_0.05_0.30": calibration_row["max_net_benefit_0.05_0.30"],
        }
    return payload


def main() -> None:
    assert_frozen_runtime()
    repeated_dir = Path(
        os.environ.get("LUAD_EGFR20INS_REPEATED_CV_RESULTS_DIR", str(PROJECT_ROOT / "results" / "egfr20ins_repeated_cv"))
    )
    fixed_dir = Path(
        os.environ.get(
            "LUAD_EGFR20INS_FIXED_BOOTSTRAP_RESULTS_DIR",
            str(PROJECT_ROOT / "results" / "egfr20ins_fixed_bootstrap"),
        )
    )
    subgroup_dir = Path(
        os.environ.get("LUAD_EGFR20INS_SUBGROUP_RESULTS_DIR", str(PROJECT_ROOT / "results" / "egfr20ins_subgroups"))
    )
    calibration_dir = Path(
        os.environ.get(
            "LUAD_EGFR20INS_CALIBRATION_DCA_RESULTS_DIR",
            str(PROJECT_ROOT / "results" / "egfr20ins_calibration_dca"),
        )
    )
    results_dir = Path(
        os.environ.get(
            "LUAD_EGFR20INS_PAPER_PACK_RESULTS_DIR",
            str(PROJECT_ROOT / "results" / "egfr20ins_paper_pack"),
        )
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = build_logger(
        "egfr20ins_paper_pack",
        results_dir / "egfr20ins_paper_pack.log",
        enable_file_logging=True,
        enable_stream_logging=True,
    )

    repeated_summary_path = repeated_dir / "repeated_cv_summary.csv"
    fixed_summary_path = fixed_dir / "fixed_test_summary.csv"
    fixed_delong_path = fixed_dir / "fixed_test_delong_comparisons.csv"
    fixed_detail_path = fixed_dir / "fixed_test_bootstrap_detail.json"
    subgroup_summary_path = subgroup_dir / "subgroup_summary.csv"
    calibration_summary_path = calibration_dir / "calibration_summary.csv"
    dca_summary_path = calibration_dir / "dca_summary.csv"
    feature_stability_summary_path = repeated_dir / "feature_stability_summary.csv"

    require_paths(
        [
            repeated_summary_path,
            fixed_summary_path,
            fixed_delong_path,
            fixed_detail_path,
            subgroup_summary_path,
            calibration_summary_path,
            dca_summary_path,
        ]
    )

    repeated_rows = read_rows_csv(repeated_summary_path)
    fixed_rows = read_rows_csv(fixed_summary_path)
    delong_rows = read_rows_csv(fixed_delong_path)
    subgroup_rows = sorted(read_rows_csv(subgroup_summary_path), key=subgroup_sort_key)
    calibration_rows = read_rows_csv(calibration_summary_path)
    dca_rows = read_rows_csv(dca_summary_path)
    fixed_detail_payload = read_json(fixed_detail_path)
    feature_stability_rows = read_rows_csv(feature_stability_summary_path) if feature_stability_summary_path.is_file() else []

    model_table_rows = build_model_table(repeated_rows, fixed_rows, calibration_rows, delong_rows)
    calibration_dca_rows = build_calibration_dca_table(calibration_rows, dca_rows)
    top_feature_rows = build_top_feature_rows(fixed_detail_payload)
    paper_feature_stability_rows = build_feature_stability_rows(feature_stability_rows)
    key_results = build_key_results(model_table_rows, calibration_dca_rows)

    write_rows_csv(
        results_dir / "paper_model_comparison_table.csv",
        model_table_rows,
        fieldnames=[
            "branch_name",
            "display_name",
            "feature_block",
            "model_kind",
            "is_main_model",
            "cv_auc",
            "cv_pr_auc",
            "cv_sensitivity",
            "cv_specificity",
            "fixed_test_auc",
            "fixed_test_pr_auc",
            "fixed_test_sensitivity",
            "fixed_test_specificity",
            "fixed_test_ppv",
            "fixed_test_npv",
            "fixed_test_f1",
            "fixed_test_brier",
            "threshold",
            "calibration_brier",
            "calibration_hl_statistic",
            "calibration_hl_df",
            "calibration_hl_p_value",
            "delong_p_vs_main",
            "cv_auc_mean",
            "cv_auc_std",
            "cv_pr_auc_mean",
            "cv_sensitivity_mean",
            "cv_specificity_mean",
            "fixed_auc_value",
            "fixed_pr_auc_value",
            "fixed_sensitivity_value",
            "fixed_specificity_value",
            "fixed_ppv_value",
            "fixed_npv_value",
            "fixed_f1_value",
        ],
    )
    write_rows_csv(
        results_dir / "paper_subgroup_table.csv",
        subgroup_rows,
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
        results_dir / "paper_calibration_dca_table.csv",
        calibration_dca_rows,
        fieldnames=[
            "branch_name",
            "display_name",
            "auc",
            "pr_auc",
            "brier",
            "hl_statistic",
            "hl_df",
            "hl_p_value",
            "net_benefit_0.05",
            "net_benefit_0.10",
            "net_benefit_0.20",
            "net_benefit_0.30",
            "max_net_benefit_0.05_0.30",
        ],
    )
    write_rows_csv(
        results_dir / "paper_delong_table.csv",
        delong_rows,
        fieldnames=["main_branch_name", "compare_branch_name", "main_auc", "compare_auc", "z_value", "p_value"],
    )
    write_rows_csv(
        results_dir / "paper_top_features.csv",
        top_feature_rows,
        fieldnames=["rank", "feature_name", "feature_group", "importance"],
    )
    write_rows_csv(
        results_dir / "paper_feature_stability_table.csv",
        paper_feature_stability_rows,
        fieldnames=[
            "branch_name",
            "display_name",
            "feature_name",
            "selection_count",
            "selection_rate",
            "total_fold_evaluations",
        ],
    )

    save_json(key_results, results_dir / "paper_key_results.json")
    save_json(
        {
            "repeated_cv_summary_path": str(repeated_summary_path),
            "fixed_test_summary_path": str(fixed_summary_path),
            "subgroup_summary_path": str(subgroup_summary_path),
            "calibration_summary_path": str(calibration_summary_path),
            "dca_summary_path": str(dca_summary_path),
            "feature_stability_summary_path": str(feature_stability_summary_path) if feature_stability_summary_path.is_file() else "",
            "main_model_name": get_main_branch_spec().name,
            "branch_specs": [branch.__dict__ for branch in default_branch_specs()],
        },
        results_dir / "paper_pack_manifest.json",
    )
    logger.info(
        "EGFR20ins paper tables built | output_dir=%s | model_table=%s | subgroup_table=%s",
        results_dir,
        results_dir / "paper_model_comparison_table.csv",
        results_dir / "paper_subgroup_table.csv",
    )


if __name__ == "__main__":
    main()
