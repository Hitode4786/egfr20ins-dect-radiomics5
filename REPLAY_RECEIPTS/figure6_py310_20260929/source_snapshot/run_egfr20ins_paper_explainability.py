from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from egfr20ins_paper_common import (  # noqa: E402
    PaperBranchSpec,
    branch_matrix_and_names,
    fixed_split_indices,
    get_fixed_eval_seed,
    load_bundle_and_feature_blocks,
    read_json,
    temporary_env,
    write_rows_csv,
)
from run_egfr20ins_binary_strategies import (  # noqa: E402
    build_calibrated_classifier,
    get_calibration_cv_splits,
    get_calibration_method,
    xgb_binary_classifier,
)
from luad_tabular_utils import build_logger, save_json  # noqa: E402


Array1D = np.ndarray
Array2D = np.ndarray


def require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def require_shap():
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "shap is required for EGFR20ins paper explainability outputs. "
            "Install it with: pip install shap"
        ) from exc
    return shap


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def require_paths(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required explainability inputs:\n" + "\n".join(missing))


def load_main_branch_spec(paper_manifest_path: Path, main_branch_name: str) -> PaperBranchSpec:
    manifest = read_json(paper_manifest_path)
    branch_specs = manifest.get("branch_specs", [])
    if isinstance(branch_specs, list):
        for item in branch_specs:
            if isinstance(item, dict) and str(item.get("name")) == main_branch_name:
                return PaperBranchSpec(
                    name=str(item.get("name")),
                    display_name=str(item.get("display_name", item.get("name", ""))),
                    feature_block=str(item.get("feature_block", "")),
                    model_kind=str(item.get("model_kind", "")),
                    rfe_dim=None if item.get("rfe_dim") is None else int(item.get("rfe_dim")),
                    seed=int(item.get("seed", 72)),
                )
    raise KeyError(f"Unable to find branch spec for main branch: {main_branch_name}")


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fit_rfe_xgboost_calibrated_with_artifacts(
    x_train: Array2D,
    y_train: Array1D,
    x_eval: Array2D,
    seed: int,
    feature_names: Sequence[str],
    rfe_dim: int,
    calibration_method: str,
    calibration_cv: Optional[int] = None,
) -> Dict[str, object]:
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64).reshape(-1)
    x_eval = np.asarray(x_eval, dtype=np.float32)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_eval_scaled = scaler.transform(x_eval)

    selector_estimator = LogisticRegression(
        solver="liblinear",
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    )
    rfe = RFE(
        estimator=selector_estimator,
        n_features_to_select=int(rfe_dim),
        step=0.1,
    )
    x_train_selected = rfe.fit_transform(x_train_scaled, y_train)
    x_eval_selected = rfe.transform(x_eval_scaled)

    if calibration_cv is None:
        calibration_cv = get_calibration_cv_splits(y_train)
    calibration_cv = int(calibration_cv)

    if calibration_cv >= 2:
        calibrated_model = build_calibrated_classifier(
            estimator=xgb_binary_classifier(seed),
            method=calibration_method,
            cv=calibration_cv,
        )
        calibrated_model.fit(x_train_selected, y_train)
        calibrated_scores = np.asarray(calibrated_model.predict_proba(x_eval_selected), dtype=np.float64)[:, 1]
        calibration_applied = True
    else:
        calibrated_model = None
        calibration_applied = False
        fallback_model = xgb_binary_classifier(seed)
        fallback_model.fit(x_train_selected, y_train)
        calibrated_scores = np.asarray(fallback_model.predict_proba(x_eval_selected), dtype=np.float64)[:, 1]

    shap_model = xgb_binary_classifier(seed)
    shap_model.fit(x_train_selected, y_train)
    xgb_scores = np.asarray(shap_model.predict_proba(x_eval_selected), dtype=np.float64)[:, 1]

    selected_indices = np.flatnonzero(rfe.support_).astype(np.int64)
    selected_feature_names = [str(feature_names[int(index)]) for index in selected_indices.tolist()]
    return {
        "scaler": scaler,
        "selector": rfe,
        "selected_indices": selected_indices,
        "selected_feature_names": selected_feature_names,
        "x_train_selected": np.asarray(x_train_selected, dtype=np.float32),
        "x_eval_selected": np.asarray(x_eval_selected, dtype=np.float32),
        "x_eval_display": np.asarray(x_eval[:, selected_indices], dtype=np.float32),
        "calibrated_scores": calibrated_scores,
        "xgb_scores": xgb_scores,
        "calibration_method": calibration_method,
        "calibration_cv": calibration_cv,
        "calibration_applied": calibration_applied,
        "calibrated_model": calibrated_model,
        "shap_model": shap_model,
    }


def build_shap_explanation(
    shap_model: Any,
    x_train_selected: Array2D,
    x_eval_selected: Array2D,
    x_eval_display: Array2D,
    selected_feature_names: Sequence[str],
    seed: int,
) -> Tuple[Any, str]:
    shap = require_shap()
    rng = np.random.default_rng(seed)

    background = np.asarray(x_train_selected, dtype=np.float32)
    if background.shape[0] > 200:
        background = background[rng.choice(background.shape[0], size=200, replace=False)]

    attempts: List[Tuple[str, Dict[str, object]]] = [
        (
            "probability",
            {
                "data": background,
                "model_output": "probability",
                "feature_perturbation": "interventional",
            },
        ),
        (
            "raw",
            {
                "model_output": "raw",
                "feature_perturbation": "tree_path_dependent",
            },
        ),
    ]

    last_error: Optional[Exception] = None
    for mode, kwargs in attempts:
        try:
            explainer = shap.TreeExplainer(shap_model, **kwargs)
            explanation = explainer(np.asarray(x_eval_selected, dtype=np.float32), check_additivity=False)
            return (
                shap.Explanation(
                    values=np.asarray(explanation.values, dtype=np.float64),
                    base_values=np.asarray(explanation.base_values, dtype=np.float64),
                    data=np.asarray(x_eval_display, dtype=np.float64),
                    feature_names=list(selected_feature_names),
                ),
                mode,
            )
        except Exception as exc:  # pragma: no cover - shap backend differences are environment-dependent
            last_error = exc

    # SHAP 0.48 cannot parse the vector-form base_score emitted by XGBoost 3.2.
    # Use XGBoost's native TreeSHAP contributions as a compatibility fallback.
    # These explain the pre-calibration raw margin, which is the output used for
    # the Figure 6 waterfall panels.
    try:
        import xgboost

        evaluation = np.asarray(x_eval_selected, dtype=np.float32)
        matrix = xgboost.DMatrix(evaluation)
        booster = shap_model.get_booster()
        contributions = np.asarray(booster.predict(matrix, pred_contribs=True), dtype=np.float64)
        expected_columns = evaluation.shape[1] + 1
        if contributions.ndim != 2 or contributions.shape != (evaluation.shape[0], expected_columns):
            raise RuntimeError(
                "Unexpected native XGBoost contribution shape: "
                f"{contributions.shape}; expected {(evaluation.shape[0], expected_columns)}."
            )

        reconstructed_margin = contributions[:, -1] + contributions[:, :-1].sum(axis=1)
        direct_margin = np.asarray(booster.predict(matrix, output_margin=True), dtype=np.float64).reshape(-1)
        max_additivity_delta = float(np.max(np.abs(reconstructed_margin - direct_margin)))
        # XGBoost calculates both outputs in float32; summing the 48 exported
        # float32 contributions in float64 can differ from its internal
        # accumulation by a few micro-units. 1e-5 remains a strict raw-margin
        # additivity guard while allowing that documented numerical rounding.
        if max_additivity_delta > 1e-5:
            raise RuntimeError(
                "Native XGBoost contribution additivity check failed: "
                f"max_delta={max_additivity_delta:.8g}."
            )

        return (
            shap.Explanation(
                values=contributions[:, :-1],
                base_values=contributions[:, -1],
                data=np.asarray(x_eval_display, dtype=np.float64),
                feature_names=list(selected_feature_names),
            ),
            "xgboost_pred_contribs_raw",
        )
    except Exception as fallback_error:  # pragma: no cover - backend differences are environment-dependent
        message = "Failed to compute SHAP values for EGFR20ins paper explainability."
        if last_error is not None:
            message += f" TreeExplainer error: {last_error}"
        raise RuntimeError(message) from fallback_error


def save_summary_plot(
    output_path: Path,
    shap_values: Any,
    x_display: Array2D,
    selected_feature_names: Sequence[str],
    max_display: int,
    title: str,
) -> None:
    shap = require_shap()
    plt = require_matplotlib()

    plt.figure(figsize=(12, 8), dpi=300)
    shap.summary_plot(
        np.asarray(shap_values.values, dtype=np.float64),
        features=np.asarray(x_display, dtype=np.float64),
        feature_names=list(selected_feature_names),
        max_display=max_display,
        show=False,
    )
    figure = plt.gcf()
    if figure.axes:
        figure.axes[0].set_title(title)
    figure.tight_layout()
    # Keep the PNG preview and emit editable publication formats from the same
    # plotted objects so the vector exports cannot drift from the raster view.
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def base_value_for_row(base_values: object, row_idx: int) -> float:
    array = np.asarray(base_values, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    flat = array.reshape(-1)
    if row_idx >= flat.size:
        return float(flat[-1])
    return float(flat[row_idx])


def save_waterfall_plot(
    output_path: Path,
    shap_values: Any,
    row_idx: int,
    selected_feature_names: Sequence[str],
    x_display: Array2D,
    max_display: int,
    title: str,
) -> None:
    shap = require_shap()
    plt = require_matplotlib()

    single = shap.Explanation(
        values=np.asarray(shap_values.values[row_idx], dtype=np.float64),
        base_values=base_value_for_row(shap_values.base_values, row_idx),
        data=np.asarray(x_display[row_idx], dtype=np.float64),
        feature_names=list(selected_feature_names),
    )
    shap.plots.waterfall(single, max_display=max_display, show=False)
    figure = plt.gcf()
    # SHAP's default waterfall canvas is too narrow for the long feature names
    # and bottom x-axis annotations when three panels are assembled as Figure 6.
    # Widen each panel and reserve explicit bottom margin to prevent numeric
    # tick labels from colliding with the E[f(X)] annotation.
    figure.set_size_inches(6.2, 4.8)
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.88)
    if figure.axes:
        figure.axes[0].set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def save_combined_waterfall_figure(
    output_path: Path,
    shap_values: Any,
    sample_rows: Sequence[Dict[str, object]],
    fixed_indices: Sequence[int],
    calibrated_scores: Sequence[float],
    xgb_scores: Sequence[float],
    selected_feature_names: Sequence[str],
    max_display: int = 12,
) -> List[Path]:
    """Draw a consistent, de-identified three-panel waterfall display."""
    plt = require_matplotlib()
    values = np.asarray(shap_values.values, dtype=np.float64)
    names = np.asarray(list(selected_feature_names), dtype=object)
    panel_data: List[Tuple[np.ndarray, List[str], float, float]] = []
    for row_idx in fixed_indices:
        row_values = values[int(row_idx)]
        order = np.argsort(np.abs(row_values))[::-1]
        keep = order[: int(max_display)]
        other = float(np.sum(row_values[order[int(max_display):]]))
        panel_vals = np.concatenate([row_values[keep], np.asarray([other])])
        panel_names = [str(names[i]) for i in keep.tolist()] + ["34 other features"]
        panel_data.append((panel_vals, panel_names, base_value_for_row(shap_values.base_values, int(row_idx)), float(np.sum(row_values))))

    global_limit = max(1.0, max(float(np.max(np.abs(vals))) for vals, _, _, _ in panel_data) * 1.25)
    fig, axes = plt.subplots(1, len(panel_data), figsize=(14.6, 5.2), sharex=True)
    if len(panel_data) == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")
    for panel_idx, (ax, (panel_vals, panel_names, base_value, shap_sum)) in enumerate(zip(axes, panel_data)):
        y = np.arange(len(panel_vals))[::-1]
        def compact_name(name: str) -> str:
            if name == "34 other features":
                return name
            compact = str(name).replace("__original_", " | ").replace("_original_", " | ").replace("_", " ")
            return compact if len(compact) <= 28 else compact[:25] + "..."

        display_names = [compact_name(str(name)) for name in panel_names]
        colors = ["#d81b60" if value >= 0 else "#1976d2" for value in panel_vals]
        ax.barh(y, panel_vals, color=colors, height=0.66, alpha=0.95)
        ax.axvline(0.0, color="#6b7280", linewidth=0.9)
        for yy, value in zip(y, panel_vals):
            ax.text(value + (0.035 if value >= 0 else -0.035), yy, f"{value:+.2f}", va="center", ha="left" if value >= 0 else "right", fontsize=7.3, color="#374151")
        ax.set_yticks(y, labels=display_names, fontsize=6.8)
        ax.set_xlim(-global_limit, global_limit)
        ax.grid(True, axis="x", alpha=0.18)
        row = sample_rows[panel_idx]
        true_label = "20-ins" if int(row["y_true"]) == 1 else "non-20-ins"
        pred_label = "20-ins" if int(row["pred"]) == 1 else "non-20-ins"
        role_labels = {
            "correct_positive": "Correct Positive",
            "hard_positive": "Borderline Positive",
            "negative_reference": "Negative Reference",
        }
        role = role_labels.get(
            str(row["selection_role"]), str(row["selection_role"]).replace("_", " ").title()
        )
        ax.set_title(f"{role}\ntrue={true_label} | pred={pred_label}\ncalibrated={float(calibrated_scores[panel_idx]):.3f} | xgb={float(xgb_scores[panel_idx]):.3f}", fontsize=8.4, pad=9)
        ax.text(0.98, 1.01, f"base={base_value:.3f}\nf(x)={base_value + shap_sum:.3f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0, color="#6b7280")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=2)
        if panel_idx == 0:
            ax.set_ylabel("Feature contribution", fontsize=8.2)
        else:
            ax.set_ylabel("")
    axes[1 if len(axes) > 1 else 0].set_xlabel("SHAP contribution to raw model output", fontsize=8.4)
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.14, top=0.82, wspace=0.24)
    outputs = []
    for suffix, dpi in ((".pdf", None), (".svg", None), (".png", 600)):
        path = output_path.with_suffix(suffix)
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if dpi is not None:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def sort_rows_by_score(rows: Sequence[Dict[str, object]], *, descending: bool) -> List[Dict[str, object]]:
    return sorted(rows, key=lambda row: float(row["score"]), reverse=descending)


def pick_unused_row(
    rows: Sequence[Dict[str, object]],
    used_patient_ids: Sequence[str],
) -> Optional[Dict[str, object]]:
    used = set(str(patient_id) for patient_id in used_patient_ids)
    for row in rows:
        if str(row["patient_id"]) not in used:
            return dict(row)
    return None


def choose_showcase_samples(
    prediction_rows: Sequence[Dict[str, str]],
    main_branch_name: str,
    threshold: float,
) -> List[Dict[str, object]]:
    score_key = f"{main_branch_name}_score"
    pred_key = f"{main_branch_name}_pred"
    parsed_rows: List[Dict[str, object]] = []
    for row in prediction_rows:
        score = float(row[score_key])
        pred_value = int(row[pred_key]) if pred_key in row else int(score >= threshold)
        parsed_rows.append(
            {
                "patient_id": str(row["patient_id"]),
                "y_true": int(row["y_true"]),
                "score": score,
                "pred": pred_value,
                "decision": "positive" if pred_value == 1 else "negative",
                "margin_to_threshold": float(score - threshold),
            }
        )

    positive_rows = [row for row in parsed_rows if int(row["y_true"]) == 1]
    negative_rows = [row for row in parsed_rows if int(row["y_true"]) == 0]
    true_positive_rows = [row for row in positive_rows if int(row["pred"]) == 1]
    false_negative_rows = [row for row in positive_rows if int(row["pred"]) == 0]
    true_negative_rows = [row for row in negative_rows if int(row["pred"]) == 0]

    selections: List[Dict[str, object]] = []
    used_patient_ids: List[str] = []

    role_specs = [
        (
            "correct_positive",
            sort_rows_by_score(true_positive_rows, descending=True),
            sort_rows_by_score(positive_rows, descending=True),
            "preferred_true_positive",
            "fallback_highest_score_positive",
        ),
        (
            "hard_positive",
            sort_rows_by_score(false_negative_rows, descending=True),
            sort_rows_by_score(positive_rows, descending=False),
            "preferred_false_negative",
            "fallback_lowest_score_positive",
        ),
        (
            "negative_reference",
            sort_rows_by_score(true_negative_rows, descending=False),
            sort_rows_by_score(negative_rows, descending=False),
            "preferred_true_negative",
            "fallback_lowest_score_negative",
        ),
    ]

    for role_name, preferred_rows, fallback_rows, preferred_note, fallback_note in role_specs:
        picked = pick_unused_row(preferred_rows, used_patient_ids)
        selection_note = preferred_note
        if picked is None:
            picked = pick_unused_row(fallback_rows, used_patient_ids)
            selection_note = fallback_note
        if picked is None:
            continue
        picked["selection_role"] = role_name
        picked["selection_note"] = selection_note
        used_patient_ids.append(str(picked["patient_id"]))
        selections.append(picked)
    return selections


def sanitize_for_filename(value: str) -> str:
    sanitized = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_"}:
            sanitized.append(char)
        else:
            sanitized.append("_")
    return "".join(sanitized).strip("_") or "sample"


def update_figures_manifest(
    figures_manifest_path: Path,
    explainability_manifest_path: Path,
    summary_plot_path: Path,
    waterfall_paths: Sequence[Path],
    explainability_dir: Path,
) -> None:
    manifest: Dict[str, object] = {}
    if figures_manifest_path.is_file():
        existing = read_json(figures_manifest_path)
        if isinstance(existing, dict):
            manifest.update(existing)
    manifest.update(
        {
            "paper_explainability_dir": str(explainability_dir),
            "explainability_manifest": str(explainability_manifest_path),
            "shap_summary_plot": str(summary_plot_path),
            "waterfall_plots": [str(path) for path in waterfall_paths],
        }
    )
    save_json(manifest, figures_manifest_path)


def top_shap_rows(
    shap_values: Any,
    selected_feature_names: Sequence[str],
    top_n: int,
) -> List[Dict[str, object]]:
    mean_abs = np.mean(np.abs(np.asarray(shap_values.values, dtype=np.float64)), axis=0)
    order = np.argsort(-mean_abs)[: int(top_n)]
    rows: List[Dict[str, object]] = []
    for rank, feature_idx in enumerate(order.tolist(), start=1):
        rows.append(
            {
                "rank": rank,
                "feature_name": str(selected_feature_names[int(feature_idx)]),
                "mean_abs_shap": float(mean_abs[int(feature_idx)]),
            }
        )
    return rows


def main() -> None:
    fixed_dir = Path(
        os.environ.get(
            "LUAD_EGFR20INS_FIXED_BOOTSTRAP_RESULTS_DIR",
            str(PROJECT_ROOT / "results" / "egfr20ins_fixed_bootstrap"),
        )
    )
    paper_dir = Path(
        os.environ.get(
            "LUAD_EGFR20INS_PAPER_PACK_RESULTS_DIR",
            str(PROJECT_ROOT / "results" / "egfr20ins_paper_pack"),
        )
    )
    figures_dir = Path(
        os.environ.get(
            "LUAD_EGFR20INS_PAPER_FIGURES_DIR",
            str(PROJECT_ROOT / "results" / "egfr20ins_paper_figures"),
        )
    )
    explainability_dir = Path(
        os.environ.get(
            "LUAD_EGFR20INS_PAPER_EXPLAINABILITY_DIR",
            str(figures_dir / "explainability"),
        )
    )
    explainability_dir.mkdir(parents=True, exist_ok=True)

    logger = build_logger(
        "egfr20ins_paper_explainability",
        explainability_dir / "egfr20ins_paper_explainability.log",
        enable_file_logging=True,
        enable_stream_logging=True,
    )

    fixed_prediction_path = fixed_dir / "fixed_test_predictions.csv"
    fixed_detail_path = fixed_dir / "fixed_test_bootstrap_detail.json"
    paper_manifest_path = paper_dir / "paper_pack_manifest.json"
    key_results_path = paper_dir / "paper_key_results.json"
    figures_manifest_path = figures_dir / "paper_figures_manifest.json"
    require_paths([fixed_prediction_path, fixed_detail_path, paper_manifest_path, key_results_path])

    prediction_rows = read_csv_rows(fixed_prediction_path)
    fixed_detail = read_json(fixed_detail_path)
    key_results = read_json(key_results_path)

    main_branch_name = str(fixed_detail.get("main_branch_name") or key_results.get("main_model_name"))
    branch = load_main_branch_spec(paper_manifest_path, main_branch_name)
    if branch.model_kind != "rfe_xgboost_calibrated":
        raise ValueError(
            f"Explainability currently supports rfe_xgboost_calibrated main branches only, got {branch.model_kind!r}."
        )
    if branch.rfe_dim is None:
        raise ValueError("Main branch RFE dimension is missing; explainability cannot rebuild the selected feature set.")

    main_result = fixed_detail.get("results", {}).get(main_branch_name)
    if not isinstance(main_result, dict):
        raise KeyError(f"Missing main branch result block in fixed-test detail: {main_branch_name}")
    threshold = safe_float(main_result.get("threshold"))
    fit_seed = int(main_result.get("fit_seed", get_fixed_eval_seed(branch.seed)))

    importance_payload = main_result.get("importance")
    calibration_method = str(
        importance_payload.get("calibration_method")  # type: ignore[union-attr]
        if isinstance(importance_payload, dict) and importance_payload.get("calibration_method") is not None
        else get_calibration_method()
    )
    calibration_cv = (
        int(importance_payload.get("calibration_cv"))  # type: ignore[union-attr]
        if isinstance(importance_payload, dict) and importance_payload.get("calibration_cv") is not None
        else None
    )

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
    x_all, feature_names = branch_matrix_and_names(
        branch,
        x_radiomics=x_radiomics,
        x_clinical=x_clinical,
        x_combined=x_combined,
        radiomics_feature_names=radiomics_feature_names,
        clinical_feature_names=clinical_feature_names,
        combined_feature_names=combined_feature_names,
    )
    x_dev = np.asarray(x_all[dev_indices], dtype=np.float32)
    x_fixed = np.asarray(x_all[fixed_test_indices], dtype=np.float32)
    y_dev = y[dev_indices]
    fixed_patient_ids = [str(bundle.sample_ids[int(index)]) for index in fixed_test_indices.tolist()]
    fixed_id_to_index = {patient_id: idx for idx, patient_id in enumerate(fixed_patient_ids)}

    with temporary_env({"LUAD_EGFR20INS_RFE_N_FEATURES": str(int(branch.rfe_dim))}):
        artifacts = fit_rfe_xgboost_calibrated_with_artifacts(
            x_train=x_dev,
            y_train=y_dev,
            x_eval=x_fixed,
            seed=fit_seed,
            feature_names=feature_names,
            rfe_dim=int(branch.rfe_dim),
            calibration_method=calibration_method,
            calibration_cv=calibration_cv,
        )

    saved_scores_lookup = {
        str(row["patient_id"]): float(row[f"{main_branch_name}_score"])
        for row in prediction_rows
        if f"{main_branch_name}_score" in row
    }
    rebuilt_scores = np.asarray(artifacts["calibrated_scores"], dtype=np.float64)
    alignment_deltas: List[float] = []
    for patient_id, rebuilt_score in zip(fixed_patient_ids, rebuilt_scores.tolist()):
        saved_score = saved_scores_lookup.get(patient_id)
        if saved_score is None:
            continue
        alignment_deltas.append(abs(float(saved_score) - float(rebuilt_score)))
    max_alignment_delta = float(max(alignment_deltas)) if alignment_deltas else float("nan")
    logger.info(
        "Rebuilt main branch for explainability | branch=%s | fit_seed=%d | rfe_dim=%d | calibration=%s | max_score_delta=%.8f",
        main_branch_name,
        fit_seed,
        int(branch.rfe_dim),
        calibration_method,
        max_alignment_delta,
    )

    shap_values, shap_mode = build_shap_explanation(
        shap_model=artifacts["shap_model"],
        x_train_selected=np.asarray(artifacts["x_train_selected"], dtype=np.float32),
        x_eval_selected=np.asarray(artifacts["x_eval_selected"], dtype=np.float32),
        x_eval_display=np.asarray(artifacts["x_eval_display"], dtype=np.float32),
        selected_feature_names=list(artifacts["selected_feature_names"]),
        seed=fit_seed,
    )

    summary_plot_path = explainability_dir / "figure_shap_summary_fixed_test.png"
    selected_feature_names = list(artifacts["selected_feature_names"])
    x_eval_display = np.asarray(artifacts["x_eval_display"], dtype=np.float32)
    summary_max_display = int(os.environ.get("LUAD_EGFR20INS_PAPER_EXPLAINABILITY_SUMMARY_TOP_N", "20"))
    waterfall_max_display = int(os.environ.get("LUAD_EGFR20INS_PAPER_EXPLAINABILITY_WATERFALL_TOP_N", "15"))
    save_summary_plot(
        summary_plot_path,
        shap_values=shap_values,
        x_display=x_eval_display,
        selected_feature_names=selected_feature_names,
        max_display=summary_max_display,
        title="SHAP Summary (Temporal held-out, underlying XGBoost before calibration)",
    )

    selected_samples = choose_showcase_samples(prediction_rows, main_branch_name=main_branch_name, threshold=threshold)
    waterfall_paths: List[Path] = []
    sample_rows: List[Dict[str, object]] = []
    selected_fixed_indices: List[int] = []
    selected_calibrated_scores: List[float] = []
    selected_xgb_scores: List[float] = []
    xgb_scores = np.asarray(artifacts["xgb_scores"], dtype=np.float64)
    for order_idx, row in enumerate(selected_samples, start=1):
        patient_id = str(row["patient_id"])
        fixed_idx = fixed_id_to_index.get(patient_id)
        if fixed_idx is None:
            logger.warning("Skipping waterfall sample missing from fixed-test index map: %s", patient_id)
            continue
        calibrated_score = float(saved_scores_lookup.get(patient_id, rebuilt_scores[fixed_idx]))
        xgb_score = float(xgb_scores[fixed_idx])
        filename = f"waterfall_{order_idx:02d}_{row['selection_role']}_{sanitize_for_filename(patient_id)}.png"
        output_path = explainability_dir / filename
        true_label = "20-ins" if int(row["y_true"]) == 1 else "non-20-ins"
        predicted_label = "20-ins" if int(row["pred"]) == 1 else "non-20-ins"
        save_waterfall_plot(
            output_path,
            shap_values=shap_values,
            row_idx=fixed_idx,
            selected_feature_names=selected_feature_names,
            x_display=x_eval_display,
            max_display=waterfall_max_display,
            title=(
                f"{ {'correct_positive': 'Correct Positive', 'hard_positive': 'Borderline Positive', 'negative_reference': 'Negative Reference'}.get(str(row['selection_role']), str(row['selection_role']).replace('_', ' ').title()) } | "
                f"true={true_label} | pred={predicted_label} | "
                f"calibrated={calibrated_score:.3f} | xgb={xgb_score:.3f}"
            ),
        )
        waterfall_paths.append(output_path)
        selected_fixed_indices.append(int(fixed_idx))
        selected_calibrated_scores.append(calibrated_score)
        selected_xgb_scores.append(xgb_score)
        sample_rows.append(
            {
                "waterfall_order": order_idx,
                "selection_role": str(row["selection_role"]),
                "selection_note": str(row["selection_note"]),
                "patient_id": patient_id,
                "y_true": int(row["y_true"]),
                "pred": int(row["pred"]),
                "threshold": float(threshold),
                "calibrated_score": calibrated_score,
                "xgb_score": xgb_score,
                "margin_to_threshold": float(calibrated_score - threshold),
                "plot_path": str(output_path),
            }
        )

    combined_waterfall_path = explainability_dir / "figure6_combined_waterfall"
    combined_waterfall_paths = save_combined_waterfall_figure(
        combined_waterfall_path,
        shap_values=shap_values,
        sample_rows=sample_rows,
        fixed_indices=selected_fixed_indices,
        calibrated_scores=selected_calibrated_scores,
        xgb_scores=selected_xgb_scores,
        selected_feature_names=selected_feature_names,
        max_display=12,
    )

    sample_table_path = explainability_dir / "selected_waterfall_samples.csv"
    write_rows_csv(
        sample_table_path,
        sample_rows,
        fieldnames=[
            "waterfall_order",
            "selection_role",
            "selection_note",
            "patient_id",
            "y_true",
            "pred",
            "threshold",
            "calibrated_score",
            "xgb_score",
            "margin_to_threshold",
            "plot_path",
        ],
    )

    top_shap_path = explainability_dir / "shap_top_features.csv"
    shap_rows = top_shap_rows(
        shap_values=shap_values,
        selected_feature_names=selected_feature_names,
        top_n=int(os.environ.get("LUAD_EGFR20INS_PAPER_EXPLAINABILITY_SUMMARY_TOP_N", "20")),
    )
    write_rows_csv(top_shap_path, shap_rows, fieldnames=["rank", "feature_name", "mean_abs_shap"])

    explainability_manifest_path = explainability_dir / "paper_explainability_manifest.json"
    save_json(
        {
            "main_branch_name": main_branch_name,
            "main_branch_display_name": branch.display_name,
            "feature_block": branch.feature_block,
            "model_kind": branch.model_kind,
            "fit_seed": fit_seed,
            "rfe_dim": int(branch.rfe_dim),
            "threshold": float(threshold),
            "calibration_method": calibration_method,
            "calibration_cv": artifacts["calibration_cv"],
            "calibration_applied": bool(artifacts["calibration_applied"]),
            "selected_feature_count": len(selected_feature_names),
            "selected_feature_names": selected_feature_names,
            "saved_prediction_alignment_max_abs_diff": max_alignment_delta,
            "shap_model_output": shap_mode,
            "summary_plot": str(summary_plot_path),
            "waterfall_plots": [str(path) for path in waterfall_paths],
            "combined_waterfall_figure": [str(path) for path in combined_waterfall_paths],
            "selected_samples_table": str(sample_table_path),
            "top_shap_table": str(top_shap_path),
            "selected_samples": sample_rows,
        },
        explainability_manifest_path,
    )

    update_figures_manifest(
        figures_manifest_path=figures_manifest_path,
        explainability_manifest_path=explainability_manifest_path,
        summary_plot_path=summary_plot_path,
        waterfall_paths=waterfall_paths,
        explainability_dir=explainability_dir,
    )
    logger.info(
        "EGFR20ins paper explainability finished | output_dir=%s | shap_mode=%s | waterfall_count=%d",
        explainability_dir,
        shap_mode,
        len(waterfall_paths),
    )


if __name__ == "__main__":
    main()
