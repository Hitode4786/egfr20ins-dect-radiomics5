from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from luad_tabular_utils import build_logger, save_json
from egfr20ins_runtime_guard import assert_frozen_runtime


def require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    import matplotlib.pyplot as plt

    return plt


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def require_paths(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required paper-figure inputs:\n" + "\n".join(missing))


def load_main_predictions(
    prediction_rows: Sequence[Dict[str, str]],
    main_branch_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray([int(row["y_true"]) for row in prediction_rows], dtype=np.int64)
    score_key = f"{main_branch_name}_score"
    if not prediction_rows or score_key not in prediction_rows[0]:
        raise KeyError(f"Prediction file missing score column: {score_key}")
    y_score = np.asarray([float(row[score_key]) for row in prediction_rows], dtype=np.float64)
    return y_true, y_score


def plot_roc(ax, y_true: np.ndarray, y_score: np.ndarray, auc_label: str, threshold: float) -> None:
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    ax.plot(fpr, tpr, color="#004c6d", linewidth=2.3, label=f"AUC-ROC {auc_label}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9aa5b1", linewidth=1.2)
    pred = (y_score >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    point_fpr = 1.0 - spec
    point_tpr = sens
    ax.scatter([point_fpr], [point_tpr], color="#c1121f", s=46, zorder=3)
    ax.annotate(
        f"thr={threshold:.3f}",
        (point_fpr, point_tpr),
        textcoords="offset points",
        # The locked operating point is at the upper-right edge of this panel.
        # Anchor its label to the left so neither the label nor its final digit is clipped.
        xytext=(-8, -10),
        ha="right",
        fontsize=8,
        color="#c1121f",
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")


def plot_pr(ax, y_true: np.ndarray, y_score: np.ndarray, pr_auc_label: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    prevalence = float(np.mean(y_true))
    ax.plot(recall, precision, color="#2f7d32", linewidth=2.3, label=f"PR-AUC {pr_auc_label}")
    ax.hlines(prevalence, 0.0, 1.0, linestyle="--", color="#9aa5b1", linewidth=1.2, label=f"Prevalence {prevalence:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")


def plot_calibration(ax, calibration_rows: Sequence[Dict[str, str]], brier: float, hl_p_value: float) -> None:
    rows_sorted = sorted(calibration_rows, key=lambda row: safe_float(row.get("prob_pred")))
    x_values = [safe_float(row.get("prob_pred")) for row in rows_sorted]
    y_values = [safe_float(row.get("prob_true")) for row in rows_sorted]
    ax.plot(x_values, y_values, marker="o", color="#8a5a44", linewidth=2.0)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9aa5b1", linewidth=1.2)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Exploratory calibration")
    if np.isfinite(hl_p_value):
        calibration_text = f"Brier = {brier:.3f}\nHL p = {hl_p_value:.3f}"
    else:
        calibration_text = f"Brier = {brier:.3f}\nHL p = NA"
    ax.text(0.03, 0.93, calibration_text, transform=ax.transAxes, fontsize=8, ha="left", va="top")
    ax.text(
        0.97,
        0.04,
        "Descriptive only\n4 positive cases",
        transform=ax.transAxes,
        fontsize=7,
        ha="right",
        va="bottom",
        color="#7f1d1d",
        bbox={"facecolor": "white", "edgecolor": "#fecaca", "alpha": 0.9, "pad": 2},
    )
    ax.grid(True, alpha=0.25)


def plot_dca(ax, dca_rows: Sequence[Dict[str, str]]) -> None:
    rows_sorted = sorted(dca_rows, key=lambda row: safe_float(row.get("threshold")))
    thresholds = [safe_float(row.get("threshold")) for row in rows_sorted]
    model_nb = [safe_float(row.get("model_net_benefit")) for row in rows_sorted]
    treat_all = [safe_float(row.get("treat_all_net_benefit")) for row in rows_sorted]
    ax.plot(thresholds, model_nb, color="#7b2cbf", linewidth=2.3, label="Model")
    ax.plot(thresholds, treat_all, linestyle="--", color="#9aa5b1", linewidth=1.4, label="Treat All")
    ax.plot(thresholds, [0.0 for _ in thresholds], linestyle=":", color="#343a40", linewidth=1.2, label="Treat None")
    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title("Exploratory decision curve analysis")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.text(
        0.97,
        0.04,
        "Exploratory only\n4 positive cases",
        transform=ax.transAxes,
        fontsize=7,
        ha="right",
        va="bottom",
        color="#7f1d1d",
        bbox={"facecolor": "white", "edgecolor": "#fecaca", "alpha": 0.9, "pad": 2},
    )


def make_main_panel(
    output_path: Path,
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    auc_label: str,
    pr_auc_label: str,
    threshold: float,
    brier: float,
    hl_p_value: float,
    calibration_rows: Sequence[Dict[str, str]],
    dca_rows: Sequence[Dict[str, str]],
) -> None:
    plt = require_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.45), dpi=600)
    plot_roc(axes[0, 0], y_true, y_score, auc_label, threshold)
    plot_pr(axes[0, 1], y_true, y_score, pr_auc_label)
    plot_calibration(axes[1, 0], calibration_rows, brier, hl_p_value)
    plot_dca(axes[1, 1], dca_rows)
    # Explicit geometry avoids tight/constrained-layout interactions that can displace
    # the DCA tick labels outside the lower-right panel on some Matplotlib versions.
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.105, top=0.925, wspace=0.40, hspace=0.42)
    for suffix, dpi in ((".png", 600), (".pdf", None), (".svg", None), (".tiff", 600)):
        export_path = output_path.with_suffix(suffix)
        save_kwargs = {"bbox_inches": "tight"}
        if dpi is not None:
            save_kwargs["dpi"] = dpi
        fig.savefig(export_path, **save_kwargs)
    plt.close(fig)


def make_confusion_matrix(output_path: Path, y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> None:
    plt = require_matplotlib()
    y_pred = (y_score >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(np.int64)
    total = max(1, int(cm.sum()))
    cm_pct = cm / total

    fig, ax = plt.subplots(figsize=(3.54, 3.10), dpi=600)
    image = ax.imshow(cm, cmap="YlOrRd")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    labels = np.array([["TN", "FP"], ["FN", "TP"]], dtype=object)
    for row_idx in range(2):
        for col_idx in range(2):
            ax.text(
                col_idx,
                row_idx,
                f"{labels[row_idx, col_idx]}\n{cm[row_idx, col_idx]}\n{cm_pct[row_idx, col_idx] * 100:.1f}%",
                ha="center",
                va="center",
                color="#1f2933",
                fontsize=10,
                fontweight="bold",
            )
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_title(f"Temporal held-out confusion matrix (thr={threshold:.3f})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def shorten_feature_name(name: str, max_len: int = 96) -> str:
    name = str(name)
    return name if len(name) <= max_len else name[: max_len - 3] + "..."


def make_top_feature_plot(output_path: Path, feature_rows: Sequence[Dict[str, str]], top_n: int) -> None:
    plt = require_matplotlib()
    rows = sorted(feature_rows, key=lambda row: safe_float(row.get("importance")), reverse=True)[:top_n]
    if not rows:
        return
    names = [shorten_feature_name(str(row.get("feature_name", ""))) for row in rows][::-1]
    values = [safe_float(row.get("importance")) for row in rows][::-1]
    colors = ["#8fbc8f" if str(row.get("feature_group", "")) == "clinical" else "#3a86ff" for row in rows][::-1]

    fig_height = max(6.0, 0.42 * len(rows) + 1.4)
    fig, ax = plt.subplots(figsize=(13.0, fig_height), dpi=300)
    ax.barh(names, values, color=colors)
    ax.set_xlabel("XGBoost Gain Importance")
    ax.set_title(f"Top {len(rows)} Feature Importances")
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(True, axis="x", alpha=0.25)
    fig.subplots_adjust(left=0.48, right=0.98, top=0.93, bottom=0.10)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def subgroup_label(row: Dict[str, str]) -> str:
    subgroup_name = str(row.get("subgroup_name", ""))
    subgroup_value = str(row.get("subgroup_value", ""))
    positives = safe_float(row.get("positives"))
    sparse_mark = "†" if np.isfinite(positives) and positives < 3 else ""
    if subgroup_name == "overall":
        return "Overall"
    label_map = {
        "age": "Age",
        "sex": "Sex",
        "smoking_history": "Smoking history",
        "metastasis": "Metastasis",
        "location_group": "Location group",
        "location_code": "Location code",
    }
    label_prefix = label_map.get(subgroup_name, subgroup_name.replace("_", " ").title())
    if subgroup_name == "age":
        return f"{label_prefix} {subgroup_value}{sparse_mark}"
    return f"{label_prefix} = {subgroup_value}{sparse_mark}"


def subgroup_sort_key(row: Dict[str, str]) -> Tuple[int, str]:
    order = {
        "overall": 0,
        "age": 1,
        "sex": 2,
        "smoking_history": 3,
        "metastasis": 4,
        "location_group": 5,
        "location_code": 6,
    }
    return (order.get(str(row.get("subgroup_name", "")), 99), subgroup_label(row))


def make_subgroup_forest(output_path: Path, subgroup_rows: Sequence[Dict[str, str]]) -> None:
    plt = require_matplotlib()
    rows = sorted(subgroup_rows, key=subgroup_sort_key)
    rows = [row for row in rows if np.isfinite(safe_float(row.get("auc"))) and np.isfinite(safe_float(row.get("sensitivity")))]
    if not rows:
        return

    labels = [subgroup_label(row) for row in rows]
    y_pos = np.arange(len(rows))[::-1]
    auc = np.asarray([safe_float(row.get("auc")) for row in rows], dtype=np.float64)
    auc_low = np.asarray([safe_float(row.get("auc_ci_lower")) for row in rows], dtype=np.float64)
    auc_high = np.asarray([safe_float(row.get("auc_ci_upper")) for row in rows], dtype=np.float64)
    sen = np.asarray([safe_float(row.get("sensitivity")) for row in rows], dtype=np.float64)
    sen_low = np.asarray([safe_float(row.get("sensitivity_ci_lower")) for row in rows], dtype=np.float64)
    sen_high = np.asarray([safe_float(row.get("sensitivity_ci_upper")) for row in rows], dtype=np.float64)

    fig_height = max(7.0, 0.42 * len(rows) + 1.8)
    fig, axes = plt.subplots(1, 2, figsize=(13, fig_height), dpi=600, sharey=True)

    axes[0].errorbar(
        auc,
        y_pos,
        xerr=np.vstack([np.maximum(0.0, auc - auc_low), np.maximum(0.0, auc_high - auc)]),
        fmt="o",
        color="#004c6d",
        ecolor="#7fb3d5",
        elinewidth=1.6,
        capsize=3,
    )
    axes[0].set_title("Subgroup AUC-ROC")
    axes[0].set_xlabel("AUC-ROC")
    axes[0].set_xlim(0.0, 1.02)
    axes[0].grid(True, axis="x", alpha=0.25)
    axes[0].text(
        -0.18,
        1.03,
        "A",
        transform=axes[0].transAxes,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    axes[1].errorbar(
        sen,
        y_pos,
        xerr=np.vstack([np.maximum(0.0, sen - sen_low), np.maximum(0.0, sen_high - sen)]),
        fmt="o",
        color="#2f7d32",
        ecolor="#95d5b2",
        elinewidth=1.6,
        capsize=3,
    )
    axes[1].set_title("Subgroup Sensitivity")
    axes[1].set_xlabel("Sensitivity")
    axes[1].set_xlim(0.0, 1.02)
    axes[1].grid(True, axis="x", alpha=0.25)
    axes[1].text(
        -0.18,
        1.03,
        "B",
        transform=axes[1].transAxes,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    axes[0].set_yticks(y_pos, labels=labels)
    axes[1].tick_params(labelleft=False)
    fig.text(
        0.01,
        0.005,
        "† Stratum contains fewer than 3 positive cases; estimates are descriptive only.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#4b5563",
    )
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.07, top=0.92, wspace=0.18)
    # Provide editable vector files and a 600-dpi TIFF alongside the PNG preview.
    for suffix in (".png", ".pdf", ".svg", ".tiff"):
        fig.savefig(output_path.with_suffix(suffix), dpi=600, bbox_inches="tight")
    plt.close(fig)


def make_feature_stability_plot(
    output_path: Path,
    feature_rows: Sequence[Dict[str, str]],
    *,
    main_branch_name: str,
    top_n: int,
    min_selection_count: int,
) -> None:
    plt = require_matplotlib()
    rows = [row for row in feature_rows if str(row.get("branch_name", "")) == main_branch_name]
    rows = [row for row in rows if int(safe_float(row.get("selection_count"))) >= int(min_selection_count)]
    rows = sorted(
        rows,
        key=lambda row: (-int(safe_float(row.get("selection_count"))), -safe_float(row.get("selection_rate")), str(row.get("feature_name", ""))),
    )[:top_n]
    if not rows:
        return

    # Feature IDs prevent long radiomics names from being clipped in the final figure.
    # Their full names are reported in Supplementary Table S7.
    labels = [f"F{index}" for index in range(1, len(rows) + 1)][::-1]
    selection_counts = [int(safe_float(row.get("selection_count"))) for row in rows][::-1]
    selection_rates = [safe_float(row.get("selection_rate")) for row in rows][::-1]

    fig_height = max(6.0, 0.42 * len(rows) + 1.4)
    fig, ax = plt.subplots(figsize=(7.1, fig_height), dpi=600)
    bars = ax.barh(labels, selection_counts, color="#fb8500", alpha=0.92)
    ax.set_xlabel("Feature reselection count (of 25 repeated CV folds)")
    ax.set_title("Feature reselection frequency across 25 repeated CV folds")
    # Reserve sufficient right-side space for the longest count/rate annotation.
    ax.set_xlim(0, 28.0)
    ax.grid(True, axis="x", alpha=0.25)
    for bar, count, rate in zip(bars, selection_counts, selection_rates):
        ax.text(
            bar.get_width() + 0.15,
            bar.get_y() + bar.get_height() / 2.0,
            f"{count}/25 ({rate:.2f})",
            va="center",
            ha="left",
            fontsize=8,
            color="#5c677d",
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    assert_frozen_runtime()
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
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger = build_logger(
        "egfr20ins_paper_figures",
        figures_dir / "egfr20ins_paper_figures.log",
        enable_file_logging=True,
        enable_stream_logging=True,
    )

    fixed_prediction_path = fixed_dir / "fixed_test_predictions.csv"
    fixed_summary_path = fixed_dir / "fixed_test_summary.csv"
    subgroup_summary_path = subgroup_dir / "subgroup_summary.csv"
    calibration_summary_path = calibration_dir / "calibration_summary.csv"
    calibration_points_path = calibration_dir / "calibration_curve_points.csv"
    dca_summary_path = calibration_dir / "dca_summary.csv"
    key_results_path = paper_dir / "paper_key_results.json"
    top_features_path = paper_dir / "paper_top_features.csv"
    feature_stability_path = paper_dir / "paper_feature_stability_table.csv"
    require_paths(
        [
            fixed_prediction_path,
            fixed_summary_path,
            subgroup_summary_path,
            calibration_summary_path,
            calibration_points_path,
            dca_summary_path,
            key_results_path,
            top_features_path,
        ]
    )

    prediction_rows = read_csv_rows(fixed_prediction_path)
    fixed_summary_rows = read_csv_rows(fixed_summary_path)
    subgroup_rows = read_csv_rows(subgroup_summary_path)
    calibration_summary_rows = read_csv_rows(calibration_summary_path)
    calibration_rows = read_csv_rows(calibration_points_path)
    dca_rows = read_csv_rows(dca_summary_path)
    key_results = read_json(key_results_path)
    feature_rows = read_csv_rows(top_features_path)
    feature_stability_rows = read_csv_rows(feature_stability_path) if feature_stability_path.is_file() else []

    main_branch_name = str(key_results["main_model_name"])
    threshold = safe_float(key_results.get("threshold"))
    y_true, y_score = load_main_predictions(prediction_rows, main_branch_name)
    fixed_summary_map = {str(row["branch_name"]): row for row in fixed_summary_rows}
    main_fixed_row = fixed_summary_map.get(main_branch_name)
    if main_fixed_row is None:
        raise KeyError(f"Fixed-test summary missing branch: {main_branch_name}")

    main_calibration_rows = [row for row in calibration_rows if str(row.get("branch_name")) == main_branch_name]
    calibration_summary_map = {str(row["branch_name"]): row for row in calibration_summary_rows}
    main_calibration_summary_row = calibration_summary_map.get(main_branch_name, {})
    main_dca_rows = [row for row in dca_rows if str(row.get("branch_name")) == main_branch_name]
    top_n = int(os.environ.get("LUAD_EGFR20INS_PAPER_TOP_FEATURES", "15"))
    top_stability_n = int(os.environ.get("LUAD_EGFR20INS_PAPER_STABILITY_TOP_N", "15"))
    min_stability_count = int(os.environ.get("LUAD_EGFR20INS_PAPER_STABILITY_MIN_COUNT", "3"))

    main_panel_path = figures_dir / "figure_main_panel_2x2.png"
    confusion_path = figures_dir / "figure_confusion_matrix.png"
    top_features_plot_path = figures_dir / "figure_top_feature_importance.png"
    subgroup_forest_path = figures_dir / "figure_subgroup_forest.png"
    feature_stability_plot_path = figures_dir / "figure_feature_stability.png"

    make_main_panel(
        main_panel_path,
        y_true=y_true,
        y_score=y_score,
        auc_label=str(key_results.get("fixed_test_auc", "")),
        pr_auc_label=str(key_results.get("fixed_test_pr_auc", "")),
        threshold=threshold,
        brier=safe_float(main_fixed_row.get("brier")),
        hl_p_value=safe_float(main_calibration_summary_row.get("hl_p_value")),
        calibration_rows=main_calibration_rows,
        dca_rows=main_dca_rows,
    )
    make_confusion_matrix(confusion_path, y_true, y_score, threshold)
    make_top_feature_plot(top_features_plot_path, feature_rows, top_n=top_n)
    make_subgroup_forest(subgroup_forest_path, subgroup_rows)
    make_feature_stability_plot(
        feature_stability_plot_path,
        feature_stability_rows,
        main_branch_name=main_branch_name,
        top_n=top_stability_n,
        min_selection_count=min_stability_count,
    )

    save_json(
        {
            "main_panel": str(main_panel_path),
            "confusion_matrix": str(confusion_path),
            "top_feature_plot": str(top_features_plot_path),
            "subgroup_forest": str(subgroup_forest_path),
            "feature_stability_plot": str(feature_stability_plot_path),
            "main_branch_name": main_branch_name,
            "threshold": threshold,
        },
        figures_dir / "paper_figures_manifest.json",
    )
    logger.info("EGFR20ins paper figures built | output_dir=%s", figures_dir)


if __name__ == "__main__":
    main()
