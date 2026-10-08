"""
Clinical translation utilities based on the fixed-split test predictions.

Outputs:
    - results/figures/nomogram_*.png
    - results/figures/calibration_curve.png
    - results/figures/dca_curve.png
    - results/clinical_tools_results.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

from path_defaults import DEFAULT_TRAIN_FEATURE_DIR, LABEL_CSV as DEFAULT_LABEL_CSV


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FEATURE_DIR = DEFAULT_TRAIN_FEATURE_DIR
LABEL_CSV = DEFAULT_LABEL_CSV
PRIMARY_COLOR = "#2E74B5"
SECONDARY_COLOR = "#ED7D31"
EGFR_CLASS_NAMES: Tuple[str, ...] = ("19-del", "20-ins", "21-L858R")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):  # type: ignore[override]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def load_fixed_predictions() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    fixed_payload = np.load(RESULTS_DIR / "fixed_test_predictions.npy", allow_pickle=True).item()
    probabilities = np.asarray(fixed_payload["probabilities"], dtype=np.float64)
    labels = np.asarray(fixed_payload["labels"], dtype=np.int64)
    patient_ids = list(fixed_payload["patient_ids"]) if "patient_ids" in fixed_payload else []
    if not patient_ids:
        patient_ids = [f"P_{188 + idx:03d}" for idx in range(len(labels))]
    return probabilities, labels, patient_ids


def load_test_clinical(patient_ids: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    clinical = np.load(FEATURE_DIR / "test_clinical.npy")
    feature_names = ["sex", "age_scaled", "location_pca_1", "smoking_history", "metastasis"]
    if len(patient_ids) != clinical.shape[0]:
        raise ValueError("Mismatch between fixed prediction patient_ids and test_clinical rows.")
    return np.asarray(clinical, dtype=np.float64), feature_names


def hosmer_lemeshow_test(y_true: np.ndarray, y_prob: np.ndarray, g: int = 10) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    order = np.argsort(y_prob)
    y_true = y_true[order]
    y_prob = y_prob[order]

    bins = np.array_split(np.arange(len(y_true)), g)
    hl_stat = 0.0
    valid_groups = 0
    for group in bins:
        if len(group) == 0:
            continue
        obs = float(np.sum(y_true[group]))
        exp = float(np.sum(y_prob[group]))
        n = float(len(group))
        exp = min(max(exp, 1e-6), n - 1e-6)
        hl_stat += ((obs - exp) ** 2) / exp + (((n - obs) - (n - exp)) ** 2) / (n - exp)
        valid_groups += 1

    dof = max(valid_groups - 2, 1)
    p_value = float(1.0 - chi2.cdf(hl_stat, df=dof))
    return {"statistic": float(hl_stat), "p_value": p_value, "dof": int(dof)}


def compute_dca(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> Dict[str, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = len(y_true)

    model_benefit = []
    treat_all_benefit = []
    prevalence = float(np.mean(y_true))

    for threshold in thresholds:
        if threshold <= 0 or threshold >= 1:
            model_benefit.append(0.0)
            treat_all_benefit.append(0.0)
            continue
        pred_positive = y_prob >= threshold
        tp = float(np.sum(pred_positive & (y_true == 1)))
        fp = float(np.sum(pred_positive & (y_true == 0)))
        net_benefit = tp / n - fp / n * (threshold / (1.0 - threshold))
        model_benefit.append(net_benefit)
        treat_all_benefit.append(prevalence - (1.0 - prevalence) * (threshold / (1.0 - threshold)))

    return {
        "thresholds": thresholds,
        "model_net_benefit": np.asarray(model_benefit, dtype=np.float64),
        "treat_all_net_benefit": np.asarray(treat_all_benefit, dtype=np.float64),
        "treat_none_net_benefit": np.zeros_like(thresholds, dtype=np.float64),
    }


def plot_nomogram_like(
    coefficients: np.ndarray,
    feature_names: Sequence[str],
    class_name: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(np.abs(coefficients))
    ordered_coef = coefficients[order]
    ordered_names = [feature_names[idx] for idx in order]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    colors = [PRIMARY_COLOR if value >= 0 else SECONDARY_COLOR for value in ordered_coef]
    ax.barh(np.arange(len(ordered_coef)), ordered_coef, color=colors, alpha=0.9)
    ax.set_yticks(np.arange(len(ordered_coef)))
    ax.set_yticklabels(ordered_names)
    ax.set_xlabel("Logistic Coefficient")
    ax.set_title(f"Nomogram Proxy: {class_name}")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_calibration_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
) -> Dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    results: Dict[str, object] = {}

    for class_idx, ax in enumerate(axes):
        binary_true = (labels == class_idx).astype(np.int64)
        prob_true, prob_pred = calibration_curve(binary_true, probabilities[:, class_idx], n_bins=10, strategy="quantile")
        hl = hosmer_lemeshow_test(binary_true, probabilities[:, class_idx], g=10)
        ax.plot(prob_pred, prob_true, marker="o", color=PRIMARY_COLOR, linewidth=2)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2)
        ax.set_title(f"{EGFR_CLASS_NAMES[class_idx]} | HL p={hl['p_value']:.3f}")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Frequency")
        ax.grid(True, alpha=0.25)
        results[EGFR_CLASS_NAMES[class_idx]] = {
            "prob_true": prob_true,
            "prob_pred": prob_pred,
            "hosmer_lemeshow": hl,
        }

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return results


def plot_dca_curve(labels: np.ndarray, probabilities: np.ndarray, output_path: Path) -> Dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = np.linspace(0.01, 0.99, 99)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    dca_results: Dict[str, object] = {}

    for class_idx, ax in enumerate(axes):
        binary_true = (labels == class_idx).astype(np.int64)
        dca = compute_dca(binary_true, probabilities[:, class_idx], thresholds)
        ax.plot(thresholds, dca["model_net_benefit"], color=PRIMARY_COLOR, linewidth=2, label="Model")
        ax.plot(thresholds, dca["treat_all_net_benefit"], color=SECONDARY_COLOR, linewidth=2, linestyle="--", label="Treat All")
        ax.plot(thresholds, dca["treat_none_net_benefit"], color="gray", linewidth=1.5, linestyle=":", label="Treat None")
        ax.set_title(EGFR_CLASS_NAMES[class_idx])
        ax.set_xlabel("Threshold Probability")
        ax.set_ylabel("Net Benefit")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        dca_results[EGFR_CLASS_NAMES[class_idx]] = dca

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return dca_results


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    probabilities, labels, patient_ids = load_fixed_predictions()
    clinical, clinical_feature_names = load_test_clinical(patient_ids)

    nomogram_results: Dict[str, object] = {}
    for class_idx, class_name in enumerate(EGFR_CLASS_NAMES):
        binary_true = (labels == class_idx).astype(np.int64)
        combined_features = np.concatenate([probabilities, clinical], axis=1)
        feature_names = [f"prob_class_{i}" for i in range(probabilities.shape[1])] + list(clinical_feature_names)
        clf = LogisticRegression(max_iter=2000, solver="lbfgs")
        clf.fit(combined_features, binary_true)
        coefficients = clf.coef_.reshape(-1)
        plot_nomogram_like(coefficients, feature_names, class_name, FIGURES_DIR / f"nomogram_{class_name}.png")
        nomogram_results[class_name] = {
            "intercept": float(clf.intercept_[0]),
            "coefficients": coefficients,
            "feature_names": feature_names,
        }

    calibration_results = plot_calibration_curve(labels, probabilities, FIGURES_DIR / "calibration_curve.png")
    dca_results = plot_dca_curve(labels, probabilities, FIGURES_DIR / "dca_curve.png")

    report = {
        "patient_ids": patient_ids,
        "nomogram": nomogram_results,
        "calibration": calibration_results,
        "dca": dca_results,
    }
    (RESULTS_DIR / "clinical_tools_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, cls=NumpyEncoder),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
