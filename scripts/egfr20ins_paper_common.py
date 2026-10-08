from __future__ import annotations

import csv
import json
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_egfr20ins_binary_strategies import (
    DataBundle,
    fit_rfe_xgboost_calibrated,
    get_threshold_min_recall,
    get_threshold_search_cv_splits,
    load_inputs,
    prepare_feature_blocks,
    select_threshold_from_scores,
)
from luad_tabular_utils import NumpyEncoder, build_logger, save_json


Array1D = np.ndarray
Array2D = np.ndarray


@dataclass(frozen=True)
class PaperBranchSpec:
    name: str
    display_name: str
    feature_block: str
    model_kind: str
    rfe_dim: Optional[int]
    seed: int


def default_branch_specs() -> List[PaperBranchSpec]:
    base_seed = int(os.environ.get("LUAD_EGFR20INS_PAPER_MODEL_SEED", "72"))
    rfe_dim = int(os.environ.get("LUAD_EGFR20INS_PAPER_RFE_DIM", "48"))
    rfe_pool = os.environ.get("LUAD_EGFR20INS_RFE_POOL", "merged").strip().lower()
    if rfe_pool != "merged":
        raise ValueError("The manuscript primary branch requires LUAD_EGFR20INS_RFE_POOL=merged.")
    return [
        PaperBranchSpec(
            name="clinical_only_logreg",
            display_name="Clinical Only",
            feature_block="clinical",
            model_kind="clinical_logreg",
            rfe_dim=None,
            seed=base_seed,
        ),
        PaperBranchSpec(
            name="radiomics_only_rfe_xgboost_calibrated",
            display_name="Radiomics Only",
            feature_block="radiomics",
            model_kind="rfe_xgboost_calibrated",
            rfe_dim=rfe_dim,
            seed=base_seed,
        ),
        PaperBranchSpec(
            name="fusion_rfe_xgboost_calibrated",
            display_name="Combined-candidate RFE",
            feature_block="fusion",
            model_kind="rfe_xgboost_calibrated",
            rfe_dim=rfe_dim,
            seed=base_seed,
        ),
    ]


def get_main_branch_spec() -> PaperBranchSpec:
    base_seed = int(os.environ.get("LUAD_EGFR20INS_PAPER_MODEL_SEED", "72"))
    rfe_dim = int(os.environ.get("LUAD_EGFR20INS_PAPER_RFE_DIM", "48"))
    rfe_pool = os.environ.get("LUAD_EGFR20INS_RFE_POOL", "merged").strip().lower()
    if rfe_pool != "merged":
        raise ValueError("The manuscript primary branch requires LUAD_EGFR20INS_RFE_POOL=merged.")
    return PaperBranchSpec(
        name="fusion_rfe_xgboost_calibrated",
        display_name="Combined-candidate RFE",
        feature_block="fusion",
        model_kind="rfe_xgboost_calibrated",
        rfe_dim=rfe_dim,
        seed=base_seed,
    )


def get_fixed_eval_seed(base_seed: int) -> int:
    return int(base_seed) + int(os.environ.get("LUAD_EGFR20INS_FIXED_SEED_OFFSET", "200"))


def safe_float(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def load_bundle_and_feature_blocks() -> Tuple[DataBundle, Array2D, Array2D, Array2D, List[str], List[str], List[str]]:
    bundle = load_inputs()
    x_radiomics, x_combined, combined_feature_names = prepare_feature_blocks(bundle)
    x_clinical = np.asarray(bundle.x_clinical, dtype=np.float32)
    return (
        bundle,
        np.asarray(x_radiomics, dtype=np.float32),
        x_clinical,
        np.asarray(x_combined, dtype=np.float32),
        list(bundle.radiomics_feature_names),
        list(bundle.clinical_feature_names),
        list(combined_feature_names),
    )


def bundle_metadata_array(
    bundle: DataBundle,
    key: str,
    *,
    fallback_index: Optional[int] = None,
    dtype: Optional[np.dtype] = None,
) -> Optional[np.ndarray]:
    metadata = getattr(bundle, "clinical_metadata", {})
    if isinstance(metadata, dict) and key in metadata:
        values = np.asarray(metadata[key])
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        return values
    if fallback_index is None:
        return None
    x_clinical = np.asarray(bundle.x_clinical)
    if x_clinical.ndim != 2 or fallback_index < 0 or fallback_index >= x_clinical.shape[1]:
        return None
    values = np.asarray(x_clinical[:, fallback_index])
    if dtype is not None:
        values = values.astype(dtype, copy=False)
    return values


def fixed_split_indices(bundle: DataBundle) -> Tuple[Array1D, Array1D]:
    y = np.asarray(bundle.y, dtype=np.int64)
    fixed_test_indices = np.asarray(bundle.fixed_test_indices, dtype=np.int64)
    fixed_mask = np.zeros(y.shape[0], dtype=bool)
    fixed_mask[fixed_test_indices] = True
    all_indices = np.arange(y.shape[0], dtype=np.int64)
    train_indices = all_indices[~fixed_mask]
    return train_indices, fixed_test_indices


def branch_matrix_and_names(
    branch: PaperBranchSpec,
    *,
    x_radiomics: Array2D,
    x_clinical: Array2D,
    x_combined: Array2D,
    radiomics_feature_names: Sequence[str],
    clinical_feature_names: Sequence[str],
    combined_feature_names: Sequence[str],
) -> Tuple[Array2D, List[str]]:
    if branch.feature_block == "radiomics":
        return np.asarray(x_radiomics, dtype=np.float32), [str(name) for name in radiomics_feature_names]
    if branch.feature_block == "clinical":
        return np.asarray(x_clinical, dtype=np.float32), [str(name) for name in clinical_feature_names]
    if branch.feature_block == "fusion":
        matrix = np.asarray(x_combined, dtype=np.float32)
        names = [str(name) for name in combined_feature_names]
        radiomics_count = len(radiomics_feature_names)
        clinical_count = len(clinical_feature_names)
        if matrix.shape[1] != radiomics_count + clinical_count:
            raise ValueError("Combined candidate matrix dimensions do not match radiomic and clinical feature names.")
        if matrix.shape[1] <= radiomics_count:
            raise ValueError("Combined-candidate RFE requires a merged radiomic-plus-clinical feature pool.")
        if branch.rfe_dim is not None and int(branch.rfe_dim) != 48:
            raise ValueError("The manuscript primary combined-candidate RFE branch is frozen at 48 features.")
        return matrix, names
    raise ValueError(f"Unsupported feature_block {branch.feature_block!r}.")


@contextmanager
def temporary_env(overrides: Dict[str, Optional[str]]) -> Iterator[None]:
    old_values: Dict[str, Optional[str]] = {}
    for key, value in overrides.items():
        old_values[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def fit_clinical_logreg(
    x_train: Array2D,
    y_train: Array1D,
    x_eval: Array2D,
    seed: int,
) -> Array1D:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_eval_scaled = scaler.transform(x_eval)
    model = LogisticRegression(
        solver="liblinear",
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=seed,
    )
    model.fit(x_train_scaled, y_train)
    return np.asarray(model.predict_proba(x_eval_scaled), dtype=np.float64)[:, 1]


def fit_branch_scores(
    branch: PaperBranchSpec,
    x_train: Array2D,
    y_train: Array1D,
    x_eval: Array2D,
    feature_names: Sequence[str],
    seed: int,
) -> Tuple[Array1D, Optional[Dict[str, object]]]:
    if branch.model_kind == "clinical_logreg":
        return fit_clinical_logreg(x_train, y_train, x_eval, seed), None
    if branch.model_kind == "rfe_xgboost_calibrated":
        env_overrides = {
            "LUAD_EGFR20INS_RFE_N_FEATURES": None if branch.rfe_dim is None else str(int(branch.rfe_dim)),
            "LUAD_EGFR20INS_RFE_POOL": "merged" if branch.feature_block == "fusion" else "radiomics",
        }
        with temporary_env(env_overrides):
            scores, payload = fit_rfe_xgboost_calibrated(
                x_train=x_train,
                y_train=y_train,
                x_eval=x_eval,
                seed=seed,
                feature_names=feature_names,
                capture_importance=True,
            )
        return np.asarray(scores, dtype=np.float64), payload
    raise ValueError(f"Unsupported model_kind {branch.model_kind!r}.")


def fit_branch_scores_without_importance(
    branch: PaperBranchSpec,
    x_train: Array2D,
    y_train: Array1D,
    x_eval: Array2D,
    feature_names: Sequence[str],
    seed: int,
) -> Array1D:
    if branch.model_kind == "clinical_logreg":
        return fit_clinical_logreg(x_train, y_train, x_eval, seed)
    env_overrides = {
        "LUAD_EGFR20INS_RFE_N_FEATURES": None if branch.rfe_dim is None else str(int(branch.rfe_dim)),
        "LUAD_EGFR20INS_RFE_POOL": "merged" if branch.feature_block == "fusion" else "radiomics",
    }
    with temporary_env(env_overrides):
        scores, _ = fit_rfe_xgboost_calibrated(
            x_train=x_train,
            y_train=y_train,
            x_eval=x_eval,
            seed=seed,
            feature_names=feature_names,
            capture_importance=False,
        )
    return np.asarray(scores, dtype=np.float64)


def search_branch_threshold(
    branch: PaperBranchSpec,
    x_train: Array2D,
    y_train: Array1D,
    feature_names: Sequence[str],
    seed: int,
) -> float:
    y_train = np.asarray(y_train, dtype=np.int64).reshape(-1)
    if y_train.size < 4 or np.unique(y_train).size < 2:
        return 0.5
    n_splits = get_threshold_search_cv_splits(y_train)
    if n_splits < 2:
        return 0.5
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_scores = np.full(y_train.shape[0], np.nan, dtype=np.float64)
    for inner_idx, (inner_train_idx, inner_val_idx) in enumerate(splitter.split(x_train, y_train)):
        scores = fit_branch_scores_without_importance(
            branch=branch,
            x_train=x_train[inner_train_idx],
            y_train=y_train[inner_train_idx],
            x_eval=x_train[inner_val_idx],
            feature_names=feature_names,
            seed=seed + 1000 + inner_idx,
        )
        oof_scores[inner_val_idx] = scores
    valid_mask = np.isfinite(oof_scores)
    if valid_mask.sum() == 0 or np.unique(y_train[valid_mask]).size < 2:
        return 0.5
    return float(
        select_threshold_from_scores(
            y_true=y_train[valid_mask],
            positive_scores=oof_scores[valid_mask],
            metric=os.environ.get("LUAD_EGFR20INS_THRESHOLD_METRIC", "recall_floor"),
        )
    )


def compute_binary_metrics_detailed(y_true: Array1D, positive_scores: Array1D, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    positive_scores = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    y_pred = (positive_scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both_classes = np.unique(y_true).size >= 2
    auc = float(roc_auc_score(y_true, positive_scores)) if has_both_classes else float("nan")
    pr_auc = float(average_precision_score(y_true, positive_scores)) if has_both_classes else float("nan")
    brier = float(brier_score_loss(y_true, positive_scores))
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    ppv = safe_div(tp, tp + fp)
    npv = safe_div(tn, tn + fn)
    accuracy = safe_div(tp + tn, len(y_true))
    balanced_accuracy = (
        float(np.nanmean(np.asarray([sensitivity, specificity], dtype=np.float64)))
        if np.isfinite(sensitivity) or np.isfinite(specificity)
        else float("nan")
    )
    f1 = safe_div(2.0 * tp, 2.0 * tp + fp + fn)
    prevalence = float(np.mean(y_true))
    return {
        "n": float(len(y_true)),
        "positives": float(int(y_true.sum())),
        "negatives": float(int((y_true == 0).sum())),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "auc": auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "threshold": float(threshold),
        "prevalence": prevalence,
    }


def summarize_metric_series(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray([float(value) for value in values if np.isfinite(value)], dtype=np.float64)
    if array.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "n": 0.0,
        }
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "ci_lower": float(np.percentile(array, 2.5)),
        "ci_upper": float(np.percentile(array, 97.5)),
        "n": float(array.size),
    }


def bootstrap_binary_metrics(
    y_true: Array1D,
    positive_scores: Array1D,
    threshold: float,
    n_bootstrap: int,
    random_state: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    positive_scores = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    base_metrics = compute_binary_metrics_detailed(y_true, positive_scores, threshold)
    metric_names = [
        "auc",
        "pr_auc",
        "brier",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "accuracy",
        "balanced_accuracy",
        "f1",
    ]
    samples: Dict[str, List[float]] = {metric_name: [] for metric_name in metric_names}
    max_attempts = max(1000, n_bootstrap * 50)
    attempts = 0
    while len(samples["accuracy"]) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        sample_indices = rng.integers(0, len(y_true), size=len(y_true))
        sample_true = y_true[sample_indices]
        sample_scores = positive_scores[sample_indices]
        if np.unique(sample_true).size < 2:
            continue
        sample_metrics = compute_binary_metrics_detailed(sample_true, sample_scores, threshold)
        if not np.isfinite(sample_metrics["auc"]):
            continue
        for metric_name in metric_names:
            metric_value = float(sample_metrics[metric_name])
            if np.isfinite(metric_value):
                samples[metric_name].append(metric_value)
    summary: Dict[str, object] = {
        "base_metrics": base_metrics,
        "valid_bootstrap_samples": len(samples["accuracy"]),
        "bootstrap_attempts": attempts,
        "n_bootstrap_requested": int(n_bootstrap),
    }
    for metric_name in metric_names:
        values = np.asarray(samples[metric_name], dtype=np.float64)
        if values.size == 0:
            summary[metric_name] = {
                "value": safe_float(base_metrics[metric_name]),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
            }
        else:
            summary[metric_name] = {
                "value": safe_float(base_metrics[metric_name]),
                "ci_lower": float(np.percentile(values, 2.5)),
                "ci_upper": float(np.percentile(values, 97.5)),
            }
    return summary


def format_metric_ci(value: float, lower: float, upper: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if not np.isfinite(lower) or not np.isfinite(upper):
        return f"{value:.3f}"
    return f"{value:.3f} ({lower:.3f}-{upper:.3f})"


def calc_ground_truth_statistics(ground_truth: Array1D, predictions: Array1D) -> Tuple[Array1D, int]:
    order = np.argsort(-predictions)
    label_1_count = int(np.sum(ground_truth))
    ordered_truth = np.asarray(ground_truth, dtype=np.int64)[order]
    return ordered_truth, label_1_count


def compute_midrank(x: Array1D) -> Array1D:
    sorted_idx = np.argsort(x)
    sorted_x = x[sorted_idx]
    midranks = np.zeros(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        midrank = 0.5 * (i + j - 1) + 1
        midranks[i:j] = midrank
        i = j
    out = np.empty(len(x), dtype=np.float64)
    out[sorted_idx] = midranks
    return out


def fast_delong(predictions_sorted_transposed: Array2D, label_1_count: int) -> Tuple[Array1D, Array2D]:
    m = int(label_1_count)
    n = int(predictions_sorted_transposed.shape[1] - m)
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = int(predictions_sorted_transposed.shape[0])

    tx = np.empty((k, m), dtype=np.float64)
    ty = np.empty((k, n), dtype=np.float64)
    tz = np.empty((k, m + n), dtype=np.float64)
    for row_idx in range(k):
        tx[row_idx, :] = compute_midrank(positive_examples[row_idx, :])
        ty[row_idx, :] = compute_midrank(negative_examples[row_idx, :])
        tz[row_idx, :] = compute_midrank(predictions_sorted_transposed[row_idx, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_roc_test(y_true: Array1D, pred_one: Array1D, pred_two: Array1D) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    pred_one = np.asarray(pred_one, dtype=np.float64)
    pred_two = np.asarray(pred_two, dtype=np.float64)
    if np.unique(y_true).size < 2:
        return {"auc_one": float("nan"), "auc_two": float("nan"), "z": float("nan"), "p": float("nan")}
    # fast_delong expects all positive examples before negative examples.
    # Sorting by a model score violates that precondition and can corrupt both
    # the reported AUCs and their paired variance.
    _, label_1_count = calc_ground_truth_statistics(y_true, pred_one)
    order = np.argsort(-y_true)
    preds = np.vstack([pred_one, pred_two])[:, order]
    aucs, covariance = fast_delong(preds, label_1_count)
    if np.ndim(covariance) == 0:
        variance = float(covariance)
    else:
        variance = float(covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1])
    variance = max(variance, 1e-12)
    z_value = float((aucs[0] - aucs[1]) / math.sqrt(variance))
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z_value))))
    return {
        "auc_one": float(aucs[0]),
        "auc_two": float(aucs[1]),
        "z": z_value,
        "p": p_value,
    }


def build_repeated_oof_predictions(
    branch: PaperBranchSpec,
    x: Array2D,
    y: Array1D,
    feature_names: Sequence[str],
    repeat_seeds: Sequence[int],
    n_splits: int,
    logger,
) -> Dict[str, object]:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    score_sums = np.zeros(y.shape[0], dtype=np.float64)
    score_counts = np.zeros(y.shape[0], dtype=np.int64)
    fold_thresholds: List[float] = []
    fold_rows: List[Dict[str, object]] = []
    for repeat_idx, repeat_seed in enumerate(repeat_seeds):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(repeat_seed))
        for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(splitter.split(x, y)):
            threshold = search_branch_threshold(
                branch=branch,
                x_train=x[fold_train_idx],
                y_train=y[fold_train_idx],
                feature_names=feature_names,
                seed=int(repeat_seed) + fold_idx,
            )
            scores = fit_branch_scores_without_importance(
                branch=branch,
                x_train=x[fold_train_idx],
                y_train=y[fold_train_idx],
                x_eval=x[fold_val_idx],
                feature_names=feature_names,
                seed=int(repeat_seed) + fold_idx,
            )
            score_sums[fold_val_idx] += scores
            score_counts[fold_val_idx] += 1
            fold_metrics = compute_binary_metrics_detailed(y[fold_val_idx], scores, threshold)
            fold_rows.append(
                {
                    "repeat": int(repeat_idx),
                    "repeat_seed": int(repeat_seed),
                    "fold": int(fold_idx),
                    "threshold": float(threshold),
                    "auc": float(fold_metrics["auc"]),
                    "pr_auc": float(fold_metrics["pr_auc"]),
                    "sensitivity": float(fold_metrics["sensitivity"]),
                    "specificity": float(fold_metrics["specificity"]),
                }
            )
            fold_thresholds.append(float(threshold))
            logger.info(
                "OOF repeat | branch=%s | repeat=%d | fold=%d | auc=%.5f | pr_auc=%.5f | threshold=%.5f",
                branch.name,
                repeat_idx,
                fold_idx,
                fold_metrics["auc"],
                fold_metrics["pr_auc"],
                threshold,
            )
    valid_mask = score_counts > 0
    mean_scores = np.full(y.shape[0], np.nan, dtype=np.float64)
    mean_scores[valid_mask] = score_sums[valid_mask] / score_counts[valid_mask]
    global_threshold = float(
        select_threshold_from_scores(
            y_true=y[valid_mask],
            positive_scores=mean_scores[valid_mask],
            metric=os.environ.get("LUAD_EGFR20INS_THRESHOLD_METRIC", "recall_floor"),
        )
    )
    overall_metrics = compute_binary_metrics_detailed(y[valid_mask], mean_scores[valid_mask], global_threshold)
    return {
        "scores": mean_scores,
        "counts": score_counts,
        "global_threshold": global_threshold,
        "overall_metrics": overall_metrics,
        "fold_rows": fold_rows,
        "fold_thresholds": fold_thresholds,
    }


def parse_location_group_mapping(raw_value: str) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for group_spec in str(raw_value).split(";"):
        group_spec = group_spec.strip()
        if not group_spec or ":" not in group_spec:
            continue
        group_name, raw_members = group_spec.split(":", 1)
        members = [member.strip() for member in raw_members.split(",") if member.strip()]
        if members:
            mapping[group_name.strip()] = members
    return mapping


def subgroup_rows_from_clinical(bundle: DataBundle, subset_indices: Optional[Sequence[int]] = None) -> List[Dict[str, object]]:
    y = np.asarray(bundle.y, dtype=np.int64)
    indices = np.arange(y.shape[0], dtype=np.int64) if subset_indices is None else np.asarray(subset_indices, dtype=np.int64)
    age_all = bundle_metadata_array(bundle, "age", fallback_index=0, dtype=np.float32)
    sex_all = bundle_metadata_array(bundle, "sex", fallback_index=1, dtype=np.int64)
    smoking_all = bundle_metadata_array(bundle, "smoking_history", fallback_index=2, dtype=np.int64)
    location_code_all = bundle_metadata_array(bundle, "location_code", fallback_index=3, dtype=np.int64)
    raw_location_all = bundle_metadata_array(bundle, "location_raw")
    metastasis_all = bundle_metadata_array(bundle, "metastasis", dtype=np.int64)

    if age_all is None or sex_all is None or smoking_all is None or location_code_all is None:
        raise ValueError("Missing required clinical metadata for subgroup analysis.")

    age = age_all[indices]
    sex = sex_all[indices]
    smoking = smoking_all[indices]
    location_code = location_code_all[indices]
    raw_location = raw_location_all[indices] if raw_location_all is not None else location_code.astype(object)
    metastasis = metastasis_all[indices] if metastasis_all is not None else None

    age_cutoff_raw = os.environ.get("LUAD_EGFR20INS_SUBGROUP_AGE_CUTOFF", "median").strip().lower()
    age_cutoff = float(np.median(age)) if age_cutoff_raw == "median" else float(age_cutoff_raw)

    subgroup_rows: List[Dict[str, object]] = [
        {"subgroup_name": "overall", "subgroup_value": "all", "indices": indices.tolist()},
        {"subgroup_name": "age", "subgroup_value": f"<{age_cutoff:.1f}", "indices": indices[age < age_cutoff].tolist()},
        {"subgroup_name": "age", "subgroup_value": f">={age_cutoff:.1f}", "indices": indices[age >= age_cutoff].tolist()},
    ]

    for value in sorted(np.unique(sex).tolist()):
        subgroup_rows.append(
            {"subgroup_name": "sex", "subgroup_value": str(int(value)), "indices": indices[sex == int(value)].tolist()}
        )
    for value in sorted(np.unique(smoking).tolist()):
        subgroup_rows.append(
            {
                "subgroup_name": "smoking_history",
                "subgroup_value": str(int(value)),
                "indices": indices[smoking == int(value)].tolist(),
            }
        )

    if metastasis is not None:
        for value in sorted(np.unique(metastasis).tolist()):
            subgroup_rows.append(
                {
                    "subgroup_name": "metastasis",
                    "subgroup_value": str(int(value)),
                    "indices": indices[metastasis == int(value)].tolist(),
                }
            )

    location_group_mapping = parse_location_group_mapping(os.environ.get("LUAD_EGFR20INS_SUBGROUP_LOCATION_MAP", ""))
    if location_group_mapping:
        raw_location_str = np.asarray([str(value) for value in raw_location], dtype=object)
        location_code_str = np.asarray([str(int(value)) for value in location_code], dtype=object)
        for group_name, members in location_group_mapping.items():
            member_set = {str(member) for member in members}
            mask = np.asarray(
                [(raw_value in member_set) or (code_value in member_set) for raw_value, code_value in zip(raw_location_str, location_code_str)],
                dtype=bool,
            )
            subgroup_rows.append(
                {
                    "subgroup_name": "location_group",
                    "subgroup_value": group_name,
                    "indices": indices[mask].tolist(),
                }
            )
    else:
        for value in sorted(np.unique(location_code).tolist()):
            subgroup_rows.append(
                {
                    "subgroup_name": "location_code",
                    "subgroup_value": str(int(value)),
                    "indices": indices[location_code == int(value)].tolist(),
                }
            )
    return subgroup_rows


def write_rows_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
