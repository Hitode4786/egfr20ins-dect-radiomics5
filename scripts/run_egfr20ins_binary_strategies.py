from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import StackingClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from luad_tabular_utils import (
    NumpyEncoder,
    build_logger,
    load_full_radiomics_by_patient,
    load_labels,
    save_json,
    split_patient_ids,
)
from egfr20ins_runtime_guard import assert_frozen_runtime


Array2D = np.ndarray
Array1D = np.ndarray


@dataclass(frozen=True)
class DataBundle:
    x_radiomics: Array2D
    x_clinical: Array2D
    y: Array1D
    fixed_test_indices: Array1D
    sample_ids: List[str]
    radiomics_feature_names: List[str]
    clinical_feature_names: List[str]
    clinical_metadata: Dict[str, Array1D] = field(default_factory=dict)


def require_xgboost():
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "xgboost is required for models: rfe_xgboost, rfe_xgboost_calibrated, balanced_bagging_xgboost, stacking."
        ) from exc
    return XGBClassifier


def require_balanced_bagging():
    try:
        from imblearn.ensemble import BalancedBaggingClassifier
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "imbalanced-learn is required for model 'balanced_bagging_xgboost'. "
            "Install it with: conda install -n py310 -c conda-forge imbalanced-learn"
        ) from exc
    return BalancedBaggingClassifier


def load_numpy_array(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing array file: {path}")
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            keys = list(payload.keys())
            if len(keys) != 1:
                raise ValueError(f"Expected exactly one array in {path}, found keys={keys}")
            return np.asarray(payload[keys[0]])
    return np.asarray(np.load(path, allow_pickle=False))


def encode_location_column(values: Sequence[object]) -> np.ndarray:
    unique_values = sorted({str(value) for value in values})
    mapping = {value: idx for idx, value in enumerate(unique_values)}
    return np.asarray([mapping[str(value)] for value in values], dtype=np.float32)


def load_from_project_mode() -> DataBundle:
    label_csv = Path(
        os.environ.get("LUAD_EGFR20INS_LABEL_CSV", str(PROJECT_ROOT / "data" / "labels.csv"))
    )
    feature_dir = Path(
        os.environ.get("LUAD_EGFR20INS_FEATURE_DIR", str(PROJECT_ROOT / "data" / "features_mask_20260713"))
    )

    labels_df = load_labels(str(label_csv)).sort_values("patient_num").reset_index(drop=True)
    radiomics_by_patient, feature_bundle = load_full_radiomics_by_patient(str(feature_dir), labels_df)
    sample_ids = [str(patient_id) for patient_id in labels_df["patient_id"].tolist()]

    x_radiomics = np.vstack([np.asarray(radiomics_by_patient[patient_id], dtype=np.float32) for patient_id in sample_ids])
    x_clinical = np.column_stack(
        [
            labels_df["age"].to_numpy(dtype=np.float32),
            labels_df["sex"].to_numpy(dtype=np.float32),
            labels_df["smoking_history"].to_numpy(dtype=np.float32),
            encode_location_column(labels_df["location"].tolist()),
        ]
    ).astype(np.float32, copy=False)
    y = (labels_df["egfr_status"].to_numpy(dtype=np.int64) == 1).astype(np.int64, copy=False)

    split_ids = split_patient_ids(labels_df)
    fixed_test_set = {str(patient_id) for patient_id in split_ids["fixed_test"]}
    fixed_test_indices = np.flatnonzero(np.asarray([patient_id in fixed_test_set for patient_id in sample_ids], dtype=bool)).astype(np.int64)

    radiomics_feature_names = [str(name) for name in feature_bundle.get("full_feature_names", [])]  # type: ignore[arg-type]
    if len(radiomics_feature_names) != x_radiomics.shape[1]:
        radiomics_feature_names = [f"radiomics_{idx}" for idx in range(x_radiomics.shape[1])]

    return DataBundle(
        x_radiomics=x_radiomics,
        x_clinical=x_clinical,
        y=y,
        fixed_test_indices=fixed_test_indices,
        sample_ids=sample_ids,
        radiomics_feature_names=radiomics_feature_names,
        clinical_feature_names=["age", "sex", "smoking_history", "location_code"],
        clinical_metadata={
            "age": labels_df["age"].to_numpy(dtype=np.float32),
            "sex": labels_df["sex"].to_numpy(dtype=np.int64),
            "smoking_history": labels_df["smoking_history"].to_numpy(dtype=np.int64),
            "location_raw": labels_df["location"].to_numpy(dtype=object),
            "location_code": encode_location_column(labels_df["location"].tolist()).astype(np.int64, copy=False),
            "metastasis": labels_df["metastasis"].to_numpy(dtype=np.int64),
        },
    )


def load_from_array_mode() -> DataBundle:
    radiomics_path = os.environ.get("LUAD_EGFR20INS_X_RADIOMICS_PATH")
    clinical_path = os.environ.get("LUAD_EGFR20INS_X_CLINICAL_PATH")
    y_path = os.environ.get("LUAD_EGFR20INS_Y_PATH")
    fixed_test_path = os.environ.get("LUAD_EGFR20INS_FIXED_TEST_INDICES_PATH")

    if not all([radiomics_path, clinical_path, y_path, fixed_test_path]):
        raise RuntimeError(
            "Array mode requires LUAD_EGFR20INS_X_RADIOMICS_PATH, LUAD_EGFR20INS_X_CLINICAL_PATH, "
            "LUAD_EGFR20INS_Y_PATH, and LUAD_EGFR20INS_FIXED_TEST_INDICES_PATH."
        )

    x_radiomics = np.asarray(load_numpy_array(Path(str(radiomics_path))), dtype=np.float32)
    x_clinical = np.asarray(load_numpy_array(Path(str(clinical_path))), dtype=np.float32)
    y = np.asarray(load_numpy_array(Path(str(y_path))), dtype=np.int64).reshape(-1)
    fixed_test_indices = np.asarray(load_numpy_array(Path(str(fixed_test_path))), dtype=np.int64).reshape(-1)

    if x_radiomics.shape[0] != x_clinical.shape[0] or x_radiomics.shape[0] != y.shape[0]:
        raise ValueError("Input arrays must share the same first dimension.")

    sample_ids = [f"sample_{idx:03d}" for idx in range(y.shape[0])]
    return DataBundle(
        x_radiomics=x_radiomics,
        x_clinical=x_clinical,
        y=y,
        fixed_test_indices=fixed_test_indices,
        sample_ids=sample_ids,
        radiomics_feature_names=[f"radiomics_{idx}" for idx in range(x_radiomics.shape[1])],
        clinical_feature_names=[f"clinical_{idx}" for idx in range(x_clinical.shape[1])],
        clinical_metadata={},
    )


def load_inputs() -> DataBundle:
    data_mode = os.environ.get("LUAD_EGFR20INS_DATA_MODE", "project").strip().lower()
    if data_mode == "project":
        return load_from_project_mode()
    if data_mode == "array":
        return load_from_array_mode()
    raise ValueError("LUAD_EGFR20INS_DATA_MODE must be 'project' or 'array'.")


def xgb_binary_classifier(seed: int) -> Any:
    XGBClassifier = require_xgboost()
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=int(os.environ.get("LUAD_EGFR20INS_XGB_N_ESTIMATORS", "300")),
        max_depth=int(os.environ.get("LUAD_EGFR20INS_XGB_MAX_DEPTH", "3")),
        learning_rate=float(os.environ.get("LUAD_EGFR20INS_XGB_LEARNING_RATE", "0.03")),
        subsample=float(os.environ.get("LUAD_EGFR20INS_XGB_SUBSAMPLE", "1.0")),
        colsample_bytree=float(os.environ.get("LUAD_EGFR20INS_XGB_COLSAMPLE", "0.8")),
        min_child_weight=float(os.environ.get("LUAD_EGFR20INS_XGB_MIN_CHILD_WEIGHT", "1.0")),
        reg_alpha=float(os.environ.get("LUAD_EGFR20INS_XGB_REG_ALPHA", "0.0")),
        reg_lambda=float(os.environ.get("LUAD_EGFR20INS_XGB_REG_LAMBDA", "1.0")),
        gamma=float(os.environ.get("LUAD_EGFR20INS_XGB_GAMMA", "0.0")),
        random_state=seed,
        n_jobs=int(os.environ.get("LUAD_EGFR20INS_N_JOBS", "8")),
    )


def balanced_bagging_xgb(seed: int) -> Any:
    BalancedBaggingClassifier = require_balanced_bagging()
    estimator = xgb_binary_classifier(seed)
    if hasattr(estimator, "set_params"):
        estimator.set_params(n_jobs=1)
    kwargs = {
        "n_estimators": int(os.environ.get("LUAD_EGFR20INS_BAGGING_N_ESTIMATORS", "50")),
        "sampling_strategy": float(os.environ.get("LUAD_EGFR20INS_BAGGING_SAMPLING_RATIO", "1.0")),
        "bootstrap": False,
        "replacement": False,
        "random_state": seed,
        "n_jobs": int(os.environ.get("LUAD_EGFR20INS_N_JOBS", "8")),
    }
    try:
        return BalancedBaggingClassifier(estimator=estimator, **kwargs)
    except TypeError:
        return BalancedBaggingClassifier(base_estimator=estimator, **kwargs)


def build_stacking_classifier(seed: int) -> StackingClassifier:
    estimators = [
        ("xgb_seed42", xgb_binary_classifier(42)),
        ("xgb_seed52", xgb_binary_classifier(52)),
        ("xgb_seed62", xgb_binary_classifier(62)),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=int(os.environ.get("LUAD_EGFR20INS_N_JOBS", "8")),
            ),
        ),
        (
            "l1_logreg",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            penalty="elasticnet",
                            l1_ratio=1.0,
                            solver="saga",
                            C=0.1,
                            class_weight="balanced",
                            max_iter=5000,
                            random_state=seed,
                            n_jobs=int(os.environ.get("LUAD_EGFR20INS_N_JOBS", "8")),
                        ),
                    ),
                ]
            ),
        ),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            solver="lbfgs",
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        ),
        stack_method="predict_proba",
        cv=5,
        n_jobs=int(os.environ.get("LUAD_EGFR20INS_N_JOBS", "8")),
        passthrough=False,
    )


def prepare_feature_blocks(bundle: DataBundle) -> Tuple[Array2D, Array2D, List[str]]:
    x_radiomics = np.asarray(bundle.x_radiomics, dtype=np.float32)
    x_clinical = np.asarray(bundle.x_clinical, dtype=np.float32)
    combined = np.hstack([x_radiomics, x_clinical]).astype(np.float32, copy=False)
    feature_names = [*bundle.radiomics_feature_names, *bundle.clinical_feature_names]
    return x_radiomics, combined, feature_names


def compute_metrics(y_true: Array1D, positive_scores: Array1D, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    positive_scores = np.asarray(positive_scores, dtype=np.float64)
    auc = float(roc_auc_score(y_true, positive_scores))
    pr_auc = float(average_precision_score(y_true, positive_scores))
    y_pred = (positive_scores >= threshold).astype(np.int64)
    recall_20ins = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    return {"auc": auc, "pr_auc": pr_auc, "recall_20ins": recall_20ins, "threshold": float(threshold)}


def select_threshold_from_scores(y_true: Array1D, positive_scores: Array1D, metric: str) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    positive_scores = np.asarray(positive_scores, dtype=np.float64).reshape(-1)

    if y_true.size == 0 or np.unique(y_true).size < 2:
        return 0.5

    metric = str(metric).strip().lower()
    if metric == "recall_floor":
        target_recall = float(os.environ.get("LUAD_EGFR20INS_THRESHOLD_MIN_RECALL", "0.75"))
        precision, recall, thresholds = precision_recall_curve(y_true, positive_scores)
        if thresholds.size == 0:
            return 0.5
        precision = precision[:-1]
        recall = recall[:-1]
        valid_indices = np.flatnonzero(recall >= target_recall)
        if valid_indices.size > 0:
            best_idx = valid_indices[int(np.argmax(thresholds[valid_indices]))]
            return float(thresholds[best_idx])
        best_idx = int(np.argmax(recall))
        return float(thresholds[best_idx])
    if metric == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, positive_scores)
        if thresholds.size <= 1:
            return 0.5
        j_scores = tpr[1:] - fpr[1:]
        return float(thresholds[1:][int(np.nanargmax(j_scores))])

    beta = 2.0 if metric == "f2" else 1.0
    precision, recall, thresholds = precision_recall_curve(y_true, positive_scores)
    if thresholds.size == 0:
        return 0.5
    precision = precision[:-1]
    recall = recall[:-1]
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    fbeta = np.divide((1.0 + beta_sq) * precision * recall, denom, out=np.zeros_like(precision), where=denom > 0.0)
    return float(thresholds[int(np.nanargmax(fbeta))])


def get_threshold_search_metric() -> str:
    return os.environ.get("LUAD_EGFR20INS_THRESHOLD_METRIC", "recall_floor").strip().lower()


def get_threshold_min_recall() -> float:
    return float(os.environ.get("LUAD_EGFR20INS_THRESHOLD_MIN_RECALL", "0.75"))


def get_threshold_search_cv_splits(y_train: Array1D) -> int:
    requested = int(os.environ.get("LUAD_EGFR20INS_THRESHOLD_CV", "3"))
    counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=2)
    feasible = int(counts.min())
    if feasible < 2:
        return 0
    return max(2, min(requested, feasible))


def get_calibration_method() -> str:
    return os.environ.get("LUAD_EGFR20INS_CALIBRATION_METHOD", "sigmoid").strip().lower()


def get_calibration_cv_splits(y_train: Array1D) -> int:
    requested = int(os.environ.get("LUAD_EGFR20INS_CALIBRATION_CV", "3"))
    counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=2)
    feasible = int(counts.min())
    if feasible < 2:
        return 0
    return max(2, min(requested, feasible))


def build_calibrated_classifier(estimator: Any, method: str, cv: int) -> Any:
    try:
        return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, method=method, cv=cv)


def search_threshold_from_inner_cv(
    model_name: str,
    x_train: Array2D,
    y_train: Array1D,
    seed: int,
    feature_names: Sequence[str],
) -> Tuple[float, Array1D]:
    y_train = np.asarray(y_train, dtype=np.int64).reshape(-1)
    if y_train.size < 4 or np.unique(y_train).size < 2:
        return 0.5, np.full(y_train.shape[0], np.nan, dtype=np.float64)

    n_splits = get_threshold_search_cv_splits(y_train)
    if n_splits < 2:
        return 0.5, np.full(y_train.shape[0], np.nan, dtype=np.float64)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_scores = np.full(y_train.shape[0], np.nan, dtype=np.float64)

    for inner_idx, (inner_train_idx, inner_val_idx) in enumerate(splitter.split(x_train, y_train)):
        scores, _ = fit_model_scores(
            model_name=model_name,
            x_train=x_train[inner_train_idx],
            y_train=y_train[inner_train_idx],
            x_eval=x_train[inner_val_idx],
            seed=seed + 1000 + inner_idx,
            feature_names=feature_names,
            capture_importance=False,
        )
        oof_scores[inner_val_idx] = np.asarray(scores, dtype=np.float64)

    valid_mask = np.isfinite(oof_scores)
    if valid_mask.sum() == 0 or np.unique(y_train[valid_mask]).size < 2:
        return 0.5, oof_scores

    threshold = select_threshold_from_scores(
        y_true=y_train[valid_mask],
        positive_scores=oof_scores[valid_mask],
        metric=get_threshold_search_metric(),
    )
    return threshold, oof_scores


def fit_anomaly_model(
    model_name: str,
    x_train_neg: Array2D,
    x_eval: Array2D,
    seed: int,
) -> Array1D:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_neg)
    x_eval_scaled = scaler.transform(x_eval)

    if model_name == "anomaly_isolation_forest":
        model = IsolationForest(
            n_estimators=300,
            contamination=float(os.environ.get("LUAD_EGFR20INS_IFOREST_CONTAMINATION", "0.1")),
            random_state=seed,
            n_jobs=int(os.environ.get("LUAD_EGFR20INS_N_JOBS", "8")),
        )
    elif model_name == "anomaly_oneclass_svm":
        model = OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=float(os.environ.get("LUAD_EGFR20INS_OCSVM_NU", "0.1")),
        )
    else:
        raise ValueError(f"Unsupported anomaly model {model_name!r}.")

    model.fit(x_train_scaled)
    raw_scores = model.decision_function(x_eval_scaled).reshape(-1)
    return -np.asarray(raw_scores, dtype=np.float64)


def fit_rfe_xgboost(
    x_train: Array2D,
    y_train: Array1D,
    x_eval: Array2D,
    seed: int,
    feature_names: Sequence[str],
    capture_importance: bool = True,
) -> Tuple[Array1D, Dict[str, object]]:
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
        n_features_to_select=int(os.environ.get("LUAD_EGFR20INS_RFE_N_FEATURES", "32")),
        step=0.1,
    )
    x_train_selected = rfe.fit_transform(x_train_scaled, y_train)
    x_eval_selected = rfe.transform(x_eval_scaled)

    model = xgb_binary_classifier(seed)
    model.fit(x_train_selected, y_train)
    probabilities = np.asarray(model.predict_proba(x_eval_selected), dtype=np.float64)[:, 1]

    if not capture_importance:
        return probabilities, {}

    selected_indices = np.flatnonzero(rfe.support_)
    importance_payload = {
        "selected_feature_indices": [int(index) for index in selected_indices.tolist()],
        "selected_feature_names": [str(feature_names[int(index)]) for index in selected_indices.tolist()],
        "xgboost_feature_importances": [
            {
                "feature_name": str(feature_names[int(selected_indices[idx])]),
                "importance": float(value),
            }
            for idx, value in sorted(
                enumerate(np.asarray(model.feature_importances_, dtype=np.float64)),
                key=lambda item: item[1],
                reverse=True,
            )
        ],
    }
    return probabilities, importance_payload


def fit_rfe_xgboost_calibrated(
    x_train: Array2D,
    y_train: Array1D,
    x_eval: Array2D,
    seed: int,
    feature_names: Sequence[str],
    capture_importance: bool = True,
) -> Tuple[Array1D, Dict[str, object]]:
    rfe_pool = os.environ.get("LUAD_EGFR20INS_RFE_POOL", "radiomics").strip().lower()
    if rfe_pool not in {"radiomics", "merged"}:
        raise ValueError("LUAD_EGFR20INS_RFE_POOL must be 'radiomics' or 'merged'.")
    rfe_dim = int(os.environ.get("LUAD_EGFR20INS_RFE_N_FEATURES", "32"))
    if rfe_pool == "merged" and rfe_dim != 48:
        raise ValueError("The merged primary analysis RFE pool is frozen at 48 selected features.")
    if rfe_pool == "merged" and len(feature_names) != x_train.shape[1]:
        raise ValueError("Feature names do not align with the merged RFE matrix.")

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
        n_features_to_select=rfe_dim,
        step=0.1,
    )
    x_train_selected = rfe.fit_transform(x_train_scaled, y_train)
    x_eval_selected = rfe.transform(x_eval_scaled)
    selected_indices = np.flatnonzero(rfe.support_)
    if rfe_pool == "merged":
        clinical_names = {"age", "sex", "smoking_history", "location_code"}
        radiomics_count = len(feature_names) - len(clinical_names)
        if set(str(name) for name in feature_names[-4:]) != clinical_names:
            raise RuntimeError("Merged RFE pool must end with the four prespecified clinical candidates.")
        if len(selected_indices) != 48 or any(int(index) >= radiomics_count for index in selected_indices):
            raise RuntimeError("Frozen merged-pool selection must contain exactly 48 radiomic features and no clinical feature.")

    calibration_method = get_calibration_method()
    calibration_cv = get_calibration_cv_splits(y_train)
    if calibration_cv >= 2:
        calibrated_model = build_calibrated_classifier(
            estimator=xgb_binary_classifier(seed),
            method=calibration_method,
            cv=calibration_cv,
        )
        calibrated_model.fit(x_train_selected, y_train)
        probabilities = np.asarray(calibrated_model.predict_proba(x_eval_selected), dtype=np.float64)[:, 1]
        calibration_applied = True
    else:
        fallback_model = xgb_binary_classifier(seed)
        fallback_model.fit(x_train_selected, y_train)
        probabilities = np.asarray(fallback_model.predict_proba(x_eval_selected), dtype=np.float64)[:, 1]
        calibration_applied = False

    if not capture_importance:
        return probabilities, {}

    importance_model = xgb_binary_classifier(seed)
    importance_model.fit(x_train_selected, y_train)
    importance_payload = {
        "rfe_pool": rfe_pool,
        "calibration_method": calibration_method,
        "calibration_cv": int(calibration_cv),
        "calibration_applied": bool(calibration_applied),
        "selected_feature_indices": [int(index) for index in selected_indices.tolist()],
        "selected_feature_names": [str(feature_names[int(index)]) for index in selected_indices.tolist()],
        "xgboost_feature_importances": [
            {
                "feature_name": str(feature_names[int(selected_indices[idx])]),
                "importance": float(value),
            }
            for idx, value in sorted(
                enumerate(np.asarray(importance_model.feature_importances_, dtype=np.float64)),
                key=lambda item: item[1],
                reverse=True,
            )
        ],
    }
    return probabilities, importance_payload


def fit_balanced_bagging(
    x_train: Array2D,
    y_train: Array1D,
    x_eval: Array2D,
    seed: int,
    feature_names: Sequence[str],
    capture_importance: bool = True,
) -> Tuple[Array1D, Dict[str, object]]:
    model = balanced_bagging_xgb(seed)
    model.fit(x_train, y_train)
    probabilities = np.asarray(model.predict_proba(x_eval), dtype=np.float64)[:, 1]

    if not capture_importance:
        return probabilities, {}

    estimators = getattr(model, "estimators_", [])
    if not estimators:
        return probabilities, {"xgboost_feature_importances": []}

    all_importances: List[np.ndarray] = []
    for estimator in estimators:
        inner_estimator = estimator
        if hasattr(estimator, "feature_importances_"):
            inner_estimator = estimator
        elif hasattr(estimator, "named_steps") and "classifier" in estimator.named_steps:
            inner_estimator = estimator.named_steps["classifier"]
        if hasattr(inner_estimator, "feature_importances_"):
            all_importances.append(np.asarray(inner_estimator.feature_importances_, dtype=np.float64))

    if not all_importances:
        return probabilities, {"xgboost_feature_importances": []}

    mean_importance = np.mean(np.stack(all_importances, axis=0), axis=0)
    importance_payload = {
        "xgboost_feature_importances": [
            {"feature_name": str(feature_names[idx]), "importance": float(value)}
            for idx, value in sorted(enumerate(mean_importance), key=lambda item: item[1], reverse=True)
        ]
    }
    return probabilities, importance_payload


def fit_stacking(
    x_train: Array2D,
    y_train: Array1D,
    x_eval: Array2D,
    seed: int,
) -> Array1D:
    model = build_stacking_classifier(seed)
    model.fit(x_train, y_train)
    return np.asarray(model.predict_proba(x_eval), dtype=np.float64)[:, 1]


def fit_model_scores(
    model_name: str,
    x_train: Array2D,
    y_train: Array1D,
    x_eval: Array2D,
    seed: int,
    feature_names: Sequence[str],
    capture_importance: bool = True,
) -> Tuple[Array1D, Optional[Dict[str, object]]]:
    if model_name in {"anomaly_isolation_forest", "anomaly_oneclass_svm"}:
        x_train_neg = x_train[y_train == 0]
        return fit_anomaly_model(model_name, x_train_neg, x_eval, seed), None
    if model_name == "rfe_xgboost":
        scores, payload = fit_rfe_xgboost(
            x_train,
            y_train,
            x_eval,
            seed,
            feature_names,
            capture_importance=capture_importance,
        )
        return scores, payload or None
    if model_name == "rfe_xgboost_calibrated":
        scores, payload = fit_rfe_xgboost_calibrated(
            x_train,
            y_train,
            x_eval,
            seed,
            feature_names,
            capture_importance=capture_importance,
        )
        return scores, payload or None
    if model_name == "balanced_bagging_xgboost":
        scores, payload = fit_balanced_bagging(
            x_train,
            y_train,
            x_eval,
            seed,
            feature_names,
            capture_importance=capture_importance,
        )
        return scores, payload or None
    if model_name == "stacking":
        return fit_stacking(x_train, y_train, x_eval, seed), None
    raise ValueError(f"Unsupported model name {model_name!r}.")


def evaluate_scheme_cv(
    model_name: str,
    x_radiomics: Array2D,
    x_combined: Array2D,
    y: Array1D,
    train_indices: Array1D,
    seed: int,
    feature_names: Sequence[str],
    logger,
) -> Tuple[List[float], List[float], List[float], Optional[Dict[str, object]]]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    x_train_combined = x_combined[train_indices]
    x_train_radiomics = x_radiomics[train_indices]
    y_train_dev = y[train_indices]

    auc_values: List[float] = []
    pr_auc_values: List[float] = []
    threshold_values: List[float] = []
    last_importance_payload: Optional[Dict[str, object]] = None

    for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(splitter.split(x_train_combined, y_train_dev)):
        y_fold_train = y_train_dev[fold_train_idx]
        y_fold_val = y_train_dev[fold_val_idx]

        threshold, _ = search_threshold_from_inner_cv(
            model_name=model_name,
            x_train=x_train_combined[fold_train_idx],
            y_train=y_fold_train,
            seed=seed + fold_idx,
            feature_names=feature_names,
        )
        scores, fold_importance_payload = fit_model_scores(
            model_name=model_name,
            x_train=x_train_combined[fold_train_idx],
            y_train=y_fold_train,
            x_eval=x_train_combined[fold_val_idx],
            seed=seed + fold_idx,
            feature_names=feature_names,
        )
        if fold_importance_payload is not None:
            last_importance_payload = fold_importance_payload

        metrics = compute_metrics(y_fold_val, scores, threshold=threshold)
        auc_values.append(metrics["auc"])
        pr_auc_values.append(metrics["pr_auc"])
        threshold_values.append(metrics["threshold"])
        logger.info(
            "Binary scheme CV | model=%s | fold=%d | auc=%.5f | pr_auc=%.5f | threshold=%.5f",
            model_name,
            fold_idx,
            metrics["auc"],
            metrics["pr_auc"],
            metrics["threshold"],
        )

    return auc_values, pr_auc_values, threshold_values, last_importance_payload


def evaluate_scheme_fixed(
    model_name: str,
    x_combined_train: Array2D,
    x_combined_test: Array2D,
    y_train: Array1D,
    y_test: Array1D,
    seed: int,
    feature_names: Sequence[str],
) -> Tuple[Dict[str, float], Optional[Dict[str, object]], Array1D, float]:
    importance_payload: Optional[Dict[str, object]] = None
    threshold, _ = search_threshold_from_inner_cv(
        model_name=model_name,
        x_train=x_combined_train,
        y_train=y_train,
        seed=seed,
        feature_names=feature_names,
    )
    scores, importance_payload = fit_model_scores(
        model_name=model_name,
        x_train=x_combined_train,
        y_train=y_train,
        x_eval=x_combined_test,
        seed=seed,
        feature_names=feature_names,
    )

    metrics = compute_metrics(y_test, scores, threshold=threshold)
    return metrics, importance_payload, scores, threshold


def write_results_summary(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "Model",
        "CV_Mean_AUC",
        "CV_Std_AUC",
        "CV_Mean_PR_AUC",
        "Fixed_Test_AUC",
        "Fixed_Test_PR_AUC",
        "Recall_20ins",
        "Threshold",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Model": row["Model"],
                    "CV_Mean_AUC": f"{float(row['CV_Mean_AUC']):.5f}",
                    "CV_Std_AUC": f"{float(row['CV_Std_AUC']):.5f}",
                    "CV_Mean_PR_AUC": f"{float(row['CV_Mean_PR_AUC']):.5f}",
                    "Fixed_Test_AUC": f"{float(row['Fixed_Test_AUC']):.5f}",
                    "Fixed_Test_PR_AUC": f"{float(row['Fixed_Test_PR_AUC']):.5f}",
                    "Recall_20ins": f"{float(row['Recall_20ins']):.5f}",
                    "Threshold": f"{float(row['Threshold']):.5f}",
                }
            )


def main() -> None:
    assert_frozen_runtime()
    results_dir = Path(
        os.environ.get("LUAD_EGFR20INS_RESULTS_DIR", str(PROJECT_ROOT / "results" / "egfr20ins_binary_strategies"))
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(
        "egfr20ins_binary_strategies",
        results_dir / "egfr20ins_binary_strategies.log",
        enable_file_logging=True,
        enable_stream_logging=True,
    )

    bundle = load_inputs()
    x_radiomics, x_combined, feature_names = prepare_feature_blocks(bundle)
    y = np.asarray(bundle.y, dtype=np.int64)
    fixed_test_indices = np.asarray(bundle.fixed_test_indices, dtype=np.int64)

    all_indices = np.arange(y.shape[0], dtype=np.int64)
    fixed_mask = np.zeros(y.shape[0], dtype=bool)
    fixed_mask[fixed_test_indices] = True
    train_indices = all_indices[~fixed_mask]

    logger.info(
        "Starting binary EGFR20ins strategies | samples=%d | positives=%d | negatives=%d | fixed_test=%d | threshold_metric=%s | threshold_min_recall=%.2f | threshold_cv=%d | calibration_method=%s | calibration_cv=%d",
        int(y.shape[0]),
        int(y.sum()),
        int((y == 0).sum()),
        int(fixed_test_indices.size),
        get_threshold_search_metric(),
        get_threshold_min_recall(),
        int(os.environ.get("LUAD_EGFR20INS_THRESHOLD_CV", "3")),
        get_calibration_method(),
        int(os.environ.get("LUAD_EGFR20INS_CALIBRATION_CV", "3")),
    )

    model_names = [
        "anomaly_isolation_forest",
        "anomaly_oneclass_svm",
        "rfe_xgboost",
        "rfe_xgboost_calibrated",
        "balanced_bagging_xgboost",
        "stacking",
    ]
    if os.environ.get("LUAD_EGFR20INS_MODELS", "").strip():
        model_names = parse_model_list(os.environ["LUAD_EGFR20INS_MODELS"])

    summary_rows: List[Dict[str, object]] = []
    detail_payload: Dict[str, object] = {
        "data_mode": os.environ.get("LUAD_EGFR20INS_DATA_MODE", "project"),
        "threshold_metric": get_threshold_search_metric(),
        "threshold_min_recall": get_threshold_min_recall(),
        "threshold_cv_splits": int(os.environ.get("LUAD_EGFR20INS_THRESHOLD_CV", "3")),
        "calibration_method": get_calibration_method(),
        "calibration_cv_splits": int(os.environ.get("LUAD_EGFR20INS_CALIBRATION_CV", "3")),
        "sample_count": int(y.shape[0]),
        "positive_count": int(y.sum()),
        "negative_count": int((y == 0).sum()),
        "fixed_test_count": int(fixed_test_indices.size),
        "results": {},
    }

    x_train = x_combined[train_indices]
    y_train = y[train_indices]
    x_fixed_test = x_combined[fixed_test_indices]
    y_fixed_test = y[fixed_test_indices]

    for model_name in model_names:
        cv_auc_values, cv_pr_auc_values, cv_threshold_values, last_importance_payload = evaluate_scheme_cv(
            model_name=model_name,
            x_radiomics=x_radiomics,
            x_combined=x_combined,
            y=y,
            train_indices=train_indices,
            seed=int(os.environ.get("LUAD_EGFR20INS_SEED", "42")),
            feature_names=feature_names,
            logger=logger,
        )
        fixed_metrics, fixed_importance_payload, fixed_scores, fixed_threshold = evaluate_scheme_fixed(
            model_name=model_name,
            x_combined_train=x_train,
            x_combined_test=x_fixed_test,
            y_train=y_train,
            y_test=y_fixed_test,
            seed=int(os.environ.get("LUAD_EGFR20INS_SEED", "42")) + 200,
            feature_names=feature_names,
        )

        row = {
            "Model": model_name,
            "CV_Mean_AUC": float(np.mean(cv_auc_values)),
            "CV_Std_AUC": float(np.std(cv_auc_values, ddof=1)) if len(cv_auc_values) > 1 else 0.0,
            "CV_Mean_PR_AUC": float(np.mean(cv_pr_auc_values)),
            "Fixed_Test_AUC": float(fixed_metrics["auc"]),
            "Fixed_Test_PR_AUC": float(fixed_metrics["pr_auc"]),
            "Recall_20ins": float(fixed_metrics["recall_20ins"]),
            "Threshold": float(fixed_threshold),
        }
        summary_rows.append(row)

        model_detail = {
            "cv_auc_values": [float(value) for value in cv_auc_values],
            "cv_pr_auc_values": [float(value) for value in cv_pr_auc_values],
            "cv_threshold_values": [float(value) for value in cv_threshold_values],
            "fixed_test_auc": float(fixed_metrics["auc"]),
            "fixed_test_pr_auc": float(fixed_metrics["pr_auc"]),
            "fixed_test_recall_20ins": float(fixed_metrics["recall_20ins"]),
            "fixed_threshold": float(fixed_threshold),
            "fixed_test_scores": fixed_scores.tolist(),
            "calibration_method": get_calibration_method(),
            "calibration_cv": int(os.environ.get("LUAD_EGFR20INS_CALIBRATION_CV", "3")),
            "threshold_metric": get_threshold_search_metric(),
            "threshold_min_recall": get_threshold_min_recall(),
        }
        chosen_importance = fixed_importance_payload or last_importance_payload
        if float(fixed_metrics["auc"]) > 0.70 and chosen_importance is not None:
            model_detail["feature_importance"] = chosen_importance
            save_json(chosen_importance, results_dir / f"{model_name}_feature_importance.json")
        detail_payload["results"][model_name] = model_detail
        logger.info(
            "Binary scheme done | model=%s | cv_mean_auc=%.5f | cv_mean_pr_auc=%.5f | fixed_test_auc=%.5f | fixed_test_pr_auc=%.5f | recall_20ins=%.5f | threshold=%.5f",
            model_name,
            row["CV_Mean_AUC"],
            row["CV_Mean_PR_AUC"],
            row["Fixed_Test_AUC"],
            row["Fixed_Test_PR_AUC"],
            row["Recall_20ins"],
            row["Threshold"],
        )

    summary_rows = sorted(summary_rows, key=lambda item: float(item["Fixed_Test_AUC"]), reverse=True)
    write_results_summary(results_dir / "results_summary.csv", summary_rows)
    save_json(detail_payload, results_dir / "results_detail.json")


def parse_model_list(raw_value: str) -> List[str]:
    parts = [part.strip() for part in str(raw_value).split(",") if part.strip()]
    if not parts:
        raise ValueError("LUAD_EGFR20INS_MODELS must be a non-empty comma-separated list.")
    return parts


if __name__ == "__main__":
    main()
