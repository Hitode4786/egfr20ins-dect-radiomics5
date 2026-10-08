from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


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
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, BaseEstimator):
            return {"sklearn_estimator": obj.__class__.__name__}
        return super().default(obj)


def find_first_non_serializable_path(obj: Any, path: str = "root") -> Optional[Tuple[str, str]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_path = f"{path}.{key}"
            issue = find_first_non_serializable_path(value, child_path)
            if issue is not None:
                return issue
        return None
    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            child_path = f"{path}[{idx}]"
            issue = find_first_non_serializable_path(value, child_path)
            if issue is not None:
                return issue
        return None
    try:
        json.dumps(obj, ensure_ascii=False, cls=NumpyEncoder)
    except TypeError:
        return path, type(obj).__name__
    return None


def build_logger(
    name: str,
    log_path: Path,
    file_mode: str = "w",
    enable_file_logging: bool = True,
    enable_stream_logging: bool = True,
) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if enable_file_logging:
        file_handler = logging.FileHandler(log_path, mode=file_mode, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if enable_stream_logging:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def save_json(data: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = json.dumps(data, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    except TypeError as exc:
        issue = find_first_non_serializable_path(data)
        if issue is not None:
            issue_path, issue_type = issue
            raise TypeError(
                f"{exc}. First non-serializable object at {issue_path} (type={issue_type})."
            ) from exc
        raise
    path.write_text(payload, encoding="utf-8")


def load_labels(label_csv: str) -> pd.DataFrame:
    labels_df = pd.read_csv(label_csv)
    labels_df = labels_df.copy()
    labels_df["patient_id"] = labels_df["patient_id"].astype(str)
    labels_df["patient_num"] = labels_df["patient_id"].str.extract(r"(\d+)$").astype(int)
    labels_df["egfr_status"] = pd.to_numeric(labels_df["egfr_status"], errors="raise").astype(int)
    labels_df["metastasis"] = pd.to_numeric(labels_df["metastasis"], errors="raise").astype(int)
    labels_df = labels_df.sort_values(["patient_num", "patient_id"]).reset_index(drop=True)
    return labels_df


def split_patient_ids(labels_df: pd.DataFrame) -> Dict[str, List[str]]:
    return {
        "dev": labels_df.loc[labels_df["patient_num"] <= 187, "patient_id"].tolist(),
        "fixed_train": labels_df.loc[labels_df["patient_num"] <= 156, "patient_id"].tolist(),
        "fixed_val": labels_df.loc[
            (labels_df["patient_num"] >= 157) & (labels_df["patient_num"] <= 187), "patient_id"
        ].tolist(),
        "fixed_test": labels_df.loc[labels_df["patient_num"] >= 188, "patient_id"].tolist(),
    }


def load_feature_metadata(feature_root: Path, labels_df: pd.DataFrame) -> Dict[str, object]:
    metadata_path = feature_root / "feature_bundle_metadata.json"
    if metadata_path.is_file():
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    selected_features_path = feature_root / "selected_features.json"
    summary_path = feature_root / "feature_summary.json"
    if not selected_features_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            "Missing feature metadata. Expected feature_bundle_metadata.json or both "
            "selected_features.json and feature_summary.json."
        )

    selected_features_payload = json.loads(selected_features_path.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    split_ids = split_patient_ids(labels_df)
    return {
        "selector_method": summary_payload.get("selector_method"),
        "selected_features": selected_features_payload.get("selected_features", []),
        "full_feature_names": selected_features_payload.get("full_feature_names", []),
        "train_patient_ids": split_ids["fixed_train"],
        "val_patient_ids": labels_df.loc[
            (labels_df["patient_num"] >= 157) & (labels_df["patient_num"] <= 187), "patient_id"
        ].tolist(),
        "test_patient_ids": split_ids["fixed_test"],
        "train_labels": labels_df.loc[labels_df["patient_num"] <= 156, "egfr_status"].astype(float).tolist(),
        "val_labels": labels_df.loc[
            (labels_df["patient_num"] >= 157) & (labels_df["patient_num"] <= 187), "egfr_status"
        ].astype(float).tolist(),
        "test_labels": labels_df.loc[labels_df["patient_num"] >= 188, "egfr_status"].astype(float).tolist(),
    }


def load_full_radiomics_by_patient(
    feature_dir: str,
    labels_df: pd.DataFrame,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    feature_root = Path(feature_dir)
    bundle = load_feature_metadata(feature_root, labels_df)
    full_paths = {
        "train_patient_ids": feature_root / "train_radiomics_full.npy",
        "val_patient_ids": feature_root / "val_radiomics_full.npy",
        "test_patient_ids": feature_root / "test_radiomics_full.npy",
    }
    missing_paths = [str(path) for path in full_paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing full radiomics caches required for per-split rebuild: "
            f"{missing_paths}. Rerun scripts/extract_radiomics_lasso.py to regenerate the feature directory."
        )

    radiomics_by_patient: Dict[str, np.ndarray] = {}
    for split_key, array_path in full_paths.items():
        feature_array = np.load(array_path, allow_pickle=False)
        patient_ids = list(bundle[split_key])
        if len(patient_ids) != feature_array.shape[0]:
            raise ValueError(f"Full radiomics length mismatch for {split_key}.")
        for idx, patient_id in enumerate(patient_ids):
            radiomics_by_patient[str(patient_id)] = np.asarray(feature_array[idx], dtype=np.float32)
    return radiomics_by_patient, bundle
