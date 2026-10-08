"""
DECT radiomics feature extraction and LASSO selection pipeline.

This script:
1. Loads per-patient raw DECT image volumes and manual binary masks from NIfTI files.
2. Extracts 107 PyRadiomics features for each of 7 sequences (749 features total).
3. Fits a training-only StandardScaler + LassoCV feature selector.
4. Saves strictly 107 selected radiomics features for train/val/test.
5. Builds clinical features from labels.csv and saves train/val/test arrays.

Notes
-----
- Spatial spacing is taken from the raw image NIfTI headers and copied into the
  temporary SimpleITK images passed to PyRadiomics.
- The manual masks are expected to be binary and aligned to the raw image grid.
- The task specification contains a contradiction for clinical dimensions:
  it asks for location one-hot -> PCA(3), but also asks for train_clinical.npy shape
  [156, 5]. To keep the default outputs aligned with the requested array shapes,
  location PCA defaults to 1 component. Pass location_pca_components=3 if you want
  the expanded variant, which will produce 7 clinical columns.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from nifti_mask_utils import load_binary_mask_aligned_to_image
from radiomics import featureextractor
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm

from path_defaults import (
    DEFAULT_RADIOMICS_INPUT_DIR,
    DEFAULT_RADIOMICS_OUTPUT_DIR,
    LABEL_CSV as DEFAULT_LABEL_CSV,
    MANUAL_MASK_DIR,
)


Array2D = np.ndarray
Array1D = np.ndarray
Array3D = np.ndarray
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEQUENCE_KEYS: Tuple[str, ...] = (
    "A80KV",
    "A140KV",
    "AIC",
    "C",
    "V80KV",
    "V140KV",
    "VIC",
)

SHAPE_FEATURES: Tuple[str, ...] = (
    "MeshVolume",
    "VoxelVolume",
    "SurfaceArea",
    "SurfaceVolumeRatio",
    "Sphericity",
    "Compactness1",
    "Compactness2",
    "SphericalDisproportion",
    "Maximum3DDiameter",
    "Maximum2DDiameterSlice",
    "Maximum2DDiameterColumn",
    "Maximum2DDiameterRow",
    "MajorAxisLength",
    "MinorAxisLength",
)

FIRSTORDER_FEATURES: Tuple[str, ...] = (
    "Energy",
    "TotalEnergy",
    "Entropy",
    "Minimum",
    "Maximum",
    "Mean",
    "Median",
    "Range",
    "MeanAbsoluteDeviation",
    "RootMeanSquared",
    "StandardDeviation",
    "Skewness",
    "Kurtosis",
    "Variance",
    "Uniformity",
    "InterquartileRange",
    "10Percentile",
    "90Percentile",
)

GLCM_FEATURES: Tuple[str, ...] = (
    "Autocorrelation",
    "JointAverage",
    "ClusterProminence",
    "ClusterShade",
    "ClusterTendency",
    "Contrast",
    "Correlation",
    "DifferenceAverage",
    "DifferenceEntropy",
    "DifferenceVariance",
    "JointEnergy",
    "JointEntropy",
    "Imc1",
    "Imc2",
    "Idm",
    "MCC",
    "Idmn",
    "Id",
    "Idn",
    "InverseVariance",
    "MaximumProbability",
    "SumAverage",
    "SumEntropy",
    "SumSquares",
)

GLRLM_FEATURES: Tuple[str, ...] = (
    "ShortRunEmphasis",
    "LongRunEmphasis",
    "GrayLevelNonUniformity",
    "GrayLevelNonUniformityNormalized",
    "RunLengthNonUniformity",
    "RunLengthNonUniformityNormalized",
    "RunPercentage",
    "GrayLevelVariance",
    "RunVariance",
    "RunEntropy",
    "LowGrayLevelRunEmphasis",
    "HighGrayLevelRunEmphasis",
    "ShortRunLowGrayLevelEmphasis",
    "ShortRunHighGrayLevelEmphasis",
    "LongRunLowGrayLevelEmphasis",
    "LongRunHighGrayLevelEmphasis",
)

GLSZM_FEATURES: Tuple[str, ...] = (
    "SmallAreaEmphasis",
    "LargeAreaEmphasis",
    "GrayLevelNonUniformity",
    "GrayLevelNonUniformityNormalized",
    "SizeZoneNonUniformity",
    "SizeZoneNonUniformityNormalized",
    "ZonePercentage",
    "GrayLevelVariance",
    "ZoneVariance",
    "ZoneEntropy",
    "LowGrayLevelZoneEmphasis",
    "HighGrayLevelZoneEmphasis",
    "SmallAreaLowGrayLevelEmphasis",
    "SmallAreaHighGrayLevelEmphasis",
    "LargeAreaLowGrayLevelEmphasis",
    "LargeAreaHighGrayLevelEmphasis",
)

GLDM_FEATURES: Tuple[str, ...] = (
    "SmallDependenceEmphasis",
    "LargeDependenceEmphasis",
    "GrayLevelNonUniformity",
    "DependenceNonUniformity",
    "DependenceNonUniformityNormalized",
    "GrayLevelVariance",
    "DependenceVariance",
    "DependenceEntropy",
    "LowGrayLevelEmphasis",
    "HighGrayLevelEmphasis",
    "SmallDependenceLowGrayLevelEmphasis",
    "SmallDependenceHighGrayLevelEmphasis",
    "LargeDependenceLowGrayLevelEmphasis",
    "LargeDependenceHighGrayLevelEmphasis",
)

NGTDM_FEATURES: Tuple[str, ...] = (
    "Coarseness",
    "Contrast",
    "Busyness",
    "Complexity",
    "Strength",
)

FEATURES_BY_CLASS: Dict[str, Tuple[str, ...]] = {
    "shape": SHAPE_FEATURES,
    "firstorder": FIRSTORDER_FEATURES,
    "glcm": GLCM_FEATURES,
    "glrlm": GLRLM_FEATURES,
    "glszm": GLSZM_FEATURES,
    "gldm": GLDM_FEATURES,
    "ngtdm": NGTDM_FEATURES,
}

EXPECTED_FEATURES_PER_SEQUENCE = 107
EXPECTED_TOTAL_RADIOMICS_DIM = EXPECTED_FEATURES_PER_SEQUENCE * len(SEQUENCE_KEYS)
SELECTED_RADIOMICS_DIM = 107

RADIOMICS_SETTINGS = {
    "binWidth": 25,
    "resampledPixelSpacing": None,
    "interpolator": "sitkBSpline",
    "normalize": False,
    "normalizeScale": 1,
    "resegmentRange": None,
    "label": 1,
}

REQUIRED_LABEL_COLUMNS = (
    "patient_id",
    "sex",
    "age",
    "location",
    "egfr_status",
    "smoking_history",
    "metastasis",
)


@dataclass(frozen=True)
class FeatureDescriptor:
    sequence: str
    feature_class: str
    feature_name: str
    full_name: str


def build_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("dect_radiomics")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_path = os.path.join(output_dir, "feature_extraction_log.txt")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def extract_patient_num(patient_id: str) -> Optional[int]:
    match = re.search(r"(\d+)$", str(patient_id))
    if match is None:
        return None
    return int(match.group(1))


def build_feature_descriptors() -> List[FeatureDescriptor]:
    descriptors: List[FeatureDescriptor] = []
    for sequence in SEQUENCE_KEYS:
        for feature_class, feature_names in FEATURES_BY_CLASS.items():
            for feature_name in feature_names:
                descriptors.append(
                    FeatureDescriptor(
                        sequence=sequence,
                        feature_class=feature_class,
                        feature_name=feature_name,
                        full_name=f"{sequence}__original_{feature_class}_{feature_name}",
                    )
                )

    if len(descriptors) != EXPECTED_TOTAL_RADIOMICS_DIM:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_RADIOMICS_DIM} radiomics descriptors, got {len(descriptors)}."
        )
    return descriptors


def load_and_split_labels(label_csv: str, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels_df = pd.read_csv(label_csv)

    missing_columns = sorted(set(REQUIRED_LABEL_COLUMNS) - set(labels_df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns in labels.csv: {missing_columns}")

    labels_df = labels_df.copy()
    labels_df["patient_id"] = labels_df["patient_id"].astype(str)
    labels_df["patient_num"] = labels_df["patient_id"].map(extract_patient_num)

    if labels_df["patient_num"].isna().any():
        bad_ids = labels_df.loc[labels_df["patient_num"].isna(), "patient_id"].tolist()
        raise ValueError(f"Failed to parse patient numbers from patient_id: {bad_ids}")

    numeric_columns = ("sex", "age", "location", "egfr_status", "smoking_history", "metastasis")
    for column in numeric_columns:
        labels_df[column] = pd.to_numeric(labels_df[column], errors="raise")

    labels_df = labels_df.sort_values(["patient_num", "patient_id"]).reset_index(drop=True)

    train_df = labels_df[(labels_df["patient_num"] >= 1) & (labels_df["patient_num"] <= 156)].reset_index(drop=True)
    val_df = labels_df[(labels_df["patient_num"] >= 157) & (labels_df["patient_num"] <= 187)].reset_index(drop=True)
    test_df = labels_df[(labels_df["patient_num"] >= 188) & (labels_df["patient_num"] <= 248)].reset_index(drop=True)

    logger.info("Loaded %d label rows.", len(labels_df))
    logger.info(
        "Split sizes | train=%d | val=%d | test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    if len(train_df) != 156 or len(val_df) != 31 or len(test_df) != 61:
        logger.warning(
            "Split sizes differ from the requested counts [156, 31, 61]. "
            "Current sizes are [%d, %d, %d].",
            len(train_df),
            len(val_df),
            len(test_df),
        )

    return train_df, val_df, test_df


def make_radiomics_extractor() -> object:
    extractor_cls = getattr(featureextractor, "RadiomicsFeatureExtractor", None)
    if extractor_cls is None:
        extractor_cls = getattr(featureextractor, "RadiomicsFeaturesExtractor")

    extractor = extractor_cls(**RADIOMICS_SETTINGS)

    if hasattr(extractor, "addProvenance"):
        extractor.addProvenance(False)
    if hasattr(extractor, "disableAllFeatures"):
        extractor.disableAllFeatures()
    if hasattr(extractor, "disableAllImageTypes"):
        extractor.disableAllImageTypes()
    if hasattr(extractor, "enableImageTypeByName"):
        extractor.enableImageTypeByName("Original")

    enable_features_by_name = getattr(extractor, "enableFeaturesByName", None)
    if enable_features_by_name is None:
        raise AttributeError(
            "PyRadiomics extractor does not expose enableFeaturesByName()."
        )
    enable_features_by_name(
        **{feature_class: list(feature_names) for feature_class, feature_names in FEATURES_BY_CLASS.items()}
    )

    return extractor


def build_zero_sequence_features() -> Array1D:
    return np.zeros(EXPECTED_FEATURES_PER_SEQUENCE, dtype=np.float32)


def build_zero_patient_features() -> Array1D:
    return np.zeros(EXPECTED_TOTAL_RADIOMICS_DIM, dtype=np.float32)


def compute_mask_bounding_box(mask_dhw: Array3D) -> Tuple[Tuple[int, int], ...]:
    coordinates = np.where(mask_dhw > 0)
    if coordinates[0].size == 0:
        raise ValueError("Cannot compute bounding box from an empty mask.")

    bbox: List[Tuple[int, int]] = []
    for axis_coords in coordinates:
        bbox.append((int(axis_coords.min()), int(axis_coords.max())))
    return tuple(bbox)


def expand_bounding_box(
    bbox: Tuple[Tuple[int, int], ...],
    shape: Tuple[int, int, int],
    padding: int = 1,
) -> Tuple[Tuple[int, int], ...]:
    expanded: List[Tuple[int, int]] = []
    for axis, (lower, upper) in enumerate(bbox):
        start = max(0, lower - padding)
        end = min(shape[axis] - 1, upper + padding)
        expanded.append((int(start), int(end)))
    return tuple(expanded)


def crop_volume_to_bounds(volume_dhw: Array3D, bounds: Tuple[Tuple[int, int], ...]) -> Array3D:
    slices = tuple(slice(start, end + 1) for start, end in bounds)
    return np.asarray(volume_dhw[slices])


def load_image_and_mask_nifti(
    image_path: str,
    mask_path: str,
) -> Tuple[Array3D, Array3D, Tuple[float, float, float]]:
    image_nifti = nib.load(image_path)

    image_hwd = image_nifti.get_fdata(dtype=np.float32)
    mask_alignment = load_binary_mask_aligned_to_image(image_nifti, mask_path)
    mask_hwd = mask_alignment.mask_hwd
    if image_hwd.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {image_hwd.shape}")
    if image_hwd.shape != mask_hwd.shape:
        raise ValueError(f"Image/mask shape mismatch: image={image_hwd.shape} mask={mask_hwd.shape}")

    image_dhw = np.transpose(np.nan_to_num(image_hwd, nan=0.0, posinf=0.0, neginf=0.0), (2, 0, 1))
    mask_dhw = np.transpose(mask_hwd > 0.5, (2, 0, 1)).astype(np.uint8, copy=False)
    mask_voxels = int(np.count_nonzero(mask_dhw))
    if mask_voxels == 0:
        raise ValueError("Manual mask is empty.")

    bbox = compute_mask_bounding_box(mask_dhw)
    crop_bounds = expand_bounding_box(bbox, image_dhw.shape, padding=1)
    cropped_image = crop_volume_to_bounds(image_dhw, crop_bounds).astype(np.float32, copy=False)
    cropped_mask = crop_volume_to_bounds(mask_dhw, crop_bounds).astype(np.uint8, copy=False)

    spacing_xyz = tuple(float(v) for v in image_nifti.header.get_zooms()[:3])
    if len(spacing_xyz) != 3 or any(v <= 0 for v in spacing_xyz):
        spacing_xyz = (1.0, 1.0, 1.0)
    return cropped_image, cropped_mask, spacing_xyz


def convert_volume_to_sitk(
    volume_dhw: Array3D,
    mask_dhw: Array3D,
    spacing_xyz: Tuple[float, float, float],
) -> Tuple[sitk.Image, sitk.Image, int]:
    volume_dhw = np.asarray(volume_dhw, dtype=np.float32)
    if volume_dhw.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume_dhw.shape}")

    volume_dhw = np.nan_to_num(volume_dhw, nan=0.0, posinf=0.0, neginf=0.0)
    mask_dhw = (np.asarray(mask_dhw) > 0.5).astype(np.uint8, copy=False)
    if mask_dhw.shape != volume_dhw.shape:
        raise ValueError(f"Image/mask shape mismatch: image={volume_dhw.shape}, mask={mask_dhw.shape}")

    mask_voxels = int(np.count_nonzero(mask_dhw))

    if mask_voxels == 0:
        raise ValueError("Input mask is all zeros.")

    if mask_voxels == mask_dhw.size:
        mask_dhw = np.zeros_like(mask_dhw, dtype=np.uint8)
        if min(mask_dhw.shape) <= 2:
            raise ValueError(f"ROI shape too small to construct a valid mask: {mask_dhw.shape}")
        mask_dhw[1:-1, 1:-1, 1:-1] = 1
        mask_voxels = int(np.count_nonzero(mask_dhw))

    image = sitk.GetImageFromArray(volume_dhw.astype(np.float32, copy=False))
    mask = sitk.GetImageFromArray(mask_dhw)
    image.SetSpacing(tuple(float(v) for v in spacing_xyz))
    mask.CopyInformation(image)
    return image, mask, mask_voxels


def extract_sequence_features(
    volume_dhw: Array3D,
    mask_dhw: Array3D,
    spacing_xyz: Tuple[float, float, float],
    extractor: object,
    sequence_key: str,
    logger: logging.Logger,
) -> Tuple[Array1D, Dict[str, object]]:
    try:
        image, mask, mask_voxels = convert_volume_to_sitk(volume_dhw, mask_dhw, spacing_xyz)
        result = extractor.execute(image, mask)
    except Exception as exc:
        return build_zero_sequence_features(), {
            "sequence": sequence_key,
            "status": "failed",
            "reason": str(exc),
            "mask_voxels": 0,
        }

    feature_values: List[float] = []
    for feature_class, feature_names in FEATURES_BY_CLASS.items():
        for feature_name in feature_names:
            result_key = f"original_{feature_class}_{feature_name}"
            value = result.get(result_key, 0.0)
            try:
                numeric_value = float(value)
            except Exception:
                numeric_value = 0.0
            if not np.isfinite(numeric_value):
                numeric_value = 0.0
            feature_values.append(numeric_value)

    if len(feature_values) != EXPECTED_FEATURES_PER_SEQUENCE:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES_PER_SEQUENCE} sequence features, got {len(feature_values)}."
        )

    return np.asarray(feature_values, dtype=np.float32), {
        "sequence": sequence_key,
        "status": "success",
        "reason": "",
        "mask_voxels": mask_voxels,
    }


def extract_patient_radiomics(
    patient_id: str,
    raw_dir: str,
    mask_dir: str,
    extractor: object,
    logger: logging.Logger,
) -> Tuple[Array1D, List[Dict[str, object]]]:
    patient_features: List[Array1D] = []
    sequence_logs: List[Dict[str, object]] = []

    patient_raw_dir = os.path.join(raw_dir, patient_id)
    patient_mask_dir = os.path.join(mask_dir, patient_id)
    for sequence_key in SEQUENCE_KEYS:
        image_path = os.path.join(patient_raw_dir, f"{sequence_key}.nii")
        mask_path = os.path.join(patient_mask_dir, f"{sequence_key}.nii")
        if not os.path.isfile(image_path):
            logger.warning("Missing raw image for %s %s: %s", patient_id, sequence_key, image_path)
            patient_features.append(build_zero_sequence_features())
            sequence_logs.append(
                {
                    "sequence": sequence_key,
                    "status": "failed",
                    "reason": "missing_raw_image",
                    "mask_voxels": 0,
                }
            )
            continue
        if not os.path.isfile(mask_path):
            logger.warning("Missing manual mask for %s %s: %s", patient_id, sequence_key, mask_path)
            patient_features.append(build_zero_sequence_features())
            sequence_logs.append(
                {
                    "sequence": sequence_key,
                    "status": "failed",
                    "reason": "missing_manual_mask",
                    "mask_voxels": 0,
                }
            )
            continue

        try:
            image_array, mask_array, spacing_xyz = load_image_and_mask_nifti(image_path, mask_path)
            sequence_features, sequence_log = extract_sequence_features(
                volume_dhw=image_array,
                mask_dhw=mask_array,
                spacing_xyz=spacing_xyz,
                extractor=extractor,
                sequence_key=sequence_key,
                logger=logger,
            )
        except Exception as exc:
            sequence_features = build_zero_sequence_features()
            sequence_log = {
                "sequence": sequence_key,
                "status": "failed",
                "reason": str(exc),
                "mask_voxels": 0,
            }
        patient_features.append(sequence_features)
        sequence_logs.append(sequence_log)

    concatenated = np.concatenate(patient_features, axis=0).astype(np.float32, copy=False)
    if concatenated.shape[0] != EXPECTED_TOTAL_RADIOMICS_DIM:
        raise ValueError(
            f"Expected patient feature dim {EXPECTED_TOTAL_RADIOMICS_DIM}, got {concatenated.shape[0]}."
        )
    return concatenated, sequence_logs


def extract_split_radiomics(
    split_df: pd.DataFrame,
    split_name: str,
    raw_dir: str,
    mask_dir: str,
    extractor: object,
    logger: logging.Logger,
) -> Tuple[Array2D, Array1D, List[str], List[Dict[str, object]]]:
    if split_df.empty:
        return (
            np.zeros((0, EXPECTED_TOTAL_RADIOMICS_DIM), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            [],
            [],
        )

    features: List[Array1D] = []
    labels: List[float] = []
    patient_ids: List[str] = []
    failure_rows: List[Dict[str, object]] = []

    progress = tqdm(
        split_df.itertuples(index=False),
        total=len(split_df),
        desc=f"Radiomics {split_name}",
    )

    for row_index, row in enumerate(progress, start=1):
        patient_id = str(row.patient_id)
        patient_ids.append(patient_id)
        labels.append(float(row.egfr_status))
        patient_raw_dir = os.path.join(raw_dir, patient_id)
        patient_mask_dir = os.path.join(mask_dir, patient_id)

        if not os.path.isdir(patient_raw_dir):
            logger.warning("Missing raw image directory for %s: %s", patient_id, patient_raw_dir)
            features.append(build_zero_patient_features())
            for sequence_key in SEQUENCE_KEYS:
                failure_rows.append(
                    {
                        "split": split_name,
                        "patient_id": patient_id,
                        "sequence": sequence_key,
                        "status": "failed",
                        "reason": "missing_raw_patient_dir",
                        "mask_voxels": 0,
                    }
                )
            continue
        if not os.path.isdir(patient_mask_dir):
            logger.warning("Missing manual mask directory for %s: %s", patient_id, patient_mask_dir)
            features.append(build_zero_patient_features())
            for sequence_key in SEQUENCE_KEYS:
                failure_rows.append(
                    {
                        "split": split_name,
                        "patient_id": patient_id,
                        "sequence": sequence_key,
                        "status": "failed",
                        "reason": "missing_mask_patient_dir",
                        "mask_voxels": 0,
                    }
                )
            continue

        try:
            patient_feature_vector, patient_logs = extract_patient_radiomics(
                patient_id=patient_id,
                raw_dir=raw_dir,
                mask_dir=mask_dir,
                extractor=extractor,
                logger=logger,
            )
            features.append(patient_feature_vector)
            for log_item in patient_logs:
                log_row = {
                    "split": split_name,
                    "patient_id": patient_id,
                    **log_item,
                }
                if log_item["status"] != "success":
                    failure_rows.append(log_row)
        except Exception as exc:
            logger.error(
                "Patient-level radiomics extraction failed for %s: %s\n%s",
                patient_id,
                exc,
                traceback.format_exc(),
            )
            features.append(build_zero_patient_features())
            for sequence_key in SEQUENCE_KEYS:
                failure_rows.append(
                    {
                        "split": split_name,
                        "patient_id": patient_id,
                        "sequence": sequence_key,
                        "status": "failed",
                        "reason": f"patient_exception: {exc}",
                        "mask_voxels": 0,
                    }
                )

        if row_index % 10 == 0 or row_index == len(split_df):
            logger.info(
                "Radiomics progress | split=%s | processed=%d/%d",
                split_name,
                row_index,
                len(split_df),
            )

    feature_matrix = np.vstack(features).astype(np.float32, copy=False)
    label_array = np.asarray(labels, dtype=np.float32)
    return feature_matrix, label_array, patient_ids, failure_rows


def summarize_selector_coefficients(coef: np.ndarray) -> Tuple[Array1D, Array1D]:
    coef = np.asarray(coef, dtype=np.float64)
    if coef.ndim == 1:
        return coef.copy(), np.abs(coef)
    if coef.ndim != 2:
        raise ValueError(f"Unsupported coefficient shape: {coef.shape}")

    abs_coef = np.abs(coef)
    dominant_class = np.argmax(abs_coef, axis=0)
    feature_indices = np.arange(coef.shape[1])
    signed_scores = coef[dominant_class, feature_indices]
    abs_scores = abs_coef[dominant_class, feature_indices]
    return signed_scores, abs_scores


def resolve_classification_cv_folds(y_train: Array1D, max_folds: int = 5) -> int:
    _, counts = np.unique(np.asarray(y_train), return_counts=True)
    if counts.size < 2:
        return 0
    min_class_count = int(np.min(counts))
    if min_class_count < 2:
        return 0
    return min(max_folds, min_class_count)


def fit_lasso_selector(
    train_scaled: Array2D,
    y_train: Array1D,
) -> Dict[str, object]:
    feature_count = int(train_scaled.shape[1])
    zero_scores = np.zeros(feature_count, dtype=np.float64)
    cv_folds = min(5, int(train_scaled.shape[0]))
    if cv_folds < 2:
        return {
            "model": None,
            "method": "lasso_skipped_insufficient_samples",
            "signed_scores": zero_scores,
            "abs_scores": zero_scores,
            "coef_matrix": None,
            "nonzero_count": 0,
            "details": {
                "cv_folds": cv_folds,
                "reason": "Need at least 2 training samples for LassoCV.",
            },
        }

    lasso = LassoCV(cv=cv_folds, max_iter=20000, random_state=42)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            lasso.fit(train_scaled, y_train)
    except ValueError as exc:
        return {
            "model": None,
            "method": "lasso_skipped_value_error",
            "signed_scores": zero_scores,
            "abs_scores": zero_scores,
            "coef_matrix": None,
            "nonzero_count": 0,
            "details": {
                "cv_folds": cv_folds,
                "reason": str(exc),
            },
        }

    signed_scores, abs_scores = summarize_selector_coefficients(lasso.coef_)
    nonzero_count = int(np.count_nonzero(abs_scores > 1e-12))
    return {
        "model": lasso,
        "method": "lasso_cv",
        "signed_scores": signed_scores,
        "abs_scores": abs_scores,
        "coef_matrix": np.asarray(lasso.coef_, dtype=np.float64),
        "nonzero_count": nonzero_count,
        "details": {
            "cv_folds": cv_folds,
            "lasso_alpha": float(lasso.alpha_),
            "max_iter": int(lasso.max_iter),
        },
    }


def fit_logistic_l1_selector(
    train_scaled: Array2D,
    y_train: Array1D,
) -> Optional[Dict[str, object]]:
    cv_folds = resolve_classification_cv_folds(y_train)
    if cv_folds < 2:
        return None

    logistic = LogisticRegressionCV(
        Cs=np.logspace(-4, 4, 12),
        cv=cv_folds,
        penalty="l1",
        solver="saga",
        scoring="neg_log_loss",
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )
    y_train_int = np.asarray(y_train, dtype=np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        logistic.fit(train_scaled, y_train_int)

    coef_matrix = np.asarray(logistic.coef_, dtype=np.float64)
    signed_scores, abs_scores = summarize_selector_coefficients(coef_matrix)
    nonzero_count = int(np.count_nonzero(abs_scores > 1e-12))
    return {
        "model": logistic,
        "method": "logistic_l1_cv",
        "signed_scores": signed_scores,
        "abs_scores": abs_scores,
        "coef_matrix": coef_matrix,
        "nonzero_count": nonzero_count,
        "details": {
            "cv_folds": cv_folds,
            "best_c": np.asarray(logistic.C_, dtype=np.float64).tolist(),
            "classes": np.asarray(logistic.classes_, dtype=np.int64).tolist(),
            "max_iter": int(logistic.max_iter),
        },
    }


def rank_features_by_anova(
    train_scaled: Array2D,
    y_train: Array1D,
) -> Optional[Dict[str, object]]:
    if np.unique(np.asarray(y_train)).size < 2:
        return None

    y_train_int = np.asarray(y_train, dtype=np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scores, _ = f_classif(train_scaled, y_train_int)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    nonzero_count = int(np.count_nonzero(scores > 1e-12))
    return {
        "model": None,
        "method": "anova_f_classif",
        "signed_scores": scores.astype(np.float64, copy=False),
        "abs_scores": scores.astype(np.float64, copy=False),
        "coef_matrix": None,
        "nonzero_count": nonzero_count,
        "details": {},
    }


def rank_features_by_variance(train_features: Array2D) -> Dict[str, object]:
    scores = np.var(train_features, axis=0, dtype=np.float64)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    nonzero_count = int(np.count_nonzero(scores > 1e-12))
    return {
        "model": None,
        "method": "variance_fallback",
        "signed_scores": scores.astype(np.float64, copy=False),
        "abs_scores": scores.astype(np.float64, copy=False),
        "coef_matrix": None,
        "nonzero_count": nonzero_count,
        "details": {},
    }


def select_top_lasso_features(
    train_features: Array2D,
    val_features: Array2D,
    test_features: Array2D,
    y_train: Array1D,
    feature_descriptors: Sequence[FeatureDescriptor],
    output_dir: str,
    logger: logging.Logger,
) -> Tuple[Array2D, Array2D, Array2D, Dict[str, object]]:
    if train_features.shape[1] != EXPECTED_TOTAL_RADIOMICS_DIM:
        raise ValueError(
            f"Expected full radiomics dim {EXPECTED_TOTAL_RADIOMICS_DIM}, got {train_features.shape[1]}."
        )

    radiomics_scaler = StandardScaler()
    train_scaled = radiomics_scaler.fit_transform(train_features)
    val_scaled = (
        radiomics_scaler.transform(val_features)
        if val_features.shape[0] > 0
        else np.zeros((0, train_features.shape[1]), dtype=np.float32)
    )
    test_scaled = (
        radiomics_scaler.transform(test_features)
        if test_features.shape[0] > 0
        else np.zeros((0, train_features.shape[1]), dtype=np.float32)
    )

    feature_variances = np.var(train_features, axis=0, dtype=np.float64)
    candidate_mask = np.isfinite(feature_variances) & (feature_variances > 1e-12)
    candidate_indices = np.flatnonzero(candidate_mask)
    zero_variance_count = int(train_features.shape[1] - candidate_indices.size)
    if candidate_indices.size == 0:
        candidate_indices = np.arange(train_features.shape[1], dtype=np.int32)
        zero_variance_count = 0
        logger.warning(
            "All radiomics features appear zero-variance on the training split; skipping variance filter."
        )
    else:
        logger.info(
            "Filtered %d zero-variance radiomics features; %d candidate features remain.",
            zero_variance_count,
            candidate_indices.size,
        )

    train_candidates = train_scaled[:, candidate_indices]
    val_candidates = val_scaled[:, candidate_indices]
    test_candidates = test_scaled[:, candidate_indices]

    lasso_result = fit_lasso_selector(train_candidates, y_train)
    selector_result = lasso_result
    if lasso_result["nonzero_count"] == 0:
        logger.warning(
            "LASSO returned zero non-zero features across %d candidate dimensions; "
            "switching to classification-aware sparse selection.",
            candidate_indices.size,
        )
        logistic_result = fit_logistic_l1_selector(train_candidates, y_train)
        if logistic_result is not None and logistic_result["nonzero_count"] > 0:
            selector_result = logistic_result
        else:
            if logistic_result is None:
                logger.warning(
                    "Logistic L1 fallback was skipped because the training labels do not support stratified CV."
                )
            else:
                logger.warning(
                    "Logistic L1 fallback also returned zero non-zero features; switching to ANOVA ranking."
                )
            anova_result = rank_features_by_anova(train_candidates, y_train)
            if anova_result is not None and anova_result["nonzero_count"] > 0:
                selector_result = anova_result
            else:
                logger.warning(
                    "ANOVA ranking was unavailable or uninformative; switching to variance ranking."
                )
                selector_result = rank_features_by_variance(train_features[:, candidate_indices])

    full_signed_scores = np.zeros(train_features.shape[1], dtype=np.float64)
    full_abs_scores = np.zeros(train_features.shape[1], dtype=np.float64)
    full_signed_scores[candidate_indices] = np.asarray(selector_result["signed_scores"], dtype=np.float64)
    full_abs_scores[candidate_indices] = np.asarray(selector_result["abs_scores"], dtype=np.float64)

    ranked_candidate_order = np.argsort(-full_abs_scores[candidate_indices], kind="mergesort")
    selected_indices = candidate_indices[ranked_candidate_order[:SELECTED_RADIOMICS_DIM]]
    if selected_indices.size < SELECTED_RADIOMICS_DIM:
        remaining_indices = np.setdiff1d(
            np.arange(train_features.shape[1], dtype=np.int32),
            selected_indices.astype(np.int32, copy=False),
            assume_unique=False,
        )
        remaining_order = np.argsort(-feature_variances[remaining_indices], kind="mergesort")
        needed = SELECTED_RADIOMICS_DIM - selected_indices.size
        selected_indices = np.concatenate([selected_indices, remaining_indices[remaining_order[:needed]]])

    selected_train = train_scaled[:, selected_indices].astype(np.float32, copy=False)
    selected_val = val_scaled[:, selected_indices].astype(np.float32, copy=False)
    selected_test = test_scaled[:, selected_indices].astype(np.float32, copy=False)

    selected_features_payload: List[Dict[str, object]] = []
    rank_lookup: Dict[int, int] = {}
    candidate_lookup = {int(full_idx): local_idx for local_idx, full_idx in enumerate(candidate_indices.tolist())}
    selector_coef_matrix = selector_result["coef_matrix"]
    lasso_alpha = lasso_result["details"].get("lasso_alpha")
    for rank, feature_index in enumerate(selected_indices, start=1):
        descriptor = feature_descriptors[int(feature_index)]
        rank_lookup[int(feature_index)] = rank
        payload: Dict[str, object] = {
            "rank": rank,
            "feature_index": int(feature_index),
            "sequence": descriptor.sequence,
            "feature_class": descriptor.feature_class,
            "feature_name": descriptor.feature_name,
            "full_name": descriptor.full_name,
            "coefficient": float(full_signed_scores[feature_index]),
            "abs_coefficient": float(full_abs_scores[feature_index]),
        }
        if selector_coef_matrix is not None:
            local_feature_index = candidate_lookup.get(int(feature_index))
            if local_feature_index is not None:
                coef_array = np.asarray(selector_coef_matrix, dtype=np.float64)
                if coef_array.ndim == 2:
                    payload["class_coefficients"] = coef_array[:, local_feature_index].astype(float).tolist()
        selected_features_payload.append(payload)

    selected_features_json = {
        "selected_dim": SELECTED_RADIOMICS_DIM,
        "selector_method": str(selector_result["method"]),
        "candidate_feature_count_after_variance_filter": int(candidate_indices.size),
        "zero_variance_feature_count": zero_variance_count,
        "lasso_alpha": float(lasso_alpha) if lasso_alpha is not None else None,
        "lasso_nonzero_before_topk": int(lasso_result["nonzero_count"]),
        "nonzero_before_topk": int(selector_result["nonzero_count"]),
        "selector_details": selector_result["details"],
        "selected_features": selected_features_payload,
    }
    selected_features_path = os.path.join(output_dir, "selected_features.json")
    with open(selected_features_path, "w", encoding="utf-8") as f:
        json.dump(selected_features_json, f, indent=2, ensure_ascii=False)

    if lasso_alpha is None:
        logger.info(
            "Feature selector fitted | method=%s | lasso_alpha=n/a | "
            "lasso_nonzero=%d | selector_nonzero=%d | selected=%d",
            str(selector_result["method"]),
            int(lasso_result["nonzero_count"]),
            int(selector_result["nonzero_count"]),
            len(selected_indices),
        )
    else:
        logger.info(
            "Feature selector fitted | method=%s | lasso_alpha=%.8f | "
            "lasso_nonzero=%d | selector_nonzero=%d | selected=%d",
            str(selector_result["method"]),
            float(lasso_alpha),
            int(lasso_result["nonzero_count"]),
            int(selector_result["nonzero_count"]),
            len(selected_indices),
        )
    logger.info("Saved selected feature list to %s", selected_features_path)

    metadata = {
        "radiomics_scaler": radiomics_scaler,
        "lasso_model": lasso_result["model"],
        "selector_model": selector_result["model"],
        "selector_method": str(selector_result["method"]),
        "lasso_nonzero_before_topk": int(lasso_result["nonzero_count"]),
        "selector_nonzero_before_topk": int(selector_result["nonzero_count"]),
        "zero_variance_feature_count": zero_variance_count,
        "candidate_feature_count_after_variance_filter": int(candidate_indices.size),
        "selected_indices": selected_indices.astype(np.int32),
        "selected_features_payload": selected_features_payload,
        "rank_lookup": rank_lookup,
        "full_feature_names": [descriptor.full_name for descriptor in feature_descriptors],
    }
    return selected_train, selected_val, selected_test, metadata


def safe_pearson_correlation(feature_column: Array1D, label_array: Array1D) -> float:
    feature_column = np.asarray(feature_column, dtype=np.float64)
    label_array = np.asarray(label_array, dtype=np.float64)

    if feature_column.size == 0 or label_array.size == 0:
        return 0.0
    if np.allclose(feature_column.std(ddof=0), 0.0):
        return 0.0
    if np.allclose(label_array.std(ddof=0), 0.0):
        return 0.0

    corr = np.corrcoef(feature_column, label_array)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(corr)


def save_feature_statistics(
    train_full_features: Array2D,
    y_train: Array1D,
    feature_descriptors: Sequence[FeatureDescriptor],
    rank_lookup: Dict[int, int],
    output_dir: str,
    logger: logging.Logger,
) -> None:
    rows: List[Dict[str, object]] = []

    for feature_index, descriptor in enumerate(feature_descriptors):
        column = train_full_features[:, feature_index]
        rows.append(
            {
                "feature_index": feature_index,
                "sequence": descriptor.sequence,
                "feature_class": descriptor.feature_class,
                "feature_name": descriptor.feature_name,
                "full_name": descriptor.full_name,
                "mean": float(np.mean(column)),
                "variance": float(np.var(column)),
                "pearson_corr_with_egfr_status": safe_pearson_correlation(column, y_train),
                "selected_rank": rank_lookup.get(feature_index),
                "selected": feature_index in rank_lookup,
            }
        )

    stats_df = pd.DataFrame(rows)
    stats_path = os.path.join(output_dir, "feature_statistics.csv")
    stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")
    logger.info("Saved feature statistics to %s", stats_path)


def build_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def fit_transform_clinical_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    location_pca_components: int,
    logger: logging.Logger,
) -> Tuple[Array2D, Array2D, Array2D, Dict[str, object]]:
    if location_pca_components < 1:
        raise ValueError("location_pca_components must be >= 1")

    age_scaler = StandardScaler()
    train_age = age_scaler.fit_transform(train_df[["age"]])
    val_age = (
        age_scaler.transform(val_df[["age"]])
        if not val_df.empty
        else np.zeros((0, 1), dtype=np.float32)
    )
    test_age = (
        age_scaler.transform(test_df[["age"]])
        if not test_df.empty
        else np.zeros((0, 1), dtype=np.float32)
    )

    location_encoder = build_one_hot_encoder()
    train_location_ohe = location_encoder.fit_transform(train_df[["location"]])
    val_location_ohe = (
        location_encoder.transform(val_df[["location"]])
        if not val_df.empty
        else np.zeros((0, train_location_ohe.shape[1]), dtype=np.float32)
    )
    test_location_ohe = (
        location_encoder.transform(test_df[["location"]])
        if not test_df.empty
        else np.zeros((0, train_location_ohe.shape[1]), dtype=np.float32)
    )

    effective_location_components = min(
        location_pca_components,
        train_location_ohe.shape[0],
        train_location_ohe.shape[1],
    )
    if effective_location_components != location_pca_components:
        logger.warning(
            "Adjusted location PCA components from %d to %d because of data shape.",
            location_pca_components,
            effective_location_components,
        )

    location_pca = PCA(n_components=effective_location_components, random_state=42)
    train_location = location_pca.fit_transform(train_location_ohe)
    val_location = (
        location_pca.transform(val_location_ohe)
        if val_location_ohe.shape[0] > 0
        else np.zeros((0, effective_location_components), dtype=np.float32)
    )
    test_location = (
        location_pca.transform(test_location_ohe)
        if test_location_ohe.shape[0] > 0
        else np.zeros((0, effective_location_components), dtype=np.float32)
    )

    train_base = train_df[["sex", "smoking_history", "metastasis"]].to_numpy(dtype=np.float32)
    val_base = val_df[["sex", "smoking_history", "metastasis"]].to_numpy(dtype=np.float32)
    test_base = test_df[["sex", "smoking_history", "metastasis"]].to_numpy(dtype=np.float32)

    # Final order: sex | age_scaled | location_pca_* | smoking_history | metastasis
    train_clinical = np.hstack([train_base[:, [0]], train_age, train_location, train_base[:, 1:]]).astype(np.float32)
    val_clinical = np.hstack([val_base[:, [0]], val_age, val_location, val_base[:, 1:]]).astype(np.float32)
    test_clinical = np.hstack([test_base[:, [0]], test_age, test_location, test_base[:, 1:]]).astype(np.float32)

    logger.info(
        "Clinical feature dims | train=%s | val=%s | test=%s",
        train_clinical.shape,
        val_clinical.shape,
        test_clinical.shape,
    )
    if location_pca_components == 1:
        logger.info(
            "Clinical output defaults to 5 dimensions to match the requested .npy shapes."
        )
    else:
        logger.warning(
            "Clinical output uses location_pca_components=%d, so the final clinical dimension is %d instead of 5.",
            location_pca_components,
            train_clinical.shape[1],
        )

    preprocessors = {
        "age_scaler": age_scaler,
        "location_encoder": location_encoder,
        "location_pca": location_pca,
        "location_pca_components": int(effective_location_components),
        "clinical_feature_order": [
            "sex",
            "age_scaled",
            *[f"location_pca_{i + 1}" for i in range(effective_location_components)],
            "smoking_history",
            "metastasis",
        ],
    }
    return train_clinical, val_clinical, test_clinical, preprocessors


def save_failure_rows(failure_rows: Sequence[Dict[str, object]], output_dir: str, logger: logging.Logger) -> None:
    failure_path = os.path.join(output_dir, "feature_extraction_failures.csv")
    failure_df = pd.DataFrame(failure_rows)
    failure_df.to_csv(failure_path, index=False, encoding="utf-8-sig")
    logger.info("Saved failure report to %s", failure_path)


def save_full_radiomics_cache(
    train_full_features: Array2D,
    val_full_features: Array2D,
    test_full_features: Array2D,
    output_dir: str,
    logger: logging.Logger,
) -> None:
    np.save(
        os.path.join(output_dir, "train_radiomics_full.npy"),
        train_full_features.astype(np.float32, copy=False),
    )
    np.save(
        os.path.join(output_dir, "val_radiomics_full.npy"),
        val_full_features.astype(np.float32, copy=False),
    )
    np.save(
        os.path.join(output_dir, "test_radiomics_full.npy"),
        test_full_features.astype(np.float32, copy=False),
    )
    logger.info(
        "Saved full 749-dim radiomics caches to %s",
        output_dir,
    )


def save_summary_report(
    train_radiomics: Array2D,
    val_radiomics: Array2D,
    test_radiomics: Array2D,
    train_clinical: Array2D,
    val_clinical: Array2D,
    test_clinical: Array2D,
    radiomics_metadata: Dict[str, object],
    failure_rows: Sequence[Dict[str, object]],
    output_dir: str,
    logger: logging.Logger,
) -> None:
    lasso_model = radiomics_metadata.get("lasso_model")
    lasso_alpha = float(lasso_model.alpha_) if lasso_model is not None else None
    lasso_nonzero_before_topk = (
        int(np.count_nonzero(np.abs(lasso_model.coef_) > 1e-12))
        if lasso_model is not None
        else int(radiomics_metadata["lasso_nonzero_before_topk"])
    )
    summary = {
        "train_radiomics_shape": list(map(int, train_radiomics.shape)),
        "val_radiomics_shape": list(map(int, val_radiomics.shape)),
        "test_radiomics_shape": list(map(int, test_radiomics.shape)),
        "train_clinical_shape": list(map(int, train_clinical.shape)),
        "val_clinical_shape": list(map(int, val_clinical.shape)),
        "test_clinical_shape": list(map(int, test_clinical.shape)),
        "selected_radiomics_dim": int(train_radiomics.shape[1]),
        "selector_method": str(radiomics_metadata["selector_method"]),
        "lasso_alpha": lasso_alpha,
        "lasso_nonzero_before_topk": lasso_nonzero_before_topk,
        "nonzero_before_topk": int(radiomics_metadata["selector_nonzero_before_topk"]),
        "candidate_feature_count_after_variance_filter": int(
            radiomics_metadata["candidate_feature_count_after_variance_filter"]
        ),
        "zero_variance_feature_count": int(radiomics_metadata["zero_variance_feature_count"]),
        "failure_rows": int(len(failure_rows)),
    }

    summary_path = os.path.join(output_dir, "feature_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved summary report to %s", summary_path)


def extract_features(
    raw_dir: str,
    mask_dir: str,
    label_csv: str,
    output_dir: str,
    location_pca_components: int = 1,
) -> Dict[str, object]:
    logger = build_logger(output_dir)
    logger.info("Starting DECT radiomics extraction and LASSO selection.")
    logger.info("Raw image directory: %s", raw_dir)
    logger.info("Manual mask directory: %s", mask_dir)
    logger.info("Label CSV: %s", label_csv)
    logger.info("Output directory: %s", output_dir)
    logger.info("Location PCA components: %d", location_pca_components)

    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f"Raw image directory does not exist: {raw_dir}. "
            "Please rebuild the HU NIfTI inputs first."
        )
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Manual mask directory does not exist: {mask_dir}")
    if not os.path.isfile(label_csv):
        raise FileNotFoundError(f"Label CSV does not exist: {label_csv}")

    train_df, val_df, test_df = load_and_split_labels(label_csv, logger)
    feature_descriptors = build_feature_descriptors()
    extractor = make_radiomics_extractor()

    train_full_features, y_train, train_ids, train_failures = extract_split_radiomics(
        split_df=train_df,
        split_name="train",
        raw_dir=raw_dir,
        mask_dir=mask_dir,
        extractor=extractor,
        logger=logger,
    )
    val_full_features, y_val, val_ids, val_failures = extract_split_radiomics(
        split_df=val_df,
        split_name="val",
        raw_dir=raw_dir,
        mask_dir=mask_dir,
        extractor=extractor,
        logger=logger,
    )
    test_full_features, y_test, test_ids, test_failures = extract_split_radiomics(
        split_df=test_df,
        split_name="test",
        raw_dir=raw_dir,
        mask_dir=mask_dir,
        extractor=extractor,
        logger=logger,
    )

    all_failures = [*train_failures, *val_failures, *test_failures]
    save_full_radiomics_cache(
        train_full_features=train_full_features,
        val_full_features=val_full_features,
        test_full_features=test_full_features,
        output_dir=output_dir,
        logger=logger,
    )

    train_radiomics, val_radiomics, test_radiomics, radiomics_metadata = select_top_lasso_features(
        train_features=train_full_features,
        val_features=val_full_features,
        test_features=test_full_features,
        y_train=y_train,
        feature_descriptors=feature_descriptors,
        output_dir=output_dir,
        logger=logger,
    )

    save_feature_statistics(
        train_full_features=train_full_features,
        y_train=y_train,
        feature_descriptors=feature_descriptors,
        rank_lookup=radiomics_metadata["rank_lookup"],
        output_dir=output_dir,
        logger=logger,
    )

    train_clinical, val_clinical, test_clinical, clinical_preprocessors = fit_transform_clinical_features(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        location_pca_components=location_pca_components,
        logger=logger,
    )

    np.save(os.path.join(output_dir, "train_radiomics.npy"), train_radiomics)
    np.save(os.path.join(output_dir, "val_radiomics.npy"), val_radiomics)
    np.save(os.path.join(output_dir, "test_radiomics.npy"), test_radiomics)

    np.save(os.path.join(output_dir, "train_clinical.npy"), train_clinical)
    np.save(os.path.join(output_dir, "val_clinical.npy"), val_clinical)
    np.save(os.path.join(output_dir, "test_clinical.npy"), test_clinical)

    model_bundle = {
        "radiomics_scaler": radiomics_metadata["radiomics_scaler"],
        "lasso_model": radiomics_metadata["lasso_model"],
        "selector_model": radiomics_metadata["selector_model"],
        "selector_method": radiomics_metadata["selector_method"],
        "selected_indices": radiomics_metadata["selected_indices"],
        "selected_features": radiomics_metadata["selected_features_payload"],
        "full_feature_names": radiomics_metadata["full_feature_names"],
        "clinical_preprocessors": clinical_preprocessors,
        "radiomics_settings": RADIOMICS_SETTINGS,
        "sequence_order": list(SEQUENCE_KEYS),
        "feature_class_order": {k: list(v) for k, v in FEATURES_BY_CLASS.items()},
        "train_patient_ids": train_ids,
        "val_patient_ids": val_ids,
        "test_patient_ids": test_ids,
        "train_labels": y_train.astype(np.float32),
        "val_labels": y_val.astype(np.float32),
        "test_labels": y_test.astype(np.float32),
    }
    bundle_path = os.path.join(output_dir, "lasso_model.pkl")
    with open(bundle_path, "wb") as f:
        pickle.dump(model_bundle, f)
    logger.info("Saved model/preprocessor bundle to %s", bundle_path)

    metadata_bundle = {
        "selector_method": str(radiomics_metadata["selector_method"]),
        "selected_indices": [int(index) for index in radiomics_metadata["selected_indices"]],
        "selected_features": radiomics_metadata["selected_features_payload"],
        "full_feature_names": radiomics_metadata["full_feature_names"],
        "sequence_order": list(SEQUENCE_KEYS),
        "feature_class_order": {k: list(v) for k, v in FEATURES_BY_CLASS.items()},
        "train_patient_ids": train_ids,
        "val_patient_ids": val_ids,
        "test_patient_ids": test_ids,
        "train_labels": y_train.astype(np.float32).tolist(),
        "val_labels": y_val.astype(np.float32).tolist(),
        "test_labels": y_test.astype(np.float32).tolist(),
    }
    metadata_path = os.path.join(output_dir, "feature_bundle_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_bundle, f, indent=2, ensure_ascii=False)
    logger.info("Saved lightweight feature metadata to %s", metadata_path)

    save_failure_rows(all_failures, output_dir, logger)
    save_summary_report(
        train_radiomics=train_radiomics,
        val_radiomics=val_radiomics,
        test_radiomics=test_radiomics,
        train_clinical=train_clinical,
        val_clinical=val_clinical,
        test_clinical=test_clinical,
        radiomics_metadata=radiomics_metadata,
        failure_rows=all_failures,
        output_dir=output_dir,
        logger=logger,
    )

    logger.info("DECT radiomics extraction pipeline completed.")
    return {
        "train_radiomics_shape": train_radiomics.shape,
        "val_radiomics_shape": val_radiomics.shape,
        "test_radiomics_shape": test_radiomics.shape,
        "train_clinical_shape": train_clinical.shape,
        "val_clinical_shape": val_clinical.shape,
        "test_clinical_shape": test_clinical.shape,
        "failure_rows": len(all_failures),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract DECT radiomics features and run LASSO feature selection."
    )
    parser.add_argument(
        "--raw-dir",
        default=str(DEFAULT_RADIOMICS_INPUT_DIR),
        help="Directory containing per-patient HU image NIfTI files.",
    )
    parser.add_argument(
        "--mask-dir",
        default=str(MANUAL_MASK_DIR),
        help="Directory containing per-patient binary mask NIfTI files.",
    )
    parser.add_argument(
        "--label-csv",
        default=str(DEFAULT_LABEL_CSV),
        help="Path to labels.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RADIOMICS_OUTPUT_DIR),
        help="Directory to save extracted features and model artifacts.",
    )
    parser.add_argument(
        "--location-pca-components",
        type=int,
        default=1,
        help=(
            "Number of PCA components used for location. "
            "Default=1 to keep train_clinical.npy at 5 columns as requested."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_features(
        raw_dir=args.raw_dir,
        mask_dir=args.mask_dir,
        label_csv=args.label_csv,
        output_dir=args.output_dir,
        location_pca_components=args.location_pca_components,
    )
