"""
DECT ROI preprocessing script.

This script reads seven NIfTI image volumes for each patient. When manual masks
are available, it crops each sequence with a mask-guided bounding box; otherwise
it falls back to the older HU-threshold lung ROI heuristic. The cropped ROIs are
resized to (64, 128, 128), z-score normalized, and saved into one .npz file per
patient together with resized binary mask arrays for downstream use.
"""

import argparse
import hashlib
import logging
import os
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

from nifti_mask_utils import load_binary_mask_aligned_to_image
from path_defaults import (
    DEFAULT_PREPROCESS_INPUT_DIR,
    DEFAULT_PREPROCESS_OUTPUT_DIR,
    LABEL_CSV as DEFAULT_LABEL_CSV,
    MANUAL_MASK_DIR,
)


Array3D = np.ndarray
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SuspiciousRawInputError(RuntimeError):
    """Raised when raw DECT inputs look like masks or duplicated exports."""


@dataclass(frozen=True)
class SequenceSpec:
    key: str
    candidate_files: Sequence[str]


def _estimate_default_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, 8))


def _process_patient_worker(
    patient_row_dict: Dict[str, object],
    raw_dir: str,
    output_dir: str,
    label_csv: str,
    target_shape: Tuple[int, int, int],
    hu_range: Tuple[float, float],
    roi_padding: int,
    mask_dir: Optional[str],
    mask_required: bool,
    allow_suspicious_inputs: bool,
    patient_dir_map: Dict[str, str],
    mask_patient_dir_map: Dict[str, str],
) -> Dict[str, object]:
    patient_row = SimpleNamespace(**patient_row_dict)
    preprocessor = DECTPreprocessor(
        raw_dir=raw_dir,
        output_dir=output_dir,
        label_csv=label_csv,
        target_shape=target_shape,
        hu_range=hu_range,
        roi_padding=roi_padding,
        mask_dir=mask_dir,
        mask_required=mask_required,
        allow_suspicious_inputs=allow_suspicious_inputs,
        patient_dir_map=patient_dir_map,
        mask_patient_dir_map=mask_patient_dir_map,
        enable_file_logging=False,
        enable_stream_logging=False,
    )

    patient_id = str(patient_row.patient_id)
    start_time = time.perf_counter()

    try:
        patient_arrays, patient_meta = preprocessor._process_patient(patient_row)
        preprocessor._save_patient_npz(patient_id, patient_arrays)
        patient_failed = any(row["status"] != "success" for row in patient_meta)
        return {
            "patient_id": patient_id,
            "meta_rows": patient_meta,
            "patient_failed": patient_failed,
            "error": None,
            "elapsed_seconds": time.perf_counter() - start_time,
        }
    except SuspiciousRawInputError:
        raise
    except Exception as exc:
        zero_arrays = preprocessor._build_zero_patient_output()
        preprocessor._save_patient_npz(patient_id, zero_arrays)
        return {
            "patient_id": patient_id,
            "meta_rows": preprocessor._build_patient_failure_rows(patient_row, str(exc)),
            "patient_failed": True,
            "error": f"{exc}\n{traceback.format_exc()}",
            "elapsed_seconds": time.perf_counter() - start_time,
        }


class DECTPreprocessor:
    """
    Preprocess dual-energy CT (DECT) scans with optional manual-mask guidance.

    The pipeline follows the requested order strictly:
    1. Load label CSV and sort by patient number.
    2. For every patient and every sequence:
       - load NIfTI using nibabel
       - if a manual mask exists, compute the 3D bounding box from the mask
       - otherwise threshold HU into [-200, 100] and keep the largest component
       - expand the box by 10 voxels in each direction
       - resize the ROI to (64, 128, 128) with scipy.ndimage.zoom(order=3)
       - resize the ROI mask with nearest-neighbor interpolation
       - perform per-scan z-score normalization
    3. Save 7 image sequences plus 7 mask arrays into one .npz file per patient.
    4. If one sequence fails, fill that sequence with zeros without skipping the patient.
    5. Save metadata and a processing report.
    """

    SEQUENCE_SPECS: Tuple[SequenceSpec, ...] = (
        SequenceSpec("A80KV", ("A80KV.nii",)),
        SequenceSpec("A140KV", ("A140KV.nii",)),
        # Keep the required output key names AIC/VIC while accepting common legacy aliases.
        SequenceSpec("AIC", ("AIC.nii", "A1C.nii")),
        SequenceSpec("C", ("C.nii",)),
        SequenceSpec("V80KV", ("V80KV.nii",)),
        SequenceSpec("V140KV", ("V140KV.nii",)),
        SequenceSpec("VIC", ("VIC.nii", "V1C.nii")),
    )
    DUPLICATE_SEQUENCE_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
        ("arterial", ("A80KV", "A140KV", "AIC")),
        ("venous", ("V80KV", "V140KV", "VIC")),
    )

    def __init__(
        self,
        raw_dir: str,
        output_dir: str,
        label_csv: str,
        target_shape: Tuple[int, int, int] = (64, 128, 128),
        hu_range: Tuple[float, float] = (-200.0, 100.0),
        roi_padding: int = 10,
        mask_dir: Optional[str] = None,
        mask_required: bool = True,
        allow_suspicious_inputs: bool = False,
        patient_dir_map: Optional[Dict[str, str]] = None,
        mask_patient_dir_map: Optional[Dict[str, str]] = None,
        enable_file_logging: bool = True,
        enable_stream_logging: bool = True,
    ) -> None:
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.label_csv = label_csv
        self.target_shape = target_shape
        self.hu_range = hu_range
        self.roi_padding = roi_padding
        self.mask_dir = mask_dir
        self.mask_required = bool(mask_required and mask_dir)
        self.allow_suspicious_inputs = allow_suspicious_inputs
        self.patient_dir_map = patient_dir_map or self._build_patient_dir_map(self.raw_dir)
        self.mask_patient_dir_map = mask_patient_dir_map or self._build_patient_dir_map(self.mask_dir)
        self.enable_file_logging = enable_file_logging
        self.enable_stream_logging = enable_stream_logging
        self.log_path = os.path.join(self.output_dir, "preprocess_log.txt")
        self.meta_path = os.path.join(self.output_dir, "meta.csv")
        self.logger = self._build_logger()

    def _build_logger(self) -> logging.Logger:
        os.makedirs(self.output_dir, exist_ok=True)
        logger_name = f"dect_preprocess_{id(self)}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.propagate = False

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        if self.enable_file_logging:
            file_handler = logging.FileHandler(self.log_path, mode="w", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        if self.enable_stream_logging:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        return logger

    def _build_patient_dir_map(self, base_dir: Optional[str]) -> Dict[str, str]:
        patient_dir_map: Dict[str, str] = {}
        if base_dir is None or not os.path.isdir(base_dir):
            return patient_dir_map

        for entry in os.scandir(base_dir):
            if not entry.is_dir():
                continue
            patient_dir_map[entry.name.lower()] = entry.path
        return patient_dir_map

    def preprocess_all(self) -> pd.DataFrame:
        return self.preprocess_all_serial()

    def preprocess_all_serial(self) -> pd.DataFrame:
        # Record the configuration at the start so the log file is self-contained.
        self.logger.info("Starting DECT preprocessing.")
        self.logger.info("Raw directory: %s", self.raw_dir)
        self.logger.info("Output directory: %s", self.output_dir)
        self.logger.info("Label CSV: %s", self.label_csv)
        self.logger.info("Target shape: %s", self.target_shape)
        self.logger.info("HU range: %s", self.hu_range)
        self.logger.info("ROI padding: %d", self.roi_padding)
        self.logger.info("Mask directory: %s", self.mask_dir if self.mask_dir else "(disabled)")
        self.logger.info("Mask required: %s", self.mask_required)
        self.logger.info("Allow suspicious inputs: %s", self.allow_suspicious_inputs)

        labels_df = self._load_and_sort_labels()
        meta_rows: List[Dict[str, object]] = []

        total_patients = len(labels_df)
        progress_bar = tqdm(
            labels_df.itertuples(index=False),
            total=total_patients,
            desc="Preprocessing patients",
        )

        for patient_index, patient_row in enumerate(progress_bar, start=1):
            patient_id = str(patient_row.patient_id)
            try:
                patient_arrays, patient_meta = self._process_patient(patient_row)
                self._save_patient_npz(patient_id, patient_arrays)
                meta_rows.extend(patient_meta)
            except SuspiciousRawInputError as exc:
                self.logger.error(
                    "Suspicious raw input detected for %s: %s",
                    patient_id,
                    exc,
                )
                raise
            except Exception as exc:
                self.logger.error(
                    "Patient-level failure for %s: %s\n%s",
                    patient_id,
                    exc,
                    traceback.format_exc(),
                )
                zero_arrays = self._build_zero_patient_output()
                self._save_patient_npz(patient_id, zero_arrays)
                meta_rows.extend(self._build_patient_failure_rows(patient_row, str(exc)))

            if patient_index % 10 == 0 or patient_index == total_patients:
                self.logger.info(
                    "Progress: processed %d / %d patients.",
                    patient_index,
                    total_patients,
                )

        meta_df = pd.DataFrame(meta_rows)
        meta_df.to_csv(self.meta_path, index=False, encoding="utf-8-sig")
        self.logger.info("Saved metadata to %s", self.meta_path)

        self._write_processing_report(meta_df, total_patients)
        self.logger.info("DECT preprocessing completed.")
        return meta_df

    def preprocess_all_parallel(self, num_workers: Optional[int] = None) -> pd.DataFrame:
        start_time = time.perf_counter()
        num_workers = int(num_workers or _estimate_default_workers())
        num_workers = max(1, num_workers)

        self.logger.info("Starting DECT preprocessing.")
        self.logger.info("Raw directory: %s", self.raw_dir)
        self.logger.info("Output directory: %s", self.output_dir)
        self.logger.info("Label CSV: %s", self.label_csv)
        self.logger.info("Target shape: %s", self.target_shape)
        self.logger.info("HU range: %s", self.hu_range)
        self.logger.info("ROI padding: %d", self.roi_padding)
        self.logger.info("Mask directory: %s", self.mask_dir if self.mask_dir else "(disabled)")
        self.logger.info("Mask required: %s", self.mask_required)
        self.logger.info("Allow suspicious inputs: %s", self.allow_suspicious_inputs)
        self.logger.info("Parallel workers: %d", num_workers)

        labels_df = self._load_and_sort_labels()
        total_patients = len(labels_df)
        meta_rows: List[Dict[str, object]] = []
        patient_failures = 0

        patient_rows = labels_df.to_dict(orient="records")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _process_patient_worker,
                    patient_row_dict,
                    self.raw_dir,
                    self.output_dir,
                    self.label_csv,
                    self.target_shape,
                    self.hu_range,
                    self.roi_padding,
                    self.mask_dir,
                    self.mask_required,
                    self.allow_suspicious_inputs,
                    self.patient_dir_map,
                    self.mask_patient_dir_map,
                )
                for patient_row_dict in patient_rows
            ]

            progress_bar = tqdm(
                as_completed(futures),
                total=total_patients,
                desc=f"Preprocessing patients ({num_workers} workers)",
            )

            for patient_index, future in enumerate(progress_bar, start=1):
                try:
                    result = future.result()
                except SuspiciousRawInputError as exc:
                    self.logger.error("Suspicious raw input detected: %s", exc)
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                meta_rows.extend(result["meta_rows"])
                if result["patient_failed"]:
                    patient_failures += 1
                if result["error"]:
                    self.logger.error(
                        "Patient-level failure for %s: %s",
                        result["patient_id"],
                        result["error"],
                    )
                if patient_index % 10 == 0 or patient_index == total_patients:
                    self.logger.info(
                        "Progress: processed %d / %d patients.",
                        patient_index,
                        total_patients,
                    )

        meta_df = pd.DataFrame(meta_rows)
        meta_df.to_csv(self.meta_path, index=False, encoding="utf-8-sig")
        self.logger.info("Saved metadata to %s", self.meta_path)

        self._write_processing_report(meta_df, total_patients)

        elapsed_seconds = time.perf_counter() - start_time
        success_patients = int(total_patients - patient_failures)
        self.logger.info(
            "Parallel preprocessing completed | success=%d | failed=%d | elapsed_seconds=%.2f",
            success_patients,
            patient_failures,
            elapsed_seconds,
        )
        return meta_df

    def _load_and_sort_labels(self) -> pd.DataFrame:
        labels_df = pd.read_csv(self.label_csv)
        required_columns = {
            "patient_id",
            "sex",
            "age",
            "location",
            "egfr_status",
            "smoking_history",
            "metastasis",
        }
        missing_columns = sorted(required_columns - set(labels_df.columns))
        if missing_columns:
            raise ValueError(
                f"Missing required columns in labels.csv: {missing_columns}"
            )

        # Extract the numeric suffix from patient IDs such as P_001 for stable sorting.
        labels_df["patient_num"] = labels_df["patient_id"].astype(str).map(
            self._extract_patient_num
        )
        if labels_df["patient_num"].isna().any():
            bad_ids = labels_df.loc[
                labels_df["patient_num"].isna(), "patient_id"
            ].tolist()
            raise ValueError(f"Failed to parse patient numbers from IDs: {bad_ids}")

        labels_df = labels_df.sort_values(
            by=["patient_num", "patient_id"], ascending=[True, True]
        ).reset_index(drop=True)
        self.logger.info("Loaded and sorted %d label rows.", len(labels_df))
        return labels_df

    @staticmethod
    def _extract_patient_num(patient_id: str) -> Optional[int]:
        match = re.search(r"(\d+)$", str(patient_id))
        if match is None:
            return None
        return int(match.group(1))

    def _process_patient(
        self, patient_row: pd.Series
    ) -> Tuple[Dict[str, Array3D], List[Dict[str, object]]]:
        patient_id = str(patient_row.patient_id)
        patient_dir = self._resolve_patient_dir(patient_id)
        mask_patient_dir = self._resolve_mask_patient_dir(patient_id)

        patient_arrays = self._build_zero_patient_output()
        patient_meta: List[Dict[str, object]] = []
        fallback_roi_bounds: Optional[Tuple[Tuple[int, int], ...]] = None

        if patient_dir is None or not os.path.isdir(patient_dir):
            self.logger.warning("Patient directory not found: %s", patient_dir)
            return patient_arrays, self._build_patient_failure_rows(
                patient_row, f"Missing patient directory: {patient_dir}"
            )
        if self.mask_required and (mask_patient_dir is None or not os.path.isdir(mask_patient_dir)):
            self.logger.warning("Mask directory not found for %s: %s", patient_id, mask_patient_dir)
            return patient_arrays, self._build_patient_failure_rows(
                patient_row, f"Missing mask directory: {mask_patient_dir}"
            )

        resolved_sequence_files = self._resolve_patient_sequence_files(patient_dir)
        self._validate_patient_raw_inputs(patient_id, patient_dir, resolved_sequence_files)

        for sequence_spec in self.SEQUENCE_SPECS:
            sequence_key = sequence_spec.key
            try:
                # Process every sequence independently so one failure does not drop the patient.
                processed_volume, meta_row = self._process_sequence(
                    patient_dir=patient_dir,
                    patient_row=patient_row,
                    sequence_spec=sequence_spec,
                    mask_patient_dir=mask_patient_dir,
                    source_file=resolved_sequence_files.get(sequence_key),
                    fallback_roi_bounds=fallback_roi_bounds,
                )
                processed_mask = np.asarray(meta_row.pop("_processed_mask"), dtype=np.float32)
                patient_arrays[sequence_key] = processed_volume
                patient_arrays[f"{sequence_key}_mask"] = processed_mask
                patient_meta.append(meta_row)
                if meta_row["status"] == "success" and meta_row["used_fallback_roi"] is False:
                    fallback_roi_bounds = (
                        (int(meta_row["roi_d_start"]), int(meta_row["roi_d_end"])),
                        (int(meta_row["roi_h_start"]), int(meta_row["roi_h_end"])),
                        (int(meta_row["roi_w_start"]), int(meta_row["roi_w_end"])),
                    )
            except SuspiciousRawInputError:
                raise
            except Exception as exc:
                self.logger.warning(
                    "Sequence failure | patient=%s | sequence=%s | error=%s",
                    patient_id,
                    sequence_key,
                    exc,
                )
                patient_arrays[sequence_key] = self._build_zero_volume()
                patient_arrays[f"{sequence_key}_mask"] = self._build_zero_volume()
                patient_meta.append(
                    self._build_sequence_failure_row(
                        patient_row=patient_row,
                        sequence_key=sequence_key,
                        source_file=None,
                        reason=str(exc),
                    )
                )

        return patient_arrays, patient_meta

    def _resolve_patient_dir(self, patient_id: str) -> Optional[str]:
        return self.patient_dir_map.get(str(patient_id).lower())

    def _resolve_mask_patient_dir(self, patient_id: str) -> Optional[str]:
        return self.mask_patient_dir_map.get(str(patient_id).lower())

    def _resolve_patient_sequence_files(self, patient_dir: str) -> Dict[str, str]:
        resolved: Dict[str, str] = {}
        for sequence_spec in self.SEQUENCE_SPECS:
            source_file = self._find_existing_sequence_file(
                patient_dir, sequence_spec.candidate_files
            )
            if source_file is not None:
                resolved[sequence_spec.key] = source_file
        return resolved

    @staticmethod
    def _quick_fingerprint(path: Path, chunk_size: int = 64 * 1024) -> str:
        digest = hashlib.md5()
        file_size = path.stat().st_size
        digest.update(str(file_size).encode("ascii"))
        with path.open("rb") as handle:
            digest.update(handle.read(chunk_size))
            if file_size > chunk_size:
                middle_offset = max(0, (file_size // 2) - (chunk_size // 2))
                handle.seek(middle_offset)
                digest.update(handle.read(chunk_size))
            if file_size > 2 * chunk_size:
                handle.seek(max(0, file_size - chunk_size))
                digest.update(handle.read(chunk_size))
        return digest.hexdigest()[:12]

    def _validate_patient_raw_inputs(
        self,
        patient_id: str,
        patient_dir: str,
        resolved_sequence_files: Dict[str, str],
    ) -> None:
        if self.allow_suspicious_inputs:
            return

        duplicate_messages: List[str] = []
        patient_dir_path = Path(patient_dir)
        for group_name, sequence_keys in self.DUPLICATE_SEQUENCE_GROUPS:
            if not all(sequence_key in resolved_sequence_files for sequence_key in sequence_keys):
                continue
            fingerprints = []
            details = []
            for sequence_key in sequence_keys:
                source_file = resolved_sequence_files[sequence_key]
                fingerprint = self._quick_fingerprint(patient_dir_path / source_file)
                fingerprints.append(fingerprint)
                details.append(f"{sequence_key}={source_file}#{fingerprint}")
            if len(set(fingerprints)) == 1:
                duplicate_messages.append(f"{group_name} triplet duplicated ({', '.join(details)})")

        if duplicate_messages:
            raise SuspiciousRawInputError(
                f"Patient {patient_id} has suspicious duplicate raw DECT exports: {'; '.join(duplicate_messages)}. "
                "Run scripts/audit_raw_dect.py and fix data/raw before preprocessing."
            )

    def _process_sequence(
        self,
        patient_dir: str,
        patient_row: pd.Series,
        sequence_spec: SequenceSpec,
        mask_patient_dir: Optional[str] = None,
        source_file: Optional[str] = None,
        fallback_roi_bounds: Optional[Tuple[Tuple[int, int], ...]] = None,
    ) -> Tuple[Array3D, Dict[str, object]]:
        if source_file is None:
            source_file = self._find_existing_sequence_file(
                patient_dir, sequence_spec.candidate_files
            )
            if source_file is None:
                raise FileNotFoundError(
                    f"None of the candidate files exist: {list(sequence_spec.candidate_files)}"
                )

        nifti_path = os.path.join(patient_dir, source_file)
        image = nib.load(nifti_path)
        # NIfTI commonly loads as (H, W, D). Convert to (D, H, W) for model input.
        volume_hwd = image.get_fdata(dtype=np.float32)
        if volume_hwd.ndim != 3:
            raise ValueError(f"Expected a 3D volume, got shape {volume_hwd.shape}")

        volume_dhw = np.transpose(volume_hwd, (2, 0, 1)).astype(np.float32, copy=False)
        raw_dtype = str(image.header.get_data_dtype())
        raw_min = float(np.min(volume_dhw))
        raw_max = float(np.max(volume_dhw))
        raw_mean = float(np.mean(volume_dhw))
        raw_std = float(np.std(volume_dhw))
        self._validate_sequence_raw_inputs(
            patient_id=str(patient_row.patient_id),
            sequence_key=sequence_spec.key,
            source_file=source_file,
            raw_dtype=raw_dtype,
            raw_min=raw_min,
            raw_max=raw_max,
        )

        used_fallback_roi = False
        fallback_reason = ""
        roi_source = "auto_threshold"
        mask_source_file: Optional[str] = None
        mask_voxels = 0
        mask_resampled = False
        mask_alignment_reason = ""
        mask_original_shape_hwd: Optional[Tuple[int, int, int]] = None
        roi_mask = None

        use_manual_mask = mask_patient_dir is not None
        if use_manual_mask:
            try:
                roi_source = "manual_mask"
                (
                    bbox,
                    roi_mask,
                    mask_source_file,
                    mask_voxels,
                    mask_resampled,
                    mask_alignment_reason,
                    mask_original_shape_hwd,
                ) = self._extract_mask_guided_roi(
                    mask_patient_dir=mask_patient_dir,
                    sequence_spec=sequence_spec,
                    image=image,
                )
                if mask_resampled:
                    self.logger.info(
                        "Resampled mask to image grid | patient=%s | sequence=%s | mask_file=%s | details=%s",
                        patient_row.patient_id,
                        sequence_spec.key,
                        mask_source_file,
                        mask_alignment_reason,
                    )
                roi_bounds = self._expand_bounding_box(bbox, volume_dhw.shape, self.roi_padding)
                roi_volume = self._crop_volume(volume_dhw, roi_bounds)
                roi_mask = self._crop_volume(roi_mask.astype(np.float32, copy=False), roi_bounds)
                candidate_voxels = mask_voxels
                component_voxels = mask_voxels
            except Exception as exc:
                if self.mask_required:
                    raise
                self.logger.warning(
                    "Manual mask unavailable for %s %s; falling back to auto ROI: %s",
                    patient_row.patient_id,
                    sequence_spec.key,
                    exc,
                )
                use_manual_mask = False

        if not use_manual_mask:
            try:
                # Follow the requested sequence strictly: threshold -> connected components
                # -> largest component -> bounding box -> padding -> crop -> resize -> normalize.
                lung_mask, candidate_voxels, component_voxels = self._extract_largest_component(
                    volume_dhw
                )
                bbox = self._compute_bounding_box(lung_mask)
                roi_bounds = self._expand_bounding_box(bbox, volume_dhw.shape, self.roi_padding)
                roi_volume = self._crop_volume(volume_dhw, roi_bounds)
                roi_mask = self._crop_volume(lung_mask.astype(np.float32), roi_bounds)
            except Exception as exc:
                if fallback_roi_bounds is None:
                    raise
                used_fallback_roi = True
                fallback_reason = str(exc)
                candidate_voxels = 0
                component_voxels = 0
                bbox = fallback_roi_bounds
                roi_bounds = fallback_roi_bounds
                roi_volume = self._crop_volume(volume_dhw, roi_bounds)
                roi_mask = np.ones_like(roi_volume, dtype=np.float32)
                roi_source = "fallback_bbox"

        if roi_volume.size == 0:
            raise ValueError("Extracted ROI is empty.")
        if not np.isfinite(roi_volume).all():
            raise ValueError("Extracted ROI contains non-finite values.")
        if np.allclose(roi_volume, 0.0):
            raise ValueError("Extracted ROI is all zeros.")
        if roi_mask is None or int(np.count_nonzero(roi_mask > 0.5)) == 0:
            raise ValueError("Extracted ROI mask is empty.")

        resized_roi = self._resize_to_target(roi_volume)
        resized_mask = self._resize_mask_to_target(roi_mask)
        normalized_roi, roi_mean, roi_std_before_norm, used_zero_fill = (
            self._zscore_normalize(resized_roi)
        )
        resized_mask = (resized_mask > 0.5).astype(np.float32, copy=False)
        if int(np.count_nonzero(resized_mask)) == 0:
            raise ValueError("Resized ROI mask is empty.")

        meta_row = self._build_sequence_success_row(
            patient_row=patient_row,
            sequence_key=sequence_spec.key,
            source_file=source_file,
            mask_source_file=mask_source_file,
            roi_source=roi_source,
            original_shape=tuple(int(v) for v in volume_hwd.shape),
            working_shape=tuple(int(v) for v in volume_dhw.shape),
            raw_dtype=raw_dtype,
            candidate_voxels=candidate_voxels,
            component_voxels=component_voxels,
            mask_voxels=mask_voxels,
            mask_resampled=mask_resampled,
            mask_alignment_reason=mask_alignment_reason,
            mask_original_shape_hwd=mask_original_shape_hwd,
            bbox=bbox,
            roi_bounds=roi_bounds,
            roi_shape_before_resize=tuple(int(v) for v in roi_volume.shape),
            resized_shape=tuple(int(v) for v in normalized_roi.shape),
            raw_min=raw_min,
            raw_max=raw_max,
            raw_mean=raw_mean,
            raw_std=raw_std,
            roi_mean=roi_mean,
            roi_std_before_norm=roi_std_before_norm,
            used_zero_fill=used_zero_fill,
            used_fallback_roi=used_fallback_roi,
            fallback_reason=fallback_reason,
        )
        meta_row["_processed_mask"] = resized_mask
        return normalized_roi, meta_row

    def _extract_mask_guided_roi(
        self,
        mask_patient_dir: str,
        sequence_spec: SequenceSpec,
        image: nib.spatialimages.SpatialImage,
    ) -> Tuple[Tuple[Tuple[int, int], ...], Array3D, str, int, bool, str, Tuple[int, int, int]]:
        mask_source_file = self._find_existing_sequence_file(mask_patient_dir, sequence_spec.candidate_files)
        if mask_source_file is None:
            raise FileNotFoundError(
                f"Missing mask file for {sequence_spec.key}: {list(sequence_spec.candidate_files)}"
            )

        mask_path = os.path.join(mask_patient_dir, mask_source_file)
        alignment = load_binary_mask_aligned_to_image(image, mask_path)
        mask_hwd = alignment.mask_hwd

        mask_dhw = np.transpose(mask_hwd > 0.5, (2, 0, 1)).astype(np.float32, copy=False)
        mask_voxels = int(np.count_nonzero(mask_dhw))
        if mask_voxels == 0:
            raise ValueError(f"Manual mask for {sequence_spec.key} is empty.")

        bbox = self._compute_bounding_box(mask_dhw)
        return (
            bbox,
            mask_dhw,
            mask_source_file,
            mask_voxels,
            alignment.was_resampled,
            alignment.reason,
            alignment.original_shape_hwd,
        )

    def _validate_sequence_raw_inputs(
        self,
        patient_id: str,
        sequence_key: str,
        source_file: str,
        raw_dtype: str,
        raw_min: float,
        raw_max: float,
    ) -> None:
        if self.allow_suspicious_inputs:
            return
        if raw_min >= -1e-6 and raw_max <= 1.0 + 1e-6:
            raise SuspiciousRawInputError(
                f"Patient {patient_id} sequence {sequence_key} ({source_file}) has suspicious raw range "
                f"[{raw_min:.3f}, {raw_max:.3f}] with dtype {raw_dtype}. "
                "This looks like a mask or normalized export, not HU CT data. "
                "Run scripts/audit_raw_dect.py and fix data/raw before preprocessing, "
                "or rerun with --allow-suspicious-inputs if you explicitly want to bypass this guard."
            )

    def _find_existing_sequence_file(
        self, patient_dir: str, candidate_files: Sequence[str]
    ) -> Optional[str]:
        for filename in candidate_files:
            file_path = os.path.join(patient_dir, filename)
            if os.path.isfile(file_path):
                return filename
        return None

    def _extract_largest_component(
        self, volume_dhw: Array3D
    ) -> Tuple[Array3D, int, int]:
        hu_min, hu_max = self.hu_range
        # Keep only voxels inside the specified HU window as candidate lung-region voxels.
        candidate_mask = (volume_dhw >= hu_min) & (volume_dhw <= hu_max)
        candidate_voxels = int(np.count_nonzero(candidate_mask))
        if candidate_voxels == 0:
            raise ValueError(
                f"No voxels found within HU range [{hu_min}, {hu_max}]."
            )

        structure = ndimage.generate_binary_structure(rank=3, connectivity=3)
        labeled_mask, num_components = ndimage.label(candidate_mask, structure=structure)
        if num_components == 0:
            raise ValueError("Connected component analysis found zero components.")

        component_sizes = np.bincount(labeled_mask.ravel())
        component_sizes[0] = 0
        largest_label = int(np.argmax(component_sizes))
        largest_size = int(component_sizes[largest_label])
        if largest_size == 0:
            raise ValueError("Largest connected component has zero voxels.")

        largest_component_mask = labeled_mask == largest_label
        return largest_component_mask, candidate_voxels, largest_size

    @staticmethod
    def _compute_bounding_box(mask_dhw: Array3D) -> Tuple[Tuple[int, int], ...]:
        coordinates = np.where(mask_dhw)
        if coordinates[0].size == 0:
            raise ValueError("Cannot compute bounding box from an empty mask.")

        bbox: List[Tuple[int, int]] = []
        for axis_coords in coordinates:
            axis_min = int(axis_coords.min())
            axis_max = int(axis_coords.max())
            bbox.append((axis_min, axis_max))
        return tuple(bbox)

    @staticmethod
    def _expand_bounding_box(
        bbox: Tuple[Tuple[int, int], ...],
        shape: Tuple[int, int, int],
        padding: int,
    ) -> Tuple[Tuple[int, int], ...]:
        expanded: List[Tuple[int, int]] = []
        for axis, (lower, upper) in enumerate(bbox):
            start = max(0, lower - padding)
            end = min(shape[axis] - 1, upper + padding)
            expanded.append((int(start), int(end)))
        return tuple(expanded)

    @staticmethod
    def _crop_volume(
        volume_dhw: Array3D, roi_bounds: Tuple[Tuple[int, int], ...]
    ) -> Array3D:
        # The bounding box uses inclusive end indices, so crop with end + 1.
        slices = tuple(slice(start, end + 1) for start, end in roi_bounds)
        return volume_dhw[slices]

    def _resize_to_target(self, volume_dhw: Array3D) -> Array3D:
        # Use cubic interpolation exactly as requested.
        zoom_factors = tuple(
            target_dim / current_dim
            for target_dim, current_dim in zip(self.target_shape, volume_dhw.shape)
        )
        resized = ndimage.zoom(volume_dhw, zoom=zoom_factors, order=3)
        resized = resized.astype(np.float32, copy=False)
        return self._crop_or_pad_to_shape(resized, self.target_shape)

    def _resize_mask_to_target(self, mask_dhw: Array3D) -> Array3D:
        zoom_factors = tuple(
            target_dim / current_dim
            for target_dim, current_dim in zip(self.target_shape, mask_dhw.shape)
        )
        resized = ndimage.zoom(mask_dhw.astype(np.float32, copy=False), zoom=zoom_factors, order=0)
        resized = resized.astype(np.float32, copy=False)
        return self._crop_or_pad_to_shape(resized, self.target_shape)

    @staticmethod
    def _crop_or_pad_to_shape(
        volume_dhw: Array3D, target_shape: Tuple[int, int, int]
    ) -> Array3D:
        result = volume_dhw

        for axis, target_dim in enumerate(target_shape):
            current_dim = result.shape[axis]
            if current_dim > target_dim:
                start = (current_dim - target_dim) // 2
                end = start + target_dim
                slicer = [slice(None)] * result.ndim
                slicer[axis] = slice(start, end)
                result = result[tuple(slicer)]
            elif current_dim < target_dim:
                pad_before = (target_dim - current_dim) // 2
                pad_after = target_dim - current_dim - pad_before
                pad_width = [(0, 0)] * result.ndim
                pad_width[axis] = (pad_before, pad_after)
                result = np.pad(result, pad_width=pad_width, mode="constant")

        return result.astype(np.float32, copy=False)

    def _zscore_normalize(
        self, volume_dhw: Array3D
    ) -> Tuple[Array3D, float, float, bool]:
        # This stage uses per-scan statistics for now, which matches the requested fallback.
        mean_value = float(np.mean(volume_dhw))
        std_value = float(np.std(volume_dhw))

        if not np.isfinite(mean_value) or not np.isfinite(std_value):
            raise ValueError("Cannot normalize ROI with non-finite mean/std.")

        if std_value < 1e-6:
            # Degenerate ROIs cannot be normalized reliably; fall back to zeros and record it.
            normalized = self._build_zero_volume()
            return normalized, mean_value, std_value, True

        normalized = (volume_dhw - mean_value) / std_value
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        return normalized.astype(np.float32, copy=False), mean_value, std_value, False

    def _save_patient_npz(self, patient_id: str, patient_arrays: Dict[str, Array3D]) -> None:
        npz_path = os.path.join(self.output_dir, f"{patient_id}.npz")
        np.savez_compressed(npz_path, **patient_arrays)

    def _build_zero_volume(self) -> Array3D:
        return np.zeros(self.target_shape, dtype=np.float32)

    def _build_zero_patient_output(self) -> Dict[str, Array3D]:
        patient_output: Dict[str, Array3D] = {}
        for sequence_spec in self.SEQUENCE_SPECS:
            patient_output[sequence_spec.key] = self._build_zero_volume()
            patient_output[f"{sequence_spec.key}_mask"] = self._build_zero_volume()
        return patient_output

    def _base_meta_row(
        self,
        patient_row: pd.Series,
        sequence_key: str,
        status: str,
        source_file: Optional[str],
        reason: str,
    ) -> Dict[str, object]:
        return {
            "patient_id": str(patient_row.patient_id),
            "patient_num": int(patient_row.patient_num),
            "sequence": sequence_key,
            "sex": patient_row.sex,
            "age": patient_row.age,
            "location": patient_row.location,
            "egfr_status": patient_row.egfr_status,
            "smoking_history": patient_row.smoking_history,
            "metastasis": patient_row.metastasis,
            "status": status,
            "reason": reason,
            "source_file": source_file,
            "mask_source_file": None,
            "roi_source": None,
            "raw_dtype": None,
            "original_shape_hwd": None,
            "working_shape_dhw": None,
            "mask_original_shape_hwd": None,
            "mask_resampled": False,
            "mask_alignment_reason": "",
            "candidate_voxels": None,
            "largest_component_voxels": None,
            "mask_voxels": None,
            "bbox_d_start": None,
            "bbox_d_end": None,
            "bbox_h_start": None,
            "bbox_h_end": None,
            "bbox_w_start": None,
            "bbox_w_end": None,
            "roi_d_start": None,
            "roi_d_end": None,
            "roi_h_start": None,
            "roi_h_end": None,
            "roi_w_start": None,
            "roi_w_end": None,
            "roi_shape_before_resize": None,
            "resized_shape": None,
            "raw_min_hu": None,
            "raw_max_hu": None,
            "raw_mean_hu": None,
            "raw_std_hu": None,
            "roi_mean_before_norm": None,
            "roi_std_before_norm": None,
            "used_zero_fill": True,
            "used_fallback_roi": False,
            "fallback_reason": "",
            "output_shape": "x".join(map(str, self.target_shape)),
        }

    def _build_sequence_success_row(
        self,
        patient_row: pd.Series,
        sequence_key: str,
        source_file: str,
        mask_source_file: Optional[str],
        roi_source: str,
        original_shape: Tuple[int, int, int],
        working_shape: Tuple[int, int, int],
        raw_dtype: str,
        candidate_voxels: int,
        component_voxels: int,
        mask_voxels: int,
        mask_resampled: bool,
        mask_alignment_reason: str,
        mask_original_shape_hwd: Optional[Tuple[int, int, int]],
        bbox: Tuple[Tuple[int, int], ...],
        roi_bounds: Tuple[Tuple[int, int], ...],
        roi_shape_before_resize: Tuple[int, int, int],
        resized_shape: Tuple[int, int, int],
        raw_min: float,
        raw_max: float,
        raw_mean: float,
        raw_std: float,
        roi_mean: float,
        roi_std_before_norm: float,
        used_zero_fill: bool,
        used_fallback_roi: bool,
        fallback_reason: str,
    ) -> Dict[str, object]:
        row = self._base_meta_row(
            patient_row=patient_row,
            sequence_key=sequence_key,
            status="success",
            source_file=source_file,
            reason="",
        )
        row.update(
            {
                "mask_source_file": mask_source_file,
                "roi_source": roi_source,
                "raw_dtype": raw_dtype,
                "original_shape_hwd": "x".join(map(str, original_shape)),
                "working_shape_dhw": "x".join(map(str, working_shape)),
                "mask_original_shape_hwd": (
                    None if mask_original_shape_hwd is None else "x".join(map(str, mask_original_shape_hwd))
                ),
                "mask_resampled": bool(mask_resampled),
                "mask_alignment_reason": mask_alignment_reason,
                "candidate_voxels": candidate_voxels,
                "largest_component_voxels": component_voxels,
                "mask_voxels": mask_voxels,
                "bbox_d_start": bbox[0][0],
                "bbox_d_end": bbox[0][1],
                "bbox_h_start": bbox[1][0],
                "bbox_h_end": bbox[1][1],
                "bbox_w_start": bbox[2][0],
                "bbox_w_end": bbox[2][1],
                "roi_d_start": roi_bounds[0][0],
                "roi_d_end": roi_bounds[0][1],
                "roi_h_start": roi_bounds[1][0],
                "roi_h_end": roi_bounds[1][1],
                "roi_w_start": roi_bounds[2][0],
                "roi_w_end": roi_bounds[2][1],
                "roi_shape_before_resize": "x".join(map(str, roi_shape_before_resize)),
                "resized_shape": "x".join(map(str, resized_shape)),
                "raw_min_hu": raw_min,
                "raw_max_hu": raw_max,
                "raw_mean_hu": raw_mean,
                "raw_std_hu": raw_std,
                "roi_mean_before_norm": roi_mean,
                "roi_std_before_norm": roi_std_before_norm,
                "used_zero_fill": bool(used_zero_fill),
                "used_fallback_roi": bool(used_fallback_roi),
                "fallback_reason": fallback_reason,
            }
        )
        return row

    def _build_sequence_failure_row(
        self,
        patient_row: pd.Series,
        sequence_key: str,
        source_file: Optional[str],
        reason: str,
    ) -> Dict[str, object]:
        return self._base_meta_row(
            patient_row=patient_row,
            sequence_key=sequence_key,
            status="failed",
            source_file=source_file,
            reason=reason,
        )

    def _build_patient_failure_rows(
        self, patient_row: pd.Series, reason: str
    ) -> List[Dict[str, object]]:
        return [
            self._build_sequence_failure_row(
                patient_row=patient_row,
                sequence_key=sequence_spec.key,
                source_file=None,
                reason=reason,
            )
            for sequence_spec in self.SEQUENCE_SPECS
        ]

    def _write_processing_report(self, meta_df: pd.DataFrame, total_patients: int) -> None:
        total_sequences = total_patients * len(self.SEQUENCE_SPECS)
        success_mask = meta_df["status"] == "success"
        success_sequences = int(success_mask.sum())
        failed_sequences = int(total_sequences - success_sequences)

        patient_status = meta_df.groupby("patient_id")["status"].apply(
            lambda values: (values == "success").all()
        )
        patients_all_success = int(patient_status.sum())
        patients_any_failure = int(total_patients - patients_all_success)

        self.logger.info("========== Processing Report ==========")
        self.logger.info("Patients total: %d", total_patients)
        self.logger.info("Patients all-success: %d", patients_all_success)
        self.logger.info("Patients with any failure: %d", patients_any_failure)
        self.logger.info("Sequences total: %d", total_sequences)
        self.logger.info("Sequences success: %d", success_sequences)
        self.logger.info("Sequences failed: %d", failed_sequences)

        success_df = meta_df.loc[success_mask].copy()
        if success_df.empty:
            self.logger.warning("No successful sequences found; HU/size statistics unavailable.")
            return

        working_shape_split = success_df["working_shape_dhw"].str.split("x", expand=True)
        working_shape_split = working_shape_split.astype(int)
        success_df["depth"] = working_shape_split[0]
        success_df["height"] = working_shape_split[1]
        success_df["width"] = working_shape_split[2]

        self.logger.info(
            "HU stats | raw_min mean=%.3f | raw_max mean=%.3f | raw_mean mean=%.3f | raw_std mean=%.3f",
            float(success_df["raw_min_hu"].mean()),
            float(success_df["raw_max_hu"].mean()),
            float(success_df["raw_mean_hu"].mean()),
            float(success_df["raw_std_hu"].mean()),
        )
        self.logger.info(
            "HU extrema | raw_min global=%.3f | raw_max global=%.3f",
            float(success_df["raw_min_hu"].min()),
            float(success_df["raw_max_hu"].max()),
        )
        suspicious_unit_interval_mask = (
            (success_df["raw_min_hu"] >= -1e-6) & (success_df["raw_max_hu"] <= 1.0 + 1e-6)
        )
        suspicious_unit_interval_count = int(suspicious_unit_interval_mask.sum())
        if suspicious_unit_interval_count > 0:
            self.logger.warning(
                "Suspicious raw intensity range detected in %d successful sequences. "
                "These inputs fall inside [0, 1] and may be masks or normalized exports rather than HU CT volumes. "
                "Run scripts/audit_raw_dect.py before training.",
                suspicious_unit_interval_count,
            )
            for row in success_df.loc[
                suspicious_unit_interval_mask, ["patient_id", "sequence", "source_file", "raw_min_hu", "raw_max_hu"]
            ].head(20).itertuples(index=False):
                self.logger.warning(
                    "Suspicious range | patient=%s | sequence=%s | source=%s | raw_min=%.3f | raw_max=%.3f",
                    row.patient_id,
                    row.sequence,
                    row.source_file,
                    row.raw_min_hu,
                    row.raw_max_hu,
                )
        self.logger.info(
            "Size stats (D,H,W) | depth min/mean/max=%d / %.2f / %d | "
            "height min/mean/max=%d / %.2f / %d | width min/mean/max=%d / %.2f / %d",
            int(success_df["depth"].min()),
            float(success_df["depth"].mean()),
            int(success_df["depth"].max()),
            int(success_df["height"].min()),
            float(success_df["height"].mean()),
            int(success_df["height"].max()),
            int(success_df["width"].min()),
            float(success_df["width"].mean()),
            int(success_df["width"].max()),
        )

        failure_df = meta_df.loc[~success_mask, ["patient_id", "sequence", "reason"]]
        if not failure_df.empty:
            self.logger.info("Failure examples (up to 20 rows):")
            for row in failure_df.head(20).itertuples(index=False):
                self.logger.info(
                    "patient=%s | sequence=%s | reason=%s",
                    row.patient_id,
                    row.sequence,
                    row.reason,
                )


def preprocess_all(raw_dir: str, output_dir: str, label_csv: str, mask_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Main entry point required by the task description.

    Parameters
    ----------
    raw_dir : str
        Directory containing patient folders such as P_001, P_002, ...
    output_dir : str
        Directory where .npz files, meta.csv and preprocess_log.txt will be saved.
    label_csv : str
        CSV file containing patient labels and metadata.
    """
    preprocessor = DECTPreprocessor(raw_dir=raw_dir, output_dir=output_dir, label_csv=label_csv, mask_dir=mask_dir)
    return preprocessor.preprocess_all()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess DECT NIfTI volumes into fixed-size ROI .npz files."
    )
    parser.add_argument(
        "--raw-dir",
        default=str(DEFAULT_PREPROCESS_INPUT_DIR),
        help="Directory containing patient folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_PREPROCESS_OUTPUT_DIR),
        help="Directory to save .npz files, meta.csv and preprocess_log.txt.",
    )
    parser.add_argument(
        "--label-csv",
        default=str(DEFAULT_LABEL_CSV),
        help="Path to labels.csv.",
    )
    parser.add_argument(
        "--mask-dir",
        default=str(MANUAL_MASK_DIR),
        help="Directory containing per-patient binary masks.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_estimate_default_workers(),
        help="Number of parallel worker processes to use.",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Run serial preprocessing instead of parallel mode.",
    )
    parser.add_argument(
        "--allow-suspicious-inputs",
        action="store_true",
        help="Bypass raw-input guards for unit-interval or duplicated DECT exports.",
    )
    parser.add_argument(
        "--allow-auto-roi",
        action="store_true",
        help="Allow fallback to the older auto ROI heuristic when manual masks are unavailable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocessor = DECTPreprocessor(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        label_csv=args.label_csv,
        mask_dir=args.mask_dir if str(args.mask_dir).strip() else None,
        mask_required=not args.allow_auto_roi,
        allow_suspicious_inputs=args.allow_suspicious_inputs,
    )
    if args.serial:
        preprocessor.preprocess_all_serial()
    else:
        preprocessor.preprocess_all_parallel(num_workers=args.workers)
