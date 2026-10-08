from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
FALLBACK_DATA_ROOTS: tuple[Path, ...] = ()
if PROJECT_ROOT.parent.name == "server_upload":
    parent_project_data_root = PROJECT_ROOT.parent.parent / "data"
    if parent_project_data_root != DATA_ROOT:
        FALLBACK_DATA_ROOTS = (parent_project_data_root,)

LEGACY_RAW_DIR = DATA_ROOT / "raw"
RAW_REBUILT_DIR = DATA_ROOT / "raw_rebuilt"
DICOM_RAW_DIR = DATA_ROOT / "dicom_raw"
LEGACY_MANUAL_MASK_DIR = DATA_ROOT / "manual_masks"

LEGACY_PROCESSED_DIR = DATA_ROOT / "processed"
PROCESSED_MASK_DIR = DATA_ROOT / "processed_mask"

LEGACY_FEATURE_DIR = DATA_ROOT / "features"
FEATURES_MASK_DIR = DATA_ROOT / "features_mask"

LABEL_CSV = DATA_ROOT / "labels.csv"


def prefer_existing_path(primary: Path, fallback: Path) -> Path:
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return primary


def prefer_existing_relative(primary_name: str, fallback_name: str) -> Path:
    for data_root in (DATA_ROOT, *FALLBACK_DATA_ROOTS):
        primary = data_root / primary_name
        if primary.exists():
            return primary
    for data_root in (DATA_ROOT, *FALLBACK_DATA_ROOTS):
        fallback = data_root / fallback_name
        if fallback.exists():
            return fallback
    return DATA_ROOT / primary_name


def prefer_existing_primary(primary_name: str) -> Path:
    for data_root in (DATA_ROOT, *FALLBACK_DATA_ROOTS):
        primary = data_root / primary_name
        if primary.exists():
            return primary
    return DATA_ROOT / primary_name


DEFAULT_PREPROCESS_INPUT_DIR = prefer_existing_primary("raw_rebuilt")
DEFAULT_PREPROCESS_OUTPUT_DIR = PROCESSED_MASK_DIR

DEFAULT_RADIOMICS_INPUT_DIR = prefer_existing_primary("raw_rebuilt")
DEFAULT_RADIOMICS_OUTPUT_DIR = FEATURES_MASK_DIR
MANUAL_MASK_DIR = prefer_existing_relative("manual_masks", "raw")

DEFAULT_TRAIN_PROCESSED_DIR = prefer_existing_relative("processed_mask", "processed")
DEFAULT_TRAIN_FEATURE_DIR = prefer_existing_relative("features_mask", "features")

DEFAULT_BUNDLE_RAW_DIR = prefer_existing_primary("raw_rebuilt")
DEFAULT_BUNDLE_PROCESSED_DIR = DEFAULT_TRAIN_PROCESSED_DIR
DEFAULT_BUNDLE_FEATURE_DIR = DEFAULT_TRAIN_FEATURE_DIR
