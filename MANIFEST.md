# Code Release Manifest

This repository is source-only. It contains no images, masks, labels, patient
identifiers, patient-level predictions, fitted models, or result artifacts.

## Core manuscript analysis

- `scripts/run_egfr20ins_binary_strategies.py`: model definitions, data loading, RFE, calibration, and locked threshold utilities.
- `scripts/egfr20ins_paper_common.py`: shared paper-analysis orchestration and metrics.
- `scripts/run_egfr20ins_repeated_cv.py`: repeated development cross-validation.
- `scripts/run_egfr20ins_fixed_test_bootstrap.py`: fixed temporal-held-out evaluation and bootstrap summaries.
- `scripts/run_egfr20ins_calibration_dca.py`: exploratory calibration and decision-curve summaries.
- `scripts/run_egfr20ins_bootstrap_stability.py`: 1,000-resample fixed-prediction stability analysis.
- `scripts/run_egfr20ins_subgroup_analysis.py`: exploratory out-of-fold subgroup summaries.
- `scripts/run_egfr20ins_paper_explainability.py`: SHAP and waterfall generation.
- `scripts/egfr20ins_runtime_guard.py`: fail-fast frozen-runtime verification for manuscript-analysis entry points.
- `scripts/run_egfr20ins_paper_figures.py`: manuscript and supplementary figure generation.
- `scripts/plot_figure1_study_flow.py`: reproducible vector/raster Figure 1 study-flow rendering from aggregate counts only.
- `scripts/build_egfr20ins_paper_tables.py`: aggregate table generation.

## Image and radiomics provenance

- `scripts/preprocess_dect.py`: authorized DECT preprocessing.
- `scripts/nifti_mask_utils.py`: NIfTI and mask utilities.
- `scripts/extract_radiomics_lasso.py`: PyRadiomics extraction and feature filtering.

## Shared utilities

- `scripts/luad_tabular_utils.py`
- `scripts/clinical_tools.py`
- `scripts/path_defaults.py`
- `scripts/checkpoint_utils.py`
- `scripts/__init__.py`

## Documentation and controls

- `README.md`: installation, controlled inputs, and execution order.
- `CONFIGURATION_AND_PROVENANCE.md`: confirmed workflow and version fields.
- `requirements.txt`: pinned core analysis environment plus optional dependencies.
- `.gitignore`: excludes common clinical-data, model, output, and cache files.
- `LICENSE`: source-code licence; authors must confirm that its terms are appropriate before deposit.
- `DATA_ACCESS_AND_SCHEMA.md`: controlled-data route and minimum authorized input contract.
- `REPRODUCIBILITY_PASSPORT.md`: per-target computational evidence level, frozen contract, and archive boundary.
- `REPLAY_RECEIPTS/figure6_py310_20260929/`: public-safe receipt for the successful controlled Python 3.10.14 Figure 6 replay; patient-level artifacts are deliberately excluded.
- `PUBLIC_RELEASE_CHECKLIST.md`: author-controlled pre-deposit gates.
- `CHANGELOG.md`: versioned separation between the controlled result pack and the public source release.
