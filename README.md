# EGFR Exon 20 Insertion DECT Radiomics: Analysis Code

This source-only repository accompanies the Radiology: Artificial Intelligence
submission *Dual-energy CT radiomics for EGFR exon 20 insertion detection in
lung adenocarcinoma: a cautionary rare-event modeling study*.

**Package-role rule:** the controlled result pack is retained as an internal
evidence source only; the public source is this local `09_Code_Release`; the
export directory never ships publicly. See `CHANGELOG.md`.

The release contains no raw or processed images, labels, patient identifiers,
patient-level feature matrices, predictions, fitted model objects, or model
checkpoints. The underlying human-participant data are restricted by institutional
governance and may be considered only through the institution's approved
data-use agreement process.

## Scope

The manuscript-specific analysis modules are in `scripts/`:

- repeated cross-validation and temporal held-out evaluation;
- threshold locking, bootstrap uncertainty, calibration and exploratory DCA;
- subgroup summaries, tables, figures, and SHAP interpretation;
- the shared model-fitting and metrics utilities.

Additional preprocessing and model-definition modules are retained to document
the broader original workflow. They are not required to reproduce the reported
tabular manuscript analyses from an authorized analysis-ready feature matrix.

## Environment

Create the frozen Python 3.10.14 environment before replay:

```bash
conda env create -f environment.yml
conda activate egfr20ins-paper
```

The manuscript-analysis environment was archived with Python 3.10.14,
PyRadiomics 3.0.1, scikit-learn 1.7.2, and XGBoost 3.2.0. The replay requires
the frozen environment; running under other versions triggers a deliberate
runtime-guard failure before analysis to prevent silent divergence. GPU-only modules additionally
require a compatible PyTorch/CUDA installation.

## Controlled Inputs

Before running the analysis, obtain institutional authorization for:

1. a de-identified cohort table containing the fields expected by the scripts;
2. de-identified analysis-ready radiomics and clinical feature matrices;
3. the fixed temporal development/held-out assignment, where permitted;
4. the parameter records required to reconstruct image preprocessing and
   radiomic extraction.

Do not commit controlled data, patient-level outputs, model weights, or local
paths. `.gitignore` blocks common sensitive file types and result directories.
`DATA_ACCESS_AND_SCHEMA.md` defines the minimum authorized input contract.

## Reproduction Sequence

Run from the repository root after setting the data-path environment variables
defined in `scripts/path_defaults.py`:

```bash
python -m scripts.run_egfr20ins_repeated_cv
python -m scripts.run_egfr20ins_fixed_test_bootstrap
python -m scripts.run_egfr20ins_calibration_dca
python -m scripts.run_egfr20ins_bootstrap_stability
python -m scripts.run_egfr20ins_subgroup_analysis
python -m scripts.build_egfr20ins_paper_tables
python -m scripts.run_egfr20ins_paper_figures
python -m scripts.run_egfr20ins_paper_explainability
python -m scripts.plot_figure1_study_flow --package-root ..
```

Some modules require the project-specific `run_egfr20ins_binary_strategies.py`
helper in `scripts/`; it is included in this release. Analyses use development
data only for feature selection, calibration, and threshold locking. The primary
temporal held-out evaluation must remain untouched until final evaluation.

The manuscript's reported internal analysis uses repeated 5-fold stratified
cross-validation on all 187 development cases (validation folds of 37 or 38
cases). The separate 156-case/31-case development split retained in selected
legacy helper functions is a fixed train/validation diagnostic, not a
cross-validation fold and not the source of the manuscript's repeated-CV
sample-size description.

The manuscript's primary branch applies RFE to the merged radiomic-plus-clinical
candidate matrix (`LUAD_EGFR20INS_RFE_POOL=merged`) and is frozen at 48 selected
features. Runtime assertions verify the four appended clinical candidate names
and fail if any clinical feature is selected into this branch. The artifact name
`fusion_rfe_xgboost_calibrated` is retained for compatibility; the branch should
be described as combined-candidate RFE, not as a fitted clinical-radiomics fusion
model.

## Replay Evidence

`REPLAY_RECEIPTS/figure6_py310_20260929/` contains a public-safe receipt for a
successful controlled Python 3.10.14 replay of Figure 6. It records exact
prediction alignment (`max_score_delta=0.00000000`), the resolved environment,
the standard output/error log, and input/output hashes. The corresponding
Figure 6 files are retained once in `../04_Main_Figures/` and match the receipt
hashes. The patient-level selected-sample artifacts remain controlled and are
not included.

The Figure 6 target is therefore `REPLAYED`. Other source modules are
`RERUNNABLE` from an authorized controlled environment; this repository does
not claim independent reproduction or external clinical validation. See
`REPRODUCIBILITY_PASSPORT.md`.

## Verification Before Deposit

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
```

Before creating a public release, an authorized author must review the Git
history and all files for protected health information, create a tagged
release, archive it with a version DOI, and add the repository URL and DOI to
the manuscript. `PUBLIC_RELEASE_CHECKLIST.md` records the remaining
author-controlled gates.

The optional pytest suite was not run in the workstation assembly environment.
Run it only after installing the pinned frozen environment and use the
bytecode-suppression setting above to keep the release tree clean.
