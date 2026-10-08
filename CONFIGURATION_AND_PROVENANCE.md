# Configuration and Provenance

## Confirmed acquisition facts

- `V80KV`, `V140KV`, and `VIC` denote venous-phase DECT series; `V` means venous, not virtual.
- All scans used 1.25-mm slice thickness. GE Revolution was used in 2021; GE Revolution and GE Revolution APEX were both used from 2023 through 2026.
- The cohort comprised 77 cases from 2021 and 171 cases from 2023-2026; 12 of 20 EGFR 20-ins cases were in the 2021 group.

## Source hierarchy

The original data-processing and training source was retained in the authors'
controlled server workspace. Later local snapshots were used to cross-check the
result-generation scripts and archived outputs. No patient data or patient-level
result artifacts are included in this release.

## Radiomics settings verified from the original extraction script

- PyRadiomics: `3.0.1` in the archived analysis environment
- `binWidth`: `25`
- `resampledPixelSpacing`: `None`
- interpolator: `sitkBSpline` when resampling is requested
- intensity normalization: disabled (`normalize=False`)
- normalization scale: `1`
- re-segmentation range: `None`
- label: `1`
- 107 features per DECT series across seven series, 749 candidates in total

## Primary XGBoost defaults verified from the original training script

- objective: `binary:logistic`
- evaluation metric: `logloss`
- tree method: `hist`
- estimators: `300`
- maximum depth: `3`
- learning rate: `0.03`
- subsample: `1.0`
- column subsampling: `0.8`
- minimum child weight: `1.0`
- `reg_alpha=0`, `reg_lambda=1`, `gamma=0`
- `n_jobs=8`; `random_state` set to the experiment seed

## Random seeds and evaluation

The repeated five-fold development analysis used seeds `42, 52, 62, 72, and
82`. The primary combined-candidate RFE branch (internal artifact name
`fusion_rfe_xgboost_calibrated`) used base seed `72`. Its candidate matrix
contained 749 radiomic features plus age, sex, smoking history, and tumor
location; RFE was applied to that combined matrix and retained 48 radiomic
features, with no clinical variable retained. Thus the frozen primary result is
not evidence for direct addition of clinical variables to the fitted predictor.
Feature selection, calibration, and threshold locking were confined to
development data. The temporal held-out cohort was evaluated only after those
operations were locked. The public analysis helper freezes this branch to a
merged RFE pool and 48 selected features, and raises an error unless all 48
selected indices fall within the radiomic columns before the four appended
clinical candidates.

## Confirmed workflow and version fields

XGBoost `3.2.0` was confirmed in the original server environment. Image
processing, manual tumor delineation, mask generation, and multi-sequence
spatial registration were performed in uAI Research Portal V1.6. Two senior
radiologists independently delineated the same cases and compared their masks.
When masks differed, the radiologists reviewed the case jointly and repeated
the delineation until consensus was reached; the consensus mask was used for
analysis. No quantitative inter-reader agreement metric was calculated. The
portal's built-in fully automated registration function aligned the
multi-sequence and multimodal images without manual registration. The supplied
records do not support reporting 3D Slicer, TotalSegmentator, Elastix, or
dcm2niix as part of the final workflow.

The confirmed Python analysis-package versions are recorded in the submission
package's statistical-software environment audit. RFE selection was found to
be sensitive to Python/scikit-learn version; the frozen runtime assertion
guards against silent divergence across environments. At the primary and
explainability entry points, the release fails before analysis unless Python
is 3.10.14 and `numpy`, `scikit-learn`, `xgboost`, and `shap` match the pinned
versions in `environment.yml`. The archived replay environment is specified in
`environment.yml` (Python 3.10.14 with locked analysis-package versions).
