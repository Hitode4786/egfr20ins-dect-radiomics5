# Reproducibility Passport

Package version: `v1.0.0`  
Release date: `2026-10-08`  
Source commit/tag: `v1.0.0`

## Targeted evidence status

| Target | Evidence | Computational status | Boundary |
|---|---|---|---|
| Locked primary-branch Figure 6 waterfall exports | Python 3.10.14 receipt in `REPLAY_RECEIPTS/figure6_py310_20260929/` | `REPLAYED` | Same controlled inputs; not independent reproduction or external validation |
| Remaining manuscript analyses | Source, configuration, frozen environment, and controlled access route | `RERUNNABLE` | No whole-pipeline receipt is included in this public release |
| External clinical validation | Not assessed | `NOT_ASSESSED` | The temporal held-out cohort is not an external validation cohort |

## Frozen replay contract

- Entrypoints: `scripts/run_egfr20ins_binary_strategies.py` and the
  manuscript-analysis modules listed in `MANIFEST.md`.
- Environment: `environment.yml`; Python 3.10.14, numpy 1.26.4,
  scikit-learn 1.7.2, XGBoost 3.2.0, and SHAP 0.48.0.
- Runtime guard: `scripts/egfr20ins_runtime_guard.py` fails before analysis if
  these core versions differ, because RFE selection was observed to be
  version-sensitive.
- Primary configuration: merged radiomic-plus-clinical candidate RFE,
  48 selected features, sigmoid calibration, and the development-locked
  threshold described in `CONFIGURATION_AND_PROVENANCE.md`.
- Acceptance evidence for Figure 6: rebuilt prediction alignment
  `max_score_delta=0.00000000`; the original controlled output hashes are in
  `REPLAY_RECEIPTS/figure6_py310_20260929/input_output_sha256.txt`.

## Archive status

`ARCHIVE_CONDITIONAL`: code is MIT licensed and the public repository has a
controlled-data route, but authors must supply a version tag, public repository
URL, version DOI, author metadata, and institutional data-access contact before
deposit. No raw patient data are included.
