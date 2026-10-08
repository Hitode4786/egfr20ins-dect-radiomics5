# Figure 6 Controlled Replay Receipt

## Target and evidence level

Target: the four Figure 6 waterfall export formats produced by
`run_egfr20ins_paper_explainability.py` for the locked primary
combined-candidate RFE branch.

Computational evidence level: **REPLAYED** for this Figure 6 target only. The
controlled Python 3.10.14 run rebuilt the locked branch with
`max_score_delta=0.00000000` and completed the render using
`xgboost_pred_contribs_raw`. This is not external validation, independent
reproduction, or evidence that the full manuscript pipeline was rerun in this
release.

## Included public-safe evidence

- `replay_receipt.md`: run timestamp, prediction-alignment result, render
  status, terminology check, and TIFF geometry/compression check.
- `figure6_py310.stdout_stderr.log`: preserved standard output/error from the
  successful controlled run.
- `environment_python_version.txt` and `environment_pip_freeze.txt`: resolved
  runtime evidence.
- `input_output_sha256.txt`: source and output identity list from the controlled
  run. The public copy uses logical artifact labels rather than controlled-server
  absolute paths.
- `source_snapshot/`: exact explainability script used in the run. Its SHA-256
  is `3acf767000cd61a351f1c07be664c2e00809e937026bf159d6de59f009a0f8af`,
  matching the controlled hash list.

The referenced Figure 6 outputs are retained once, under
`../../04_Main_Figures/Figure6_revised.{pdf,svg,png,tiff}`. Their hashes match
the receipt. This avoids duplicate delivery files.

## Deliberately excluded artifacts

The original selected-sample CSV and full explainability JSON contain
pseudonymous patient-level identifiers and prediction values. They remain in
the governed controlled archive and are not redistributed in this public code
release. Their identities remain listed in the controlled hash receipt where
lawful.

## Redaction note

The complete immutable server-side receipt remains under institutional control.
For public release, controlled-server absolute paths and local-build paths in the
preserved stdout/stderr log were replaced with neutral logical labels. The
redacted log carries its own SHA-256 (listed in `input_output_sha256.txt`); the
source snapshot hash, the Figure 6 output hashes, and the controlled-side
receipt identities are unchanged.
