# Controlled Data Access and Input Contract

## Access class

The study data are human-participant imaging, pathology, and clinical records.
They are **controlled access** and are not included in this repository. Access
requires institutional approval and an applicable data-use agreement. This
repository must not be interpreted as permission to redistribute images, masks,
labels, patient-level features, fitted models, or prediction outputs.

## Minimum authorized input contract

An authorized replay requires the following, supplied inside a controlled
compute environment rather than committed to Git:

1. A de-identified cohort/label table with a stable study-specific patient key,
   EGFR subtype label, four prespecified clinical candidate fields, and the
   frozen development/held-out assignment.
2. Analysis-ready radiomic feature matrices aligned one-to-one to the cohort
   table, containing 107 features for each of seven DECT series (749 radiomic
   candidates).
3. The fixed prediction, bootstrap, and paper-table artifacts used as inputs to
   the downstream table/figure scripts, or authorization to regenerate them
   from the controlled source data.
4. The controlled configuration described in
   `CONFIGURATION_AND_PROVENANCE.md`, including merged-candidate RFE, 48
   retained features, sigmoid calibration, and the locked threshold contract.

## Public-release boundary

Do not add direct identifiers, dates, accession numbers, free text, raw DICOM,
NIfTI, masks, feature rows, patient-level predictions, local absolute paths,
access tokens, or data-use agreements to a public repository. A future public
synthetic fixture may test file schemas and command wiring, but it must be
clearly labelled as non-equivalent to the controlled study data and must not be
used to claim result reproduction.
