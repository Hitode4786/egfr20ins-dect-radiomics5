# Change Log

## 2026-09-29 — package-role separation

- **Controlled result pack retained as internal evidence source only; public source = local `09_Code_Release`; export directory never ships publicly.**
- The desktop/server result export may contain real-data-derived tables, logs,
  pseudonymous case identifiers, patient-level predictions, and absolute paths.
  It is therefore a governed provenance/evidence source, not a public release
  artifact.
- The public source of truth is this repaired local `09_Code_Release`, including
  the frozen environment, runtime guard, public-safe Figure 6 replay receipt, and
  controlled-data access contract. Do not replace it with a later raw export.
