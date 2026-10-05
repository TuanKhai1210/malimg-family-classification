# Metadata to be produced by A and reviewed by C

Expected approved files:

- `dataset_manifest_v1.csv`: source sample ID, archive-relative path, family, dimensions, file hash, decoded pixel hash, inclusion flag and reason.
- `label_map_v1.json`: 24 family names sorted alphabetically and IDs 0–23.
- `split_manifest_v1.csv`: all included samples with `train`/`val`/`test` and deterministic `eval_representative` flag.
- `split_summary_v1.json`: counts of files and unique pixel groups by class and split, method, seed, constraints, and checksums.

Do not call these files final until the instructor confirms the 24-class scope and C verifies that no pixel hash appears in more than one partition. Store only metadata here; raw images belong in ignored `data/`.
