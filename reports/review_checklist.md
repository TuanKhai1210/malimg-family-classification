# Review before the first benchmark run

- [ ] A verifies the inventory, exclusions, group counts and achieved split ratios.
- [ ] C reviews that no pixel hash crosses train/validation/test and that
  representatives are chosen by path before any prediction.
- [ ] A confirms dataset uniqueness/24-class approval with the instructor and
  records it honestly in configuration and the report.
- [ ] B reviews resize policy, weights, dimensions, batch size/device and train-only
  scaling/PCA. Start with E2/E3/E4; optional baseline snippets are separate.
- [ ] B/C agree the C grid and training unit before the first validation run.
- [ ] Everyone checks that X, y and sample IDs refer to the same image.
- [ ] Run environment checks, unit tests and the small integration check.
- [ ] Run validation, inspect warnings and compare configurations before final test.
- [ ] Freeze the choices; run final test once. Changing choices afterwards requires
  a documented new protocol/output directory, not deletion of the final lock.
- [ ] A second member verifies clean Colab Run all, artifacts and links.
- [ ] Fill the actual report, member/course details and collaboration evidence.
- [ ] Inspect the final ZIP and submit through the instructor's designated channel.
