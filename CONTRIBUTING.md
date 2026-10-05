# Group workflow

1. Fill in A/B/C names in `README.md` and agree on the reviewer for each work item in `docs/project_plan.md`.
2. Create a short branch per work item, for example `data/split-v1` or `features/resnet50`. Use a pull request for review before merging shared interfaces.
3. Record config, split manifest checksum and experiment ID with every reported result. Review sample ID and label alignment before adding a model score to the report.
4. Do not commit raw data, feature caches, credentials or results containing unreviewed labels. Put small code tests and documentation in Git; place large artifacts through the course-approved sharing method.
5. Keep meeting notes and actual contribution evidence. The final contribution percentages and due dates follow instructor/LMS guidance.

The core integration checkpoint is: from a manifest row, load an image, extract and save a feature with its sample ID and label, reload it, and call the shared metric function. Mark this as a smoke test until the instructor approves the dataset scope and A/C lock split v1.
