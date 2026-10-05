# Contributing

This is a three-person course project. The workflow makes each change reviewable and every reported result traceable.

## Branches and pull requests

1. Start from an up-to-date `main` and create a focused branch, such as `feat/data-split-v1`, `fix/feature-alignment`, or `docs/report-method`.
2. Keep one concern per pull request. Explain the problem, change, validation, and any change to data or evaluation in the [pull request template](.github/PULL_REQUEST_TEMPLATE.md).
3. Ask the reviewer assigned in [the project plan](docs/project_plan.md) to check the acceptance criteria before merging. Changes to the split, label map, metric, or feature format need cross-role review.
4. Prefer **squash merge** with a Conventional Commit pull request title. Do not force-push to `main` or rewrite shared history.
5. Record config, seed, split manifest hash, experiment ID, and relevant logs when a result is used in the report.

## Commit messages

Every new commit and pull request title uses [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/):

```text
<type>(<scope>): <imperative summary>
```

Use one of `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci`, or `chore`. Use a short lowercase scope such as `data`, `eda`, `features`, `models`, `eval`, `colab`, `report`, `readme`, `repo`, or `tests`. Keep the subject at most 72 characters; start the summary with a lowercase verb, omit the final period, and describe the change rather than the activity. Add a blank line and a body when the reason, tradeoff, or data impact needs explanation.

Good examples:

```text
docs(readme): clarify the benchmark scope and current status
feat(data): generate pixel-group split manifest
fix(features): preserve sample order when saving embeddings
test(eval): cover representative-per-group scoring
```

Avoid subjects such as `update`, `fix bug`, `final`, or `add files`. Mark a breaking interface change with `!` before the colon and explain the migration in the body, for example `refactor(features)!: version the feature-store schema`.

The [quality workflow](.github/workflows/quality.yml) checks pull request titles and new non-merge commits in each pull request or push to `main`. It also reruns when a pull request title is edited. Check a message locally with `python scripts/check_commit_message.py --message "docs(readme): explain the data policy"`. The existing initial scaffold commit predates this convention; do not rewrite published history to rename it.

These automated checks report failures; they do not by themselves prevent direct pushes or merging. Reviewers must confirm that the checks pass. [AGENTS.md](AGENTS.md) records the same working agreements for coding assistants.

## Before requesting review

- Run `python -m unittest discover -s tests -v` and describe any skipped check or missing dependency.
- Run `python scripts/check_commit_message.py --rev HEAD`.
- Keep generated data, feature arrays, images, archives, secrets, and local environment files out of Git.
- Verify any change to manifests, labels, splits, or saved features against the shared sample IDs. A score without config, split version, and metric unit does not belong in the report.
- Check that the Colab notebook still documents what is implemented and does not claim a successful final `Run all` until it has been verified from a clean runtime.

The first integration checkpoint is a small path from manifest row to image, extracted feature, saved and reloaded feature, and shared metric. Treat its output as a smoke test until the instructor approves the dataset scope and the group locks split v1.
