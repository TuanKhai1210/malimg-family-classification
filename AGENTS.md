# Repository working agreements

Read `CONTRIBUTING.md` before making changes. Use `docs/project_plan.md` and
`docs/data_audit.md` for project scope and verified data facts.

## Commits and review

- When creating commits or pull requests, use the repository's Conventional
  Commits format: `type(scope): imperative summary`.
- Write English commit subjects, at most 72 characters, with a lowercase
  imperative verb and no final period. Describe the actual change; avoid
  messages such as `update`, `fix bug`, or `final`.
- Keep each commit focused on one purpose. Add a body explaining the reason,
  tradeoff, or data impact when the subject is insufficient.
- Validate the proposed message with `scripts/check_commit_message.py` before
  committing, then validate the resulting commit with `--rev HEAD`.
- Follow the branch and review workflow in `CONTRIBUTING.md`. Preserve published
  history and unrelated user changes; never rename old shared commits merely to
  apply the new convention.

## Validation and project claims

- Run the checks relevant to the change and `git diff --check` before committing.
  For Python or workflow changes, run `python -m unittest discover -s tests -v`.
  Report missing dependencies or skipped checks accurately.
- Keep the README's implemented/planned status accurate. Do not invent model
  results, instructor approval, a locked split, or a successful Colab run.
- Preserve sample/feature alignment and the group-based split policy. Document
  changes to data, metrics, or experimental contracts so teammates can review them.
- Keep credentials, raw datasets, feature caches, and generated outputs out of
  Git, following the repository's ignore rules and contribution guide.
