<div align="center">

# MalImg Family Classification

**A reproducible machine learning project for classifying malware families from binary-derived images.**<br>
Pretrained visual features · Traditional classifiers · Leakage-aware evaluation

[![Project status](https://img.shields.io/badge/status-active%20development-2563eb?style=flat-square)](#project-status)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](#getting-started)
[![PyTorch](https://img.shields.io/badge/PyTorch-frozen%20features-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](#method)
[![Quality checks](https://github.com/TuanKhai1210/malimg-family-classification/actions/workflows/quality.yml/badge.svg?branch=main)](https://github.com/TuanKhai1210/malimg-family-classification/actions/workflows/quality.yml)

[Overview](#overview) · [Kaggle workspace](docs/kaggle_training.md) · [Getting started](#getting-started) · [Review and train](docs/review_and_train.md) · [Method](#method) · [Contributing](CONTRIBUTING.md)

</div>

## Overview

This three-person Machine Learning course project studies **which malware family** an image belongs to. Each grayscale image is derived from the bytes of one malware sample. The implemented core pipeline prepares an audited group split, extracts frozen pretrained features, saves and reloads those features, and fits traditional classifiers. The study covers known families in the selected benchmark.

| Project fact | Current plan |
|---|---|
| Course task | Task 3 — Machine Learning with Image Data |
| Dataset | MalImg; audited archive: 9,339 images across 25 families |
| Main benchmark | 24 families and 8,539 files after excluding `Yuner.A`, subject to instructor approval |
| Core experiments | ResNet50 + Logistic Regression; ResNet50 + Linear SVM; ConvNeXt-Tiny + Linear SVM |
| Primary evaluation | Macro-F1 on one deterministic representative per exact-pixel group in validation/test |
| Team and deadline | 3 members; submission deadline provided by the team: **30 November 2026** |

The [project plan](docs/project_plan.md) contains the task breakdown, owners A/B/C, schedule, deliverables, and acceptance criteria. The [data audit](docs/data_audit.md) explains the verified image counts and the decision to exclude `Yuner.A` from the main benchmark.

## Getting started

For Kaggle, import [`main_kaggle.ipynb`](notebooks/main_kaggle.ipynb) and follow the
[Kaggle training and recovery guide](docs/kaggle_training.md). It separates data,
training, analysis and explicit final evaluation, adds epoch-resumable SGD on
frozen features, and verifies backups to a private Kaggle Dataset after each epoch.
LR/LinearSVC retain trial-level checkpoints. Credentials belong in Kaggle Secrets;
local session files alone are not a durable backup.

Clone the repository and use Python 3.11:

```bash
git clone https://github.com/TuanKhai1210/malimg-family-classification.git
cd malimg-family-classification
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on macOS/Linux. Then run:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The tests use synthetic images/features and small classifier fits. They do not download MalImg, download pretrained weights or train a benchmark model. Install `requirements-dev.txt` to additionally validate notebook format.

Prepare images, candidate manifests, the label map, the pixel-group split and EDA:

```bash
python scripts/check_environment.py
python scripts/run_pipeline.py prepare
```

The archive is approximately **1.17 GB**. Preparation verifies its audited SHA-256 before selectively extracting images. An existing ZIP can be supplied with `--archive PATH`; it is still verified. Generated data and artifacts stay out of Git. The public URL, checksum, split settings, transforms and experiment search grids are in [`configs/project.json`](configs/project.json).

After the group reviews the configuration and candidate split, run frozen feature extraction and validation selection:

```bash
python scripts/run_pipeline.py train
```

This runs E2/E3/E4, reuses only verified matching feature caches, fits preprocessing on train, saves all C trials and selects each configuration using validation Macro-F1. After freezing the choices, explicitly evaluate the held-out test set:

```bash
python scripts/run_pipeline.py final
```

See the Vietnamese [review and training guide](docs/review_and_train.md) for Colab, GPU setup, outputs, optional E0/E1 baselines, final-test locks and packaging. Open [`main_colab.ipynb`](notebooks/main_colab.ipynb) in Colab; its first mode is `prepare` for review, and `all` enables the full validation pipeline. The final submission requires an explicit final-test run and clean Colab verification.

For long sessions, use the optional [checkpoint and recovery workflow](docs/checkpoint_and_recovery.md).
It saves trial status and progress, resumes verified completed trials, and backs up
features and versioned result snapshots to a mounted directory such as Google Drive.
Drive is disabled by default; public-data preparation remains independent of personal storage.
An interrupted classifier fit starts over, while completed trials can be reused.

## Method

The implemented path is **image → preprocessing → frozen pretrained feature extractor → saved `.npy` features → traditional classifier → evaluation**. Feature arrays, labels and sample IDs are checked against the manifest; weights, transforms, versions and artifact hashes are recorded. Cache signatures detect changes to data, extraction settings and implementation.

The source archive contains 800 `Yuner.A` files with one unique decoded pixel image. The main benchmark excludes this family so that exact-pixel duplicates cannot be placed across training and held-out partitions for that class. The remaining images are split by family and exact-pixel group; all files in a group stay in one partition. The current candidate contains **5,978 / 1,281 / 1,280 files** and **5,615 / 1,202 / 1,201 pixel groups** in train/validation/test. It remains subject to team review and instructor approval. This policy does not rule out near-duplicates or prove independence between malware samples.

The core comparison uses the same locked split and evaluation unit. Validation selects configurations; test is reserved for the final assessment. File-level metrics will be reported alongside the representative-per-group metrics so the effect of duplicate weighting remains visible. Fine-tuning is an optional extension after the core pipeline works.

## Project status

> **Reviewable implementation — 30 October 2026.** Full-image preparation, candidate splitting and EDA have run locally. The team's saved classical baseline artifacts were reviewed privately on 26 October. Frozen extraction, classical selection, epoch-resumable SGD, private Kaggle backup/recovery and evaluation have local synthetic validation; no SGD MalImg benchmark or live Kaggle upload success is claimed here. Instructor approval, group acceptance of the split, final testing/report and clean platform acceptance remain pending.

The [`notebooks/main_colab.ipynb`](notebooks/main_colab.ipynb) now installs dependencies, obtains data through the public source and calls the shared pipeline without mounting a personal drive. It saves and reloads `.npy` features. The final course submission also needs a real PDF report, group details and contribution evidence; [`final_report_template.md`](reports/final_report_template.md) is a drafting aid.

| Area | Owner | Next deliverable |
|---|---|---|
| Data, EDA, split | A | Review generated EDA/split and confirm the scope with the instructor |
| Features and classifiers | B | Review settings, then run features and E2/E3/E4 on GPU |
| Evaluation and Colab | C | Review result/lock behavior and verify clean Colab `Run all` |

Names and student details will be added after the first group meeting. The instructor must also confirm dataset uniqueness across groups and the 24-class scope.

## Repository structure

```text
configs/       Shared configuration
docs/          Project plan and dataset audit
features/      Saved feature output; large files are ignored by Git
metadata/      Versioned manifest, split, and label map once approved
modules/       Data, splits, EDA, frozen features, selection, evaluation, packaging
notebooks/     Colab and Kaggle front ends
reports/       Progress and final report artifacts
results/       Generated experiments and figures; ignored by Git
scripts/       Pipeline CLI, environment checks, packaging and audit utilities
tests/         Data/cache/alignment checks and synthetic end-to-end integration
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch names, Conventional Commit messages, pull request review, validation, and artifact handling. New commits and pull requests are checked by the [quality workflow](.github/workflows/quality.yml). Each experiment must record its config, split version, feature provenance, and result files before it enters the report.

## Dataset and citation

The dataset's README asks users to cite the original paper and states that the dataset should not be redistributed. This repository contains no raw images or original archive; use the public source link in the configuration. Do not commit raw data, credentials, or large feature caches. Check the course-approved sharing method before publishing extracted embeddings.

Nataraj et al., [*Malware Images: Visualization and Automatic Classification*](https://doi.org/10.1145/2016904.2016908), VizSec 2011.

Before final submission, the group must add the course code, instructor, member names and student IDs, and links to the report and Colab notebook, following the course assignment.
