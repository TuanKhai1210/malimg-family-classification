<div align="center">

# MalImg Family Classification

**A reproducible machine learning project for classifying malware families from binary-derived images.**<br>
Pretrained visual features · Traditional classifiers · Leakage-aware evaluation

[![Project status](https://img.shields.io/badge/status-active%20development-2563eb?style=flat-square)](#project-status)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](#getting-started)
[![PyTorch planned](https://img.shields.io/badge/PyTorch-feature%20pipeline%20planned-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](#method)
[![Quality checks](https://github.com/TuanKhai1210/malimg-family-classification/actions/workflows/quality.yml/badge.svg?branch=main)](https://github.com/TuanKhai1210/malimg-family-classification/actions/workflows/quality.yml)

[Overview](#overview) · [Getting started](#getting-started) · [Method](#method) · [Project status](#project-status) · [Contributing](CONTRIBUTING.md)

</div>

## Overview

This three-person Machine Learning course project studies **which malware family** an image belongs to. Each grayscale image is derived from the bytes of one malware sample. The planned core pipeline extracts features with pretrained image models, saves those features, and fits traditional classifiers on them. It does not perform benign-versus-malicious detection.

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

The tests use small synthetic examples and do not download MalImg or train a model.

To reproduce the dataset inventory, download the public archive and verify its SHA-256 before inspecting it:

```bash
python scripts/download_malimg.py
python scripts/inspect_malimg.py data/MalImg_original_dataset.zip
```

The archive is approximately **1.17 GB**. The audit writes local files under `metadata/`; those files are evidence for the original 25-class archive, not an approved train/validation/test split. The public URL and audited checksum are recorded in [`configs/project.json`](configs/project.json).

## Method

The planned path is **image → preprocessing → frozen pretrained feature extractor → saved `.npy` features → traditional classifier → evaluation**. The feature matrix, labels, sample IDs, model weights, transforms, and manifest version must remain aligned and traceable.

The source archive contains 800 `Yuner.A` files with one unique decoded pixel image. The main benchmark excludes this family so that exact-pixel duplicates cannot be placed across training and held-out partitions for that class. The remaining images will be split by family and exact-pixel group; all files in a group must stay in one partition. This policy does not rule out near-duplicates or prove independence between malware samples.

The core comparison uses the same locked split and evaluation unit. Validation selects configurations; test is reserved for the final assessment. File-level metrics will be reported alongside the representative-per-group metrics so the effect of duplicate weighting remains visible. Fine-tuning is an optional extension after the core pipeline works.

## Project status

> **In progress — 5 October 2026.** The archive has been audited and the repository scaffold is available. Instructor approval of MalImg and the 24-class scope, the official split, pretrained features, model results, and clean Colab `Runtime > Run all` verification are still pending. No benchmark scores are claimed here.

The current [`notebooks/main_colab.ipynb`](notebooks/main_colab.ipynb) is an integration starter. The final course submission must install dependencies and prepare data in Colab automatically, run the full pipeline without mounting a personal drive, save and reload `.npy` or `.h5` features, and include a PDF report with real experiments and contribution evidence.

| Area | Owner | Next deliverable |
|---|---|---|
| Data, EDA, split | A | Approved 24-class manifest, label map, audited split v1 |
| Features and classifiers | B | Two pretrained extractors, feature files, E2/E3/E4 |
| Evaluation and Colab | C | Shared metrics, result registry, clean `Run all` |

Names and student details will be added after the first group meeting. The instructor must also confirm dataset uniqueness across groups and the 24-class scope.

## Repository structure

```text
configs/       Shared configuration
docs/          Project plan and dataset audit
features/      Saved feature output; large files are ignored by Git
metadata/      Versioned manifest, split, and label map once approved
modules/       Shared data contracts, feature I/O, classifiers, evaluation
notebooks/     Colab front end
reports/       Progress and final report artifacts
results/       Generated experiments and figures; ignored by Git
scripts/       Dataset download, audit, and repository checks
tests/         Small checks for alignment and evaluation contracts
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch names, Conventional Commit messages, pull request review, validation, and artifact handling. New commits and pull requests are checked by the [quality workflow](.github/workflows/quality.yml). Each experiment must record its config, split version, feature provenance, and result files before it enters the report.

## Dataset and citation

The dataset's README asks users to cite the original paper and states that the dataset should not be redistributed. This repository contains no raw images or original archive; use the public source link in the configuration. Do not commit raw data, credentials, or large feature caches. Check the course-approved sharing method before publishing extracted embeddings.

Nataraj et al., [*Malware Images: Visualization and Automatic Classification*](https://doi.org/10.1145/2016904.2016908), VizSec 2011.

Before final submission, the group must add the course code, instructor, member names and student IDs, and links to the report and Colab notebook, following the course assignment.
