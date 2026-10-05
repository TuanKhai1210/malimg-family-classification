# Malware Family Classification from MalImg Images

Machine Learning course assignment, Semester I 2026–2027. The three-person group plans to classify malware **families** from binary-derived grayscale images using pretrained visual features and traditional classifiers. The internal submission deadline is **30 November 2026**.

**Status (5 October 2026):** this is the starter repository. The MalImg archive has been audited, but the instructor's dataset/subset approval, official split, pretrained features, model results and final Colab `Run all` verification are pending. No scores in this repository are presented as results.

## Decision and scope

- Use the public MalImg archive. The audited copy contains 9,339 images in 25 families.
- Exclude `Yuner.A` from the **main 24-class benchmark**: its 800 files have one unique decoded pixel image. The resulting planned set contains 8,539 files and 8,018 unique pixel groups.
- Keep exact pixel duplicates within one train/validation/test partition. Target approximately 70/15/15 by file count, subject to group constraints. The split is **not yet created**.
- Compare frozen ResNet50 + Logistic Regression (E2), frozen ResNet50 + Linear SVM (E3), and frozen ConvNeXt-Tiny + Linear SVM (E4). Feature files must be saved and reloaded before classifier fitting.
- Select configurations on validation, then evaluate once on the locked test set. Primary planned metric: Macro-F1 using one deterministic representative of each exact pixel group in validation/test; also report file-level metrics.
- Fine-tuning and GitHub presentation are extension/bonus work after the core pipeline is stable.

This dataset does not support benign-versus-malicious detection. Any conclusion concerns known family labels under this benchmark and split.

The complete task breakdown, acceptance criteria and schedule are in [docs/project_plan.md](docs/project_plan.md). The [audit note](docs/data_audit.md) records verified counts and checksum. Requirements come from `MLAssignment_ELv2.pdf` supplied to the team; the instructor must confirm MalImg and the 24-class subset and verify that another group has not selected the same dataset.

## Group roles

| Role | Owner | Initial deliverable | Reviewer |
|---|---|---|---|
| A | Name pending | Dataset card, 24-class manifest, split v1, EDA, instructor/LMS check | C |
| B | Name pending | Pretrained extraction, feature store, E2/E3/E4 | A |
| C | Name pending | Shared metrics, experiment registry, Colab integration and clean `Run all` | B |

All three members write their report sections. Replace role placeholders after the first group meeting and record contribution percentages from actual work and instructor guidance.

## Repository map

```text
configs/       Shared project settings; no local paths
docs/          Team plan and dataset audit
features/      Output directory for saved embeddings (ignored by Git)
metadata/      Approved manifest, split and label map will live here
modules/       Common contracts, feature I/O and evaluation helpers
notebooks/     Main Colab front end
reports/       Progress and final report sources/results
results/       Generated experiment outputs (ignored by Git)
scripts/       Download and dataset audit tools
tests/         Small contract checks on synthetic examples
```

`notebooks/main_colab.ipynb` currently checks repository structure and configuration. It is a **starter notebook**, not the final end-to-end assignment notebook. A/B/C will integrate public download, EDA, extraction, classifier comparison and final evaluation before claiming `Run all` compliance.

## Data access and reproducibility

The public archive URL is in `configs/project.json` and `scripts/download_malimg.py`. The audited archive is 1,174,609,734 bytes with SHA-256:

```text
9766ae9f1daa520e367fb486ca94728fe1485c0f5cb8314c312d77089a1fe9ec
```

After cloning, in a Python environment:

```bash
python -m pip install -r requirements.txt
python scripts/download_malimg.py
python scripts/inspect_malimg.py data/MalImg_original_dataset.zip
python -m unittest discover -s tests -v
```

The download is about 1.17 GB. The audit command writes local `metadata/malimg_manifest.csv` and `metadata/malimg_summary.json`; these are generated evidence, **not** the approved 24-class split. Never treat arbitrary archive paths as trusted extraction destinations. Do not commit the raw archive, images, feature caches, credentials or private data. The dataset README asks users to cite its paper and says the dataset should not be redistributed; use the public source link rather than uploading a copy to GitHub.

Citation: Nataraj et al., [*Malware Images: Visualization and Automatic Classification*](https://doi.org/10.1145/2016904.2016908), VizSec 2011.

## Course submission requirements

The final Colab notebook must run with `Runtime > Run all` from a clean runtime, install its dependencies, download/extract the dataset from a public link without mounting a personal drive, and use saved `.npy` or `.h5` features for downstream classification. The final ZIP must contain `notebooks/`, `modules/`, `reports/`, and `features/`; the report must include EDA, methods, real experiment results, analysis, task distribution and contribution percentages. The course uses its LMS for progress deadlines and the instructor's Drive folder for final submission.

Fill in before public final submission: course code, instructor, group name, member names, student IDs and emails, exact execution steps, and links to the report and Colab notebook.
