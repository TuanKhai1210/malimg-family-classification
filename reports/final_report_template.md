# MalImg Family Classification — Report Draft

**Status: template; no benchmark results have been filled in.**

## Course and group

Fill in course code, semester, academic year, instructor, group name, member
names, student IDs and emails. Confirm the dataset scope and LMS deadlines.

## Abstract

Write after the experiments. State the actual dataset scope, methods, primary
evaluation unit and measured findings. Do not claim an experiment that was omitted.

## Problem and objectives

Describe classification among known malware families from byte-derived images.
Explain the 24-class scope, exclusion of Yuner.A and remaining dataset limitations.

## Related work and references

Cite Nataraj et al. (VizSec 2011), the pretrained architectures/weights and tools
actually used. Explain the differences from published benchmarks before comparing.

## Data and EDA

Use `metadata/dataset_manifest_v1.csv`, `metadata/split_summary_v1.json` and
`results/eda/`. Report class imbalance, dimensions, modes, duplicate groups,
exclusions, file/group counts and achieved split ratios. Record approval status.

## Method

Explain configurable RGB conversion, direct resize/padding, normalization,
frozen feature extraction, feature persistence/alignment and train-only scaling.
Describe classifier settings, class weights and any PCA/HOG extension actually run.

## Experimental protocol

Record manifest hash, seed, software/hardware, exact weights and transforms,
training unit and C search budget. Explain validation selection and final test lock.
Primary metric: Macro-F1 on deterministic pixel-group representatives. Report
file-level metrics separately. SVM decision scores are not calibrated probabilities.

## Results and comparison

Insert real rows from `results/comparison.csv` after validation/final evaluation.
Mark the split and unit for every table. Compare E2/E3 for classifier choice and
E3/E4 for the actual pretrained representation configurations. Include per-class
F1/support, confusion figures and cost metadata. Do not use smoke-test metrics here.

## Error analysis and limitations

Use prediction/error artifacts to inspect common confusions. Discuss near-duplicates,
few-group classes, domain shift, different preprocessing/weights, duplicated training
files, the single seed and the limits of malware-family classification.

## Team contribution and evidence

| Member | Owned tasks | Reviewed work | Evidence | Contribution |
|---|---|---|---|---|
| A — fill name/ID | Data/EDA/split | Features/alignment | Fill links/minutes/commits | To confirm |
| B — fill name/ID | Features/classifiers | Evaluation | Fill links/minutes/commits | To confirm |
| C — fill name/ID | Evaluation/Colab/integration | Split | Fill links/minutes/commits | To confirm |

Record coding-assistant support accurately and follow course rules. Each member
must understand, review and explain their part; agree percentages from real work
and the instructor's interpretation.

## Reproduction and submission

Link the repository, final Colab notebook and accessible features. Verify a clean
Colab `Runtime > Run all`, export this completed report to `final_report.pdf`,
then create and inspect the final ZIP. Include environment and experiment records.
