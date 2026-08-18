# Compact result summary

## ChEMBL37-CANAL20

Across five protocols and five seeds, CANAL-ExtraTrees calibrated obtains:

| Metric | CANAL-ExtraTrees calibrated | ExtraTrees naive |
|---|---:|---:|
| Exact-label MAE | 0.762 | 0.770 |
| Censored NLL | 0.704 | 0.784 |
| Target-macro Spearman | 0.529 | 0.529 |
| Boundary satisfaction | 0.311 | 0.150 |

The equal-task paired aggregate NLL difference (CANAL minus naive) is -0.0780 with 95% CI [-0.1035, -0.0525]; 19 of 20 tasks favor CANAL and the Holm-adjusted p value is 0.000162. The aggregate MAE difference is -0.0075 with 95% CI [-0.0192, 0.0043]. Joint temporal-scaffold MAE improves by 0.0495 in the favorable direction with Holm-adjusted p = 0.0092.

The positive affine calibration leaves within-task rankings unchanged in every split-seed evaluation. Exact-label Spearman is therefore identical for CANAL-ExtraTrees calibrated and its uncalibrated backbone.

## BindingDB-IndependentStrict

After strict provenance and entity-pair filtering, CANAL-ExtraTrees calibrated leads the seven rerun variants in aggregate:

| Metric | Mean |
|---|---:|
| Exact-label MAE | 0.664 |
| Censored NLL | 0.430 |
| Target-macro Spearman | 0.565 |

The overlap audit starts from 16,944 BindingDB-All records. It flags 291 shared-article rows and 14 canonical molecule-target rows; exact molecule-target-endpoint and near-matched-measurement overlap are zero. Removing the union of the declared strict rules retains 16,639 records.

Machine-readable values, all methods, all protocols, five-seed variation, calibration diagnostics, and paired statistics are provided under `results/`.
