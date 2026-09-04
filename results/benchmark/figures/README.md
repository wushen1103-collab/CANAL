# Repository-hosted benchmark visualizations

These figures preserve the complete benchmark analyses outside the Scientific Data Supplementary Information. They are deposited as reusable benchmark artifacts rather than article-type results.

- `reference_performance_matrices.pdf`: all 15 methods across five split protocols and three metrics.
- `ranking_reversal_audit.pdf`: split-by-split Kendall rank correlations and within-split MAE-versus-NLL ranking agreement.
- `task_level_paired_effect_distributions.pdf`: target-level paired differences across split protocols.
- `falsification_sensitivity_controls.pdf`: semi-synthetic latent-recovery and metadata-permutation controls.
- `paired_effect_forest_plots.pdf`: paired means and confidence intervals for the calibrated and uncalibrated ExtraTrees reference routes.

The numerical sources are `../chembl_reference_metrics.csv`, `../chembl_split_summary.csv`, `../paired_target_metrics.csv`, `../paired_statistics.csv`, `../ranking_summary.csv`, and `../ranking_reversal.json`.
