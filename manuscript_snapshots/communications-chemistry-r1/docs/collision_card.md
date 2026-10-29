# Collision Card

Date: 2026-08-10

## CANAL Claim Boundary

CANAL models the recorded molecular activity label as an observation generated from
latent activity, assay/source/document provenance, censoring relation, and label
reliability. The primary evidence is calibration, censored likelihood, source/assay
OOD robustness, and external reliability proxies.

CANAL must not claim to recover the true biological activity or eliminate assay noise.
The defensible claim is better modeling of public-database observations under
assay/source/censoring heterogeneity.

## Near-Neighbor Work

| Work | Link | Overlap | CANAL Difference | Collision Risk |
|---|---|---|---|---|
| MOLAR | https://arxiv.org/abs/2606.1839 | Noisy-label molecular representation learning with latent clean property framing. | CANAL focuses on assay/source/document observation channels, censored relations, source-OOD evaluation, and reliability proxies. | Medium: noisy-label framing is close, but observation mechanism and evidence are distinct. |
| Censored-label UQ | https://arxiv.org/abs/2409.04313 | Direct overlap on using censored regression labels in drug discovery; the paper adapts ensemble/Bayesian/Gaussian models with Tobit-style survival analysis. | CANAL must differentiate through public ChEMBL assay/document/source provenance, strict scaffold/source/temporal splits, repeated-measurement noise diagnostics, and hybrid anchor-plus-censor objectives. | High for any broad "censored labels in drug discovery" claim; medium if CANAL claims assay/source-aware observation modeling and v2 diagnostics. |
| DrugOOD | https://github.com/tencent-ailab/DrugOOD | OOD benchmark for AI-aided drug discovery. | CANAL builds source/assay/scaffold/temporal OOD on bioactivity records and evaluates censored/noisy activity calibration. | Low: useful benchmark signal, not same task. |
| TDC | https://tdcommons.ai/ | Public therapeutic ML tasks and official splits. | CANAL uses TDC only as a boundary experiment when assay metadata are absent. | Low. |
| Chemprop | https://github.com/chemprop/chemprop | Strong D-MPNN molecular baseline. | CANAL can use Chemprop as a backbone but changes the observation model and likelihood. | Low. |
| Uni-Mol | https://github.com/dptech-corp/Uni-Mol | 3D molecular representation baseline. | Representation baseline only; CANAL contribution is metadata/censoring observation model. | Low. |
| MolFormer | https://github.com/IBM/molformer | Transformer molecular representation baseline. | Representation baseline only; CANAL contribution is observation channel and reliability. | Low. |

## Reviewer Questions to Answer

1. Is assay metadata just an extra feature? Run a strong baseline that feeds assay/source IDs directly into the predictor.
2. Does censored likelihood matter? Compare against midpoint/threshold imputation and exact-only training.
3. Does hierarchy matter? Remove assay/source shrinkage and test cold-start assays/sources.
4. Does reliability posterior mean anything? Evaluate against repeated-measure disagreement, cross-source disagreement, and ChEMBL validity flags.
5. Does the method preserve normal prediction accuracy? Report exact-test MAE/RMSE/Spearman alongside NLL/ECE/coverage.

## Initial Decision

Proceed with ChEMBL v35 smoke protocol first. If censored/source/noise handling does not move NLL, coverage, or worst-source metrics on representative high-sample targets, downgrade to a data-audit/reliability benchmark rather than forcing the CANAL claim.

## 2026-08-10 Claim Adjustment

After checking the censored-label UQ near neighbor, CANAL should not claim novelty for applying Tobit/censored likelihood to drug-discovery regression. The defensible novelty is the combination of:

1. ChEMBL-scale assay/document/source-aware observation modeling.
2. Strict OOD protocols that separate scaffold, assay/source, temporal, and joint scaffold-source axes.
3. Empirical repeated-measurement and validity-flag diagnostics showing public bioactivity noise.
4. A hybrid censored likelihood plus anchor objective that trades off point MAE and censored-label calibration.
