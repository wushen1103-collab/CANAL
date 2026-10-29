# Communications Chemistry revision snapshot

This directory fixes the CANAL source used for the revised Communications Chemistry
manuscript, including ChEMBL 35 preprocessing, leakage-controlled split construction,
model training, evaluation, statistics, and figure generation.

- The directory root matches `Supplementary_Software_1.zip`.
- `revision_audit/` matches `Supplementary_Software_2_revision_audit.tar.gz` and adds
  the final scripts used to render the reviewer-requested affine-frontier and
  threshold-audit figures.
- The primary experiment host path after migration is
  `/home/wsk/projects/33_CANAL`; historical paths retained in logs identify the same
  migrated project.
- Raw ChEMBL databases and derived model artifacts are not redistributed here.

The Git tag `communications-chemistry-r1` identifies this manuscript-specific
snapshot. The commit hash recorded by Git is the stable code identifier cited in the
revised manuscript.
