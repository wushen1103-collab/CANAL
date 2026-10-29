# Data provenance and redistribution

The experiments derive bioactivity records from named ChEMBL releases and use public
TDC datasets for auxiliary representation checks. Upstream database contents are not
licensed by this repository and are not included in Git.

Users should obtain each upstream release from its official distributor, retain the
release identifier and download checksum, and comply with the upstream terms. The data
construction scripts preserve activity relations, parent-molecule identifiers, assay
and document provenance, and the frozen split assignments used by the benchmark.

For an archival release, publish compact row identifiers, split assignments, manifests,
configuration files, and checksums when permitted. Large raw databases, processed
feature caches, model checkpoints, and prediction exports should remain in external
storage unless their redistribution terms have been checked separately.
