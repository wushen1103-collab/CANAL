# Resource Policy

User constraint: use at most half of the machine.

Remote machine snapshot:

- GPUs: 8x RTX 4090D, so CANAL may use at most 4 GPUs concurrently.
- CPUs: 384 logical cores, so CANAL may use at most 192 CPU threads concurrently.

Operational policy:

- Smoke and data-build stages use CPU only, normally `CANAL_CPU_THREADS=32`.
- Single-model deep learning stages start on the least-busy GPU only.
- Full parallel stages may expand up to 4 GPUs only after checking `nvidia-smi`.
- Leave the machine responsive: do not launch jobs that exceed 192 total CPU threads, and prefer 32 to 64 threads per CPU-heavy job.
- Never generate plots during experiment execution; tables and logs only.
