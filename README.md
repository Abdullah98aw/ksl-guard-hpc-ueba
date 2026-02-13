## Reproducible Synthetic Evaluation (Conference Paper)

To support conference-grade evaluation without relying on restricted institutional logs, this repository includes a **synthetic HPC telemetry generator** with explicit anomaly injection and ground-truth labels.

Generate 30 days of telemetry:

```bash
python scripts/generate_synth_dataset.py --out-dir data/synth --days 30 --n-users 500 --seed 42 --anomaly-rate 0.02
```

Outputs are written to `data/synth/` including SSH events, SLURM lifecycle events, sacct-style job telemetry, and an injection manifest.

See `docs/synthetic_dataset.md` for schema and injection definitions.

## Reproducible Synthetic Evaluation

Generate 30 days of HPC telemetry:

python scripts/generate_synth_dataset.py --out-dir data/synth --days 30 --n-users 500 --seed 42
