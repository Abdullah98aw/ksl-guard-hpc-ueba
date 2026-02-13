# Synthetic Dataset (for Reproducible Conference-Grade Evaluation)

This folder describes the **synthetic HPC telemetry** used to evaluate KSL-Guard **without claiming access to real KAUST logs**.

## What it generates

Running the generator produces:

- `ssh_events.csv`  
  Columns: `ts_utc,user,ip_bucket,event_type,auth_method,is_anomaly,anomaly_type,anomaly_id`

- `slurm_events.csv`  
  Columns: `ts_utc,job_id,user,event_type,partition,account,is_anomaly,anomaly_type,anomaly_id`

- `sacct_jobs.csv`  
  Columns include core job fields and job telemetry (elapsed, cpu, rss, io, etc.) plus anomaly labels.

- `injection_manifest.csv`  
  Row-level ground-truth for injected anomalies (type and ID).

- `dataset_meta.json`  
  Config + counts + generation time.

## Injection scenarios (ground truth)

- `ssh_credential_misuse_burst`: many SSH failures then success from a rare IP bucket
- `slurm_job_submission_flood`: many job submissions in a short time window
- `sacct_io_spike`: extreme I/O jobs (read/write bytes) simulating staging/exfil behavior
- `ssh_offhours_pattern`: off-hours access pattern shift

## How to run

From the repository root:

```bash
python scripts/generate_synth_dataset.py --out-dir data/synth --days 30 --n-users 500 --seed 42 --anomaly-rate 0.02
```

## Notes for the paper

- Clearly label results as **simulation-based**.
- Report mean ± CI over multiple random seeds.
- Do NOT describe this dataset as real institutional logs.
