# Results — User-Level UEBA Evaluation

**Dataset configuration (June 2026).**

Evaluation was conducted using a calibrated synthetic HPC telemetry dataset consisting of:

- 50 users  
- 3 days of activity  
- 50,000 SSH events  
- 10,000 SLURM jobs  
- Anomaly budget controlled via `anomaly_rate = 0.01`

Under the operational definition of a malicious user as one with **≥ 20 injected anomalous rows**, a total of **16 out of 50 users** were labeled positive (32%).

---

## Evaluated Models

Four approaches were compared:

1. Rule-based baseline  
2. SSH-only anomaly scoring  
3. Job-only anomaly scoring  
4. Fusion-based scoring (SSH + Job telemetry)

Evaluation reflects SOC-style user prioritization (UEBA ranking).

---

## Quantitative Results

| Model              | Precision@5 | Precision@10 | Recall@10 | Alerts/Day |
|--------------------|------------|-------------|-----------|------------|
| Rule-based         | 1.00       | 1.00        | 0.625     | 0.33       |
| SSH-only           | 1.00       | 1.00        | 0.625     | 0.33       |
| Job-only           | 1.00       | 0.70        | 0.4375    | 0.33       |
| Fusion (0.5/0.5)   | 1.00       | 1.00        | 0.625     | 0.33       |

---

## Interpretation

The fusion model achieves:

- **Precision@5 = 1.00**
- **Precision@10 = 1.00**
- **Recall@10 = 0.625**

This indicates that selecting the top 10 highest-risk users successfully identifies 62.5% of malicious users while maintaining a low operational alert rate (~0.33 alerts/day at the 0.98 alert quantile).

The job-only model exhibits reduced Precision@10 (0.70) and Recall@10 (0.4375), suggesting that identity-layer SSH behavior remains a critical signal for early-stage user-level prioritization.

These findings reinforce the importance of combining authentication-layer and scheduler-layer telemetry in HPC-focused UEBA systems.

---

# Reproducibility

All experiments are fully reproducible using the commands documented in the repository:

- Synthetic dataset generation  
- User-level evaluation pipeline  

Repository:  
🔗 <span style="color:blue">https://github.com/Abdullah98aw/ksl-guard-hpc-ueba</span>

Synthetic dataset documentation:  
🔗 <span style="color:blue">docs/synthetic_dataset.md</span>

Evaluation script:  
🔗 <span style="color:blue">scripts/run_user_level_experiments.py</span>

---

## Production Integration Notes

KSL-Guard is designed for low-risk operational deployment in HPC environments:

- **Passive monitoring**: no modifications to SLURM or SSH are required.
- **Batch-first**: recommended daily/hourly scoring using exported logs and `sacct` CSV.
- **SIEM-ready output**: produces structured JSON alerts suitable for Splunk / Elastic / Sentinel pipelines.
- **Privacy-aware**: supports pseudonymized user identifiers and feature-only retention policies.
- **Scalable**: can run on a separate monitoring host and scale with log volume via partitioned processing.

---

# Notes

- The dataset is fully synthetic and privacy-safe.
- Results reflect controlled injection scenarios.
- Further sensitivity analysis (weight tuning and multiple seeds) can be conducted for extended evaluation.

---
