# Evaluation Metrics & Protocol

This section is written as a practical evaluation plan for an HPC security team.

## 1) Offline sanity checks (no ground-truth labels required)
Even without labeled attacks, you can validate the system by tracking:

- **Alert rate**: alerts/day and alerts/user/day (target a manageable number for SOC)
- **Stability over time**: does the risk ranking remain stable for normal periods?
- **Drift signals**: does the average anomaly score drift after policy changes or upgrades?

## 2) Labeled evaluation (recommended)
If KAUST/KSL can provide a limited set of labeled incidents (even a few), report:

- AUROC and PR-AUC (for event-level anomaly)
- Precision@K (top K users/day)
- Mean Time To Detect (MTTD) under simulated scenarios

## 3) Injection-based evaluation (safe simulation)
Create controlled injections to mimic common threats:

### A) Credential misuse patterns
- high rate of SSH failures from a new IP bucket
- login at unusual hours compared to user history
- rapid open/close sessions (automation)

### B) Scheduler abuse patterns
- job submit bursts (dozens per minute)
- rare partitions/GRES usage for a user
- unusual job runtime distributions (too long / too short)

### C) Job telemetry outliers (sacct)
- extreme read/write bytes compared to user baseline
- unusual memory profile (MaxRSS)
- abnormal CPU time vs. elapsed time

Measure:
- detection rate of injected anomalies (hit rate)
- false positives on “clean” windows
- operator workload (alerts/day)

## 4) Calibration targets (practical)
- Start with `alert_quantile=0.98` and adjust:
  - if too noisy: increase to 0.99
  - if too quiet: decrease to 0.95
- Adjust fusion weights:
  - identity-heavy threat model: higher `seq_weight`
  - allocation abuse risk: higher `job_weight`
