# Deployment Guide — KSL-Guard (HPC UEBA)

This guide explains how to deploy KSL-Guard in a KAUST-like HPC environment (e.g., SLURM-based clusters) using a low-risk, passive approach.

## 1) Data Sources
- SSH authentication logs (e.g., /var/log/auth.log or /var/log/secure)
- SLURM controller logs (e.g., slurmctld.log)
- SLURM accounting exports (sacct CSV)

## 2) Recommended Operating Mode (Batch)
- Export `sacct` daily (or hourly for higher sensitivity)
- Collect SSH + SLURM logs from central log storage
- Run scoring daily/hourly
- Forward alerts JSON to SIEM

## 3) Minimal Integration Steps
1. Place KSL-Guard on a monitoring host (no scheduler modification required)
2. Configure log export paths
3. Run the scoring pipeline on schedule (cron / task scheduler)
4. Ship alerts to SIEM (Splunk / Elastic / Sentinel)

## 4) Operational Safety
- Do not block jobs or logins in Phase 1
- Use shadow-mode for 30 days
- Calibrate alert thresholds to SOC capacity

## 5) Recommended Rollout
- Days 0–30: Shadow monitoring
- Days 31–60: SOC integration & tuning
- Days 61–90: Production operation + drift monitoring
