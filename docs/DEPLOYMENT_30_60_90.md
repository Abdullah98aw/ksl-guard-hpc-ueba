# 30‑60‑90 Day Deployment Plan (HPC / KSL)

This plan assumes KSL has log access and a small security + HPC admin collaboration.

## Days 0–30: Discovery + Data Readiness
**Goal:** run the prototype on real logs in a safe, offline manner.

- Identify log sources:
  - SSH auth logs (login nodes)
  - slurmctld logs
  - sacct export fields available
- Define minimal governance:
  - where raw logs live
  - retention window for prototype
  - access controls
- Build parsing adapters:
  - adjust regex in `ksl_guard/parsing.py` for the local log formats
- First baseline run:
  - train on a “known normal” period (e.g., 7–14 days)
  - generate daily alerts and review with operators

Deliverables:
- baseline alert rate & sample incidents
- updated parsers for KSL formats
- documented threshold choice

## Days 31–60: Pilot + Operator Workflow
**Goal:** integrate with an operator feedback loop and reduce noise.

- Add enrichment (optional):
  - map user -> project/account/partition (if available)
  - tag known admin/service accounts
- Calibrate:
  - tune `alert_quantile`
  - tune fusion weights
- Create a lightweight workflow:
  - alert JSON -> SIEM index or shared dashboard
  - ticket template for top-risk users
- Run a safe injection test:
  - simulated bursts, rare sources, abnormal sacct jobs
  - measure detection + false positives

Deliverables:
- pilot KPI report (alerts/day, precision@K, top causes)
- operator playbook for triage
- recommended governance for ongoing use

## Days 61–90: Productionization Options
**Goal:** choose the right path: batch production or near-real-time.

Option A (batch production):
- nightly ETL
- daily scoring + SIEM alerts

Option B (near-real-time):
- tail logs -> queue -> API scoring
- rate-limited alerting to avoid flooding

Hardening checklist:
- model retraining cadence (weekly/monthly)
- monitoring for drift
- versioned artifacts + rollback
- privacy review: hashed IDs and IP bucketing defaults

Deliverables:
- production design decision + architecture
- integration PRs / deployment manifests (site-specific)
