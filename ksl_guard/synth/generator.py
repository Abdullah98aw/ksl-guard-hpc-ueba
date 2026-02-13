"""
KSL-Guard Synthetic Dataset Generator
====================================

Generates reproducible synthetic HPC telemetry for research evaluation:
- SSH auth/session events (ssh_events.csv)
- SLURM controller-style job lifecycle events (slurm_events.csv)
- SLURM accounting-style job telemetry (sacct_jobs.csv)
- Ground-truth manifest for injected anomalies (injection_manifest.csv)

Design goals:
- Reproducible via seed
- Privacy-safe identifiers (synthetic users, IP buckets)
- Parameterized distributions to emulate HPC usage heterogeneity
- Explicit anomaly injection (credential misuse, job floods, IO spikes)

IMPORTANT:
This generator produces *synthetic* data. Do not represent the outputs as real KAUST logs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
import math
import random
import csv
import os


@dataclass
class SynthConfig:
    # Time horizon
    start_utc: str = "2026-01-01T00:00:00Z"
    days: int = 30

    # Population
    n_users: int = 500
    n_ip_buckets: int = 200  # e.g., /24 buckets represented as "ipb-###"
    n_partitions: int = 6
    n_accounts: int = 25

    # Activity scale (approx)
    target_ssh_events: int = 1_200_000
    target_jobs: int = 240_000

    # Heterogeneity / skew (Zipf-like)
    user_activity_skew: float = 1.15  # higher -> more heavy-tail

    # Anomaly injection
    anomaly_rate: float = 0.02  # fraction of events/jobs tagged as anomalous via injection
    anomaly_seed_offset: int = 1337

    # Output
    out_dir: str = "out_synth"
    seed: int = 42


def _parse_iso_z(s: str) -> datetime:
    # expects '...Z'
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def _zipf_weights(n: int, s: float) -> List[float]:
    # p(k) ∝ 1/k^s for k=1..n
    weights = [1.0 / (k ** s) for k in range(1, n + 1)]
    total = sum(weights)
    return [w / total for w in weights]


def _choice_weighted(rng: random.Random, items: List[str], weights: List[float]) -> str:
    # small helper; relies on rng.random
    x = rng.random()
    cum = 0.0
    for item, w in zip(items, weights):
        cum += w
        if x <= cum:
            return item
    return items[-1]


def _rand_ts_in_window(rng: random.Random, start: datetime, end: datetime) -> datetime:
    span = (end - start).total_seconds()
    return start + timedelta(seconds=rng.random() * span)


def _lognormal(rng: random.Random, mu: float, sigma: float) -> float:
    # return lognormal sample
    return math.exp(rng.normalvariate(mu, sigma))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _build_vocab(config: SynthConfig) -> Dict[str, List[str]]:
    users = [f"user{idx:04d}" for idx in range(1, config.n_users + 1)]
    ipb = [f"ipb-{idx:03d}" for idx in range(1, config.n_ip_buckets + 1)]
    partitions = [f"part{idx}" for idx in range(1, config.n_partitions + 1)]
    accounts = [f"acct{idx:02d}" for idx in range(1, config.n_accounts + 1)]
    return {"users": users, "ipb": ipb, "partitions": partitions, "accounts": accounts}


def generate(config: SynthConfig) -> Dict[str, str]:
    """
    Generate dataset files and return file paths.
    """
    rng = random.Random(config.seed)
    vocab = _build_vocab(config)

    start = _parse_iso_z(config.start_utc)
    end = start + timedelta(days=config.days)

    os.makedirs(config.out_dir, exist_ok=True)

    # User activity weights (Zipf)
    user_weights = _zipf_weights(config.n_users, config.user_activity_skew)

    # --- SSH events ---
    # We simulate: ssh_fail, ssh_accept, ssh_session_open, ssh_session_close
    ssh_event_types = ["ssh_fail", "ssh_accept", "ssh_session_open", "ssh_session_close"]

    ssh_path = os.path.join(config.out_dir, "ssh_events.csv")
    with open(ssh_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts_utc", "user", "ip_bucket", "event_type", "auth_method", "is_anomaly", "anomaly_type", "anomaly_id"])

        # Baseline generation
        for i in range(config.target_ssh_events):
            user = _choice_weighted(rng, vocab["users"], user_weights)
            ipb = vocab["ipb"][rng.randrange(len(vocab["ipb"]))]
            ts = _rand_ts_in_window(rng, start, end)

            # baseline behavior: mostly accepts, some fails
            p_fail = 0.08 + 0.10 * (rng.random() < 0.02)  # a tiny bit bursty sometimes
            if rng.random() < p_fail:
                et = "ssh_fail"
            else:
                # accepted sessions have open/close nearby, but we keep single-row event; open/close handled separately
                et = "ssh_accept" if rng.random() < 0.6 else "ssh_session_open"

            auth_method = "pubkey" if rng.random() < 0.75 else "password"
            w.writerow([ts.isoformat().replace("+00:00", "Z"), user, ipb, et, auth_method, 0, "", ""])

    # --- SLURM job records (accounting) + controller-style events ---
    # We create jobs with IDs and then derive lifecycle events.
    jobs_path = os.path.join(config.out_dir, "sacct_jobs.csv")
    slurm_events_path = os.path.join(config.out_dir, "slurm_events.csv")

    with open(jobs_path, "w", newline="", encoding="utf-8") as f_jobs, open(slurm_events_path, "w", newline="", encoding="utf-8") as f_ev:
        jw = csv.writer(f_jobs)
        ew = csv.writer(f_ev)

        jw.writerow(["job_id", "user", "account", "partition", "submit_ts_utc", "start_ts_utc", "end_ts_utc",
                     "state", "nnodes", "ncpus", "elapsed_sec", "cpu_sec", "max_rss_mb",
                     "read_bytes", "write_bytes", "exit_code", "is_anomaly", "anomaly_type", "anomaly_id"])
        ew.writerow(["ts_utc", "job_id", "user", "event_type", "partition", "account", "is_anomaly", "anomaly_type", "anomaly_id"])

        for j in range(1, config.target_jobs + 1):
            user = _choice_weighted(rng, vocab["users"], user_weights)
            account = vocab["accounts"][rng.randrange(len(vocab["accounts"]))]
            partition = vocab["partitions"][rng.randrange(len(vocab["partitions"]))]

            submit_ts = _rand_ts_in_window(rng, start, end)
            # queue wait: lognormal with small mean; clamp to 0..6h
            wait_sec = _clamp(_lognormal(rng, mu=5.0, sigma=1.0), 0.0, 6 * 3600.0)
            start_ts = submit_ts + timedelta(seconds=wait_sec)

            # runtime: heavy-tail (lognormal) with cap; seconds
            runtime_sec = _clamp(_lognormal(rng, mu=7.0, sigma=1.2), 30.0, 72 * 3600.0)
            end_ts = start_ts + timedelta(seconds=runtime_sec)

            # resources: correlate with partition a bit
            base_cpus = 8 if partition in ("part1", "part2") else 32
            ncpus = int(_clamp(_lognormal(rng, mu=math.log(base_cpus), sigma=0.6), 1, 2048))
            nnodes = int(_clamp(_lognormal(rng, mu=math.log(max(1, ncpus // 32)), sigma=0.7), 1, 256))

            elapsed_sec = int(runtime_sec)
            cpu_sec = int(elapsed_sec * ncpus * _clamp(rng.normalvariate(0.65, 0.20), 0.02, 1.2))
            max_rss_mb = int(_clamp(_lognormal(rng, mu=math.log(1024), sigma=0.9) * (1 + ncpus / 256), 128, 1024*512))

            # I/O: lognormal, weakly correlated with runtime + nodes
            read_bytes = int(_clamp(_lognormal(rng, mu=18.0, sigma=1.1) * (1 + nnodes/16), 0, 10**14))
            write_bytes = int(_clamp(_lognormal(rng, mu=17.5, sigma=1.1) * (1 + nnodes/16), 0, 10**14))

            # state: mostly COMPLETED
            state = "COMPLETED" if rng.random() < 0.93 else ("CANCELLED" if rng.random() < 0.6 else "FAILED")
            exit_code = "0:0" if state == "COMPLETED" else ("0:9" if state == "CANCELLED" else "1:0")

            # write job
            jw.writerow([j, user, account, partition,
                         submit_ts.isoformat().replace("+00:00", "Z"),
                         start_ts.isoformat().replace("+00:00", "Z"),
                         end_ts.isoformat().replace("+00:00", "Z"),
                         state, nnodes, ncpus, elapsed_sec, cpu_sec, max_rss_mb,
                         read_bytes, write_bytes, exit_code, 0, "", ""])

            # derive controller events
            ew.writerow([submit_ts.isoformat().replace("+00:00", "Z"), j, user, "job_submit", partition, account, 0, "", ""])
            ew.writerow([start_ts.isoformat().replace("+00:00", "Z"), j, user, "job_start", partition, account, 0, "", ""])
            ew.writerow([end_ts.isoformat().replace("+00:00", "Z"), j, user, "job_end", partition, account, 0, "", ""])

    # --- Inject anomalies ---
    # We do post-processing: mark certain rows in ssh_events and jobs as anomalies, and emit a manifest.
    manifest_path = os.path.join(config.out_dir, "injection_manifest.csv")

    # Load CSVs
    import pandas as pd  # local import to keep top light
    ssh_df = pd.read_csv(ssh_path)
    jobs_df = pd.read_csv(jobs_path)
    slurm_df = pd.read_csv(slurm_events_path)

    rng2 = random.Random(config.seed + config.anomaly_seed_offset)

    anomalies = []
    anomaly_id = 0

    # Helper: mark rows by indices
    def mark(df, idxs, a_type):
        nonlocal anomaly_id
        for idx in idxs:
            anomaly_id += 1
            df.loc[idx, "is_anomaly"] = 1
            df.loc[idx, "anomaly_type"] = a_type
            df.loc[idx, "anomaly_id"] = anomaly_id
            anomalies.append({"anomaly_id": anomaly_id, "anomaly_type": a_type, "target": "row", "row_index": int(idx)})

    # A1: Credential misuse / SSH burst from rare IP bucket for a victim user
    n_bursts = max(5, int(config.n_users * 0.01))
    for _ in range(n_bursts):
        victim = vocab["users"][rng2.randrange(len(vocab["users"]))]
        rare_ipb = f"ipb-{config.n_ip_buckets + rng2.randrange(50):03d}"  # outside normal set -> rare
        # pick a time window
        t0 = _rand_ts_in_window(rng2, start, end - timedelta(hours=1))
        # select a subset of rows for victim near t0; if none, pick random victim rows
        victim_rows = ssh_df.index[ssh_df["user"] == victim].tolist()
        if not victim_rows:
            continue
        # pick 80 rows and set as fails then accept
        chosen = [victim_rows[rng2.randrange(len(victim_rows))] for _ in range(80)]
        for i, idx in enumerate(chosen):
            ssh_df.loc[idx, "ts_utc"] = (t0 + timedelta(seconds=i*20)).isoformat().replace("+00:00", "Z")
            ssh_df.loc[idx, "ip_bucket"] = rare_ipb
            ssh_df.loc[idx, "event_type"] = "ssh_fail" if i < 70 else "ssh_accept"
            ssh_df.loc[idx, "auth_method"] = "password"
        mark(ssh_df, chosen, "ssh_credential_misuse_burst")

    # A2: Job submission flood (many submits in short window for one user)
    n_floods = max(3, int(config.n_users * 0.006))
    for _ in range(n_floods):
        attacker = vocab["users"][rng2.randrange(len(vocab["users"]))]
        attacker_job_idxs = jobs_df.index[jobs_df["user"] == attacker].tolist()
        if len(attacker_job_idxs) < 200:
            continue
        t0 = _rand_ts_in_window(rng2, start, end - timedelta(hours=2))
        chosen = attacker_job_idxs[:200]  # deterministic slice for reproducibility
        for i, idx in enumerate(chosen):
            # compress submit times into 10 minutes
            submit = t0 + timedelta(seconds=(i % 600))
            start_ts = submit + timedelta(seconds=rng2.randrange(60, 600))
            end_ts = start_ts + timedelta(seconds=rng2.randrange(30, 900))
            jobs_df.loc[idx, "submit_ts_utc"] = submit.isoformat().replace("+00:00", "Z")
            jobs_df.loc[idx, "start_ts_utc"] = start_ts.isoformat().replace("+00:00", "Z")
            jobs_df.loc[idx, "end_ts_utc"] = end_ts.isoformat().replace("+00:00", "Z")
            jobs_df.loc[idx, "state"] = "FAILED" if rng2.random() < 0.3 else "CANCELLED"
            jobs_df.loc[idx, "exit_code"] = "1:0" if jobs_df.loc[idx, "state"] == "FAILED" else "0:9"
        mark(jobs_df, chosen, "slurm_job_submission_flood")

    # A3: Abnormal I/O exfiltration-like jobs (huge read/write)
    n_io = max(200, int(config.target_jobs * 0.002))
    io_candidates = jobs_df.sample(n=min(n_io*2, len(jobs_df)), random_state=config.seed).index.tolist()
    chosen_io = io_candidates[:n_io]
    for idx in chosen_io:
        jobs_df.loc[idx, "read_bytes"] = int(5e13 + rng2.random()*5e13)
        jobs_df.loc[idx, "write_bytes"] = int(2e13 + rng2.random()*4e13)
        jobs_df.loc[idx, "elapsed_sec"] = int(_clamp(jobs_df.loc[idx, "elapsed_sec"], 60, 7200))
    mark(jobs_df, chosen_io, "sacct_io_spike")

    # A4: Off-hours access pattern (rare time-of-day for a user)
    n_off = max(1000, int(config.target_ssh_events * 0.003))
    off_idxs = ssh_df.sample(n=min(n_off, len(ssh_df)), random_state=config.seed+7).index.tolist()
    # push times to 03:00-04:00 UTC
    for i, idx in enumerate(off_idxs):
        base = _rand_ts_in_window(rng2, start, end)
        forced = base.replace(hour=3, minute=rng2.randrange(0,60), second=rng2.randrange(0,60), microsecond=0)
        ssh_df.loc[idx, "ts_utc"] = forced.isoformat().replace("+00:00", "Z")
    mark(ssh_df, off_idxs, "ssh_offhours_pattern")

    # Propagate job anomalies to slurm events for same job_ids
    job_anom_ids = set(jobs_df.loc[jobs_df["is_anomaly"] == 1, "job_id"].astype(int).tolist())
    slurm_df["is_anomaly"] = 0
    slurm_df["anomaly_type"] = ""
    slurm_df["anomaly_id"] = ""
    slurm_anom_idxs = slurm_df.index[slurm_df["job_id"].isin(job_anom_ids)].tolist()
    # Mark these slurm events as anomalies (same type "job_linked")
    if slurm_anom_idxs:
        for idx in slurm_anom_idxs:
            slurm_df.loc[idx, "is_anomaly"] = 1
            slurm_df.loc[idx, "anomaly_type"] = "job_linked_anomaly"
            slurm_df.loc[idx, "anomaly_id"] = 0

    # Save back
    ssh_df.to_csv(ssh_path, index=False)
    jobs_df.to_csv(jobs_path, index=False)
    slurm_df.to_csv(slurm_events_path, index=False)

    # Manifest
    mf = pd.DataFrame(anomalies)
    mf.to_csv(manifest_path, index=False)

    # Metadata file
    meta_path = os.path.join(config.out_dir, "dataset_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"config": asdict(config),
                   "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
                   "counts": {
                       "ssh_events": int(len(ssh_df)),
                       "slurm_events": int(len(slurm_df)),
                       "jobs": int(len(jobs_df)),
                       "manifest": int(len(mf)),
                       "job_anomalies": int((jobs_df["is_anomaly"]==1).sum()),
                       "ssh_anomalies": int((ssh_df["is_anomaly"]==1).sum())
                   }}, f, indent=2)

    return {
        "ssh_events": ssh_path,
        "slurm_events": slurm_events_path,
        "sacct_jobs": jobs_path,
        "manifest": manifest_path,
        "meta": meta_path
    }
