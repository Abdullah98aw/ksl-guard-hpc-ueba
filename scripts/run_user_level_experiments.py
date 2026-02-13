#!/usr/bin/env python3
"""
Lightweight, conference-friendly user-level evaluation for KSL-Guard (synthetic dataset).

- No heavy ML training (works on modest laptops).
- Uses pandas/numpy only.
- Produces:
  - results/user_level_metrics.csv
  - results/user_rankings_<model>.csv
  - results/user_features.csv

Evaluation unit: USER-level ranking (UEBA style).
Ground truth: a user is "malicious" if they have >= N injected anomalous rows across SSH + jobs.

Models (lightweight):
1) rule_based
2) zscore_ssh_only
3) zscore_job_only
4) fusion_zscore (SSH + job)

Run:
  python scripts/run_user_level_experiments.py --data-dir data/synth_small
"""

import argparse
import os
import math
import numpy as np
import pandas as pd


def safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def zscore_series(s: pd.Series) -> pd.Series:
    # standard z-score with safe guards
    s = s.astype(float)
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def robust_zscore_series(s: pd.Series) -> pd.Series:
    # robust z-score using median and MAD (more stable with heavy tails)
    s = s.astype(float)
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return 0.6745 * (s - med) / mad


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    k = min(k, len(scores))
    order = np.argsort(-scores)[:k]
    return float(y_true[order].sum()) / float(k) if k > 0 else 0.0


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    positives = float(y_true.sum())
    if positives == 0:
        return 0.0
    k = min(k, len(scores))
    order = np.argsort(-scores)[:k]
    return float(y_true[order].sum()) / positives


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/synth_small", help="Path to generated synthetic dataset folder")
    p.add_argument("--out-dir", default="results", help="Where to write outputs")
    p.add_argument("--malicious-min-anoms", type=int, default=5, help="Ground truth: user malicious if anomalies >= this threshold")
    p.add_argument("--alert-quantile", type=float, default=0.98, help="Alert threshold as quantile over user risk scores")
    p.add_argument("--w-ssh", type=float, default=0.5, help="Fusion weight for SSH score")
    p.add_argument("--w-job", type=float, default=0.5, help="Fusion weight for Job score")
    args = p.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir
    safe_mkdir(out_dir)

    ssh_path = os.path.join(data_dir, "ssh_events.csv")
    jobs_path = os.path.join(data_dir, "sacct_jobs.csv")
    meta_path = os.path.join(data_dir, "dataset_meta.json")

    if not os.path.exists(ssh_path) or not os.path.exists(jobs_path):
        raise FileNotFoundError(f"Missing files in {data_dir}. Expected ssh_events.csv and sacct_jobs.csv")

    ssh = pd.read_csv(ssh_path)
    jobs = pd.read_csv(jobs_path)

    # --- Basic parsing ---
    ssh["ts_utc"] = pd.to_datetime(ssh["ts_utc"], utc=True, errors="coerce")
    jobs["submit_ts_utc"] = pd.to_datetime(jobs["submit_ts_utc"], utc=True, errors="coerce")

    # --- USER LIST ---
    users = sorted(set(ssh["user"].dropna().astype(str).unique()).union(set(jobs["user"].dropna().astype(str).unique())))
    uf = pd.DataFrame({"user": users}).set_index("user")

    # --- Ground truth per user (from injected anomalies) ---
    ssh_anoms = ssh.groupby("user")["is_anomaly"].sum().rename("ssh_anoms")
    job_anoms = jobs.groupby("user")["is_anomaly"].sum().rename("job_anoms")
    uf = uf.join(ssh_anoms, how="left").join(job_anoms, how="left").fillna(0)
    uf["total_anoms"] = uf["ssh_anoms"] + uf["job_anoms"]
    uf["y_true"] = (uf["total_anoms"] >= args.malicious_min_anoms).astype(int)

    # --- SSH FEATURES (lightweight UEBA) ---
    ssh["hour"] = ssh["ts_utc"].dt.hour.fillna(0).astype(int)
    ssh["is_offhours"] = ((ssh["hour"] >= 0) & (ssh["hour"] <= 5)).astype(int)

    ssh_fail = (ssh["event_type"] == "ssh_fail").astype(int)
    ssh_accept = (ssh["event_type"] == "ssh_accept").astype(int)

    ssh_feat = pd.DataFrame({
        "ssh_events": ssh.groupby("user").size(),
        "ssh_fail": ssh.groupby("user")[ssh_fail.name].sum() if ssh_fail.name in ssh.columns else ssh.groupby("user").apply(lambda g: (g["event_type"]=="ssh_fail").sum()),
    })

    # The above line may not work depending on pandas temp name; do it robustly:
    ssh_feat["ssh_fail"] = ssh.groupby("user").apply(lambda g: int((g["event_type"] == "ssh_fail").sum()))
    ssh_feat["ssh_accept"] = ssh.groupby("user").apply(lambda g: int((g["event_type"] == "ssh_accept").sum()))
    ssh_feat["ssh_offhours"] = ssh.groupby("user")["is_offhours"].sum()
    ssh_feat["ssh_unique_ipb"] = ssh.groupby("user")["ip_bucket"].nunique()

    ssh_feat["ssh_fail_rate"] = ssh_feat["ssh_fail"] / ssh_feat["ssh_events"].clip(lower=1)

    uf = uf.join(ssh_feat, how="left").fillna(0)

    # --- JOB FEATURES (from sacct_jobs) ---
    jobs["is_bad_state"] = jobs["state"].isin(["FAILED", "CANCELLED"]).astype(int)
    jobs["io_total"] = jobs["read_bytes"].fillna(0) + jobs["write_bytes"].fillna(0)

    job_feat = pd.DataFrame({
        "job_count": jobs.groupby("user").size(),
        "job_bad_state": jobs.groupby("user")["is_bad_state"].sum(),
        "io_total_sum": jobs.groupby("user")["io_total"].sum(),
        "io_total_mean": jobs.groupby("user")["io_total"].mean(),
        "cpu_sec_sum": jobs.groupby("user")["cpu_sec"].sum(),
        "elapsed_sec_sum": jobs.groupby("user")["elapsed_sec"].sum(),
        "max_rss_mb_mean": jobs.groupby("user")["max_rss_mb"].mean(),
    }).fillna(0)

    job_feat["job_bad_rate"] = job_feat["job_bad_state"] / job_feat["job_count"].clip(lower=1)
    job_feat["cpu_util_proxy"] = job_feat["cpu_sec_sum"] / job_feat["elapsed_sec_sum"].clip(lower=1)  # proxy
    uf = uf.join(job_feat, how="left").fillna(0)

    # --- Scoring (models) ---
    # 1) rule_based: simple weighted sum of suspicious signals
    # Normalize by robust z to avoid heavy tails dominating.
    uf["rb_score"] = (
        0.35 * robust_zscore_series(uf["ssh_fail_rate"]) +
        0.25 * robust_zscore_series(uf["ssh_unique_ipb"]) +
        0.15 * robust_zscore_series(uf["ssh_offhours"]) +
        0.15 * robust_zscore_series(uf["job_bad_rate"]) +
        0.10 * robust_zscore_series(np.log1p(uf["io_total_sum"]))
    ).fillna(0)

    # 2) zscore_ssh_only
    uf["ssh_score"] = (
        0.45 * robust_zscore_series(uf["ssh_fail_rate"]) +
        0.25 * robust_zscore_series(uf["ssh_unique_ipb"]) +
        0.30 * robust_zscore_series(uf["ssh_offhours"])
    ).fillna(0)

    # 3) zscore_job_only
    uf["job_score"] = (
        0.35 * robust_zscore_series(uf["job_bad_rate"]) +
        0.35 * robust_zscore_series(np.log1p(uf["io_total_sum"])) +
        0.20 * robust_zscore_series(uf["cpu_util_proxy"]) +
        0.10 * robust_zscore_series(uf["max_rss_mb_mean"])
    ).fillna(0)

    # 4) fusion
    w_ssh = float(args.w_ssh)
    w_job = float(args.w_job)
    if abs((w_ssh + w_job) - 1.0) > 1e-6:
        # normalize automatically
        s = w_ssh + w_job
        w_ssh, w_job = w_ssh / s, w_job / s

    uf["fusion_score"] = (w_ssh * uf["ssh_score"] + w_job * uf["job_score"]).fillna(0)

    # --- Alerts/day estimate (based on quantile threshold over users) ---
    # Approximate days from data by reading SSH timestamps range (fallback: 1)
    days = 1.0
    if ssh["ts_utc"].notna().any():
        tmin = ssh["ts_utc"].min()
        tmax = ssh["ts_utc"].max()
        span = (tmax - tmin).total_seconds() / (3600 * 24)
        days = max(1.0, span)

    def alerts_per_day(scores: pd.Series) -> float:
        thr = scores.quantile(args.alert_quantile)
        alerts = float((scores >= thr).sum())
        return alerts / days

    models = [
        ("rule_based", "rb_score"),
        ("zscore_ssh_only", "ssh_score"),
        ("zscore_job_only", "job_score"),
        ("fusion_zscore", "fusion_score"),
    ]

    # --- Metrics ---
    y_true = uf["y_true"].to_numpy().astype(int)

    rows = []
    for model_name, col in models:
        scores = uf[col].to_numpy().astype(float)
        rows.append({
            "model": model_name,
            "malicious_min_anoms": args.malicious_min_anoms,
            "users_total": int(len(uf)),
            "users_positive": int(y_true.sum()),
            "precision_at_5": precision_at_k(y_true, scores, 5),
            "precision_at_10": precision_at_k(y_true, scores, 10),
            "recall_at_5": recall_at_k(y_true, scores, 5),
            "recall_at_10": recall_at_k(y_true, scores, 10),
            "alerts_per_day_est": alerts_per_day(uf[col]),
            "alert_quantile": args.alert_quantile,
            "w_ssh": w_ssh if model_name == "fusion_zscore" else "",
            "w_job": w_job if model_name == "fusion_zscore" else "",
        })

        # rankings output
        rank_out = uf.reset_index()[["user", "y_true", "total_anoms", col]].sort_values(col, ascending=False)
        rank_out.rename(columns={col: "score"}, inplace=True)
        rank_out.to_csv(os.path.join(out_dir, f"user_rankings_{model_name}.csv"), index=False)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(out_dir, "user_level_metrics.csv"), index=False)

    # Features output (useful for paper appendix / reproducibility)
    feat_cols = [
        "y_true","total_anoms",
        "ssh_events","ssh_fail","ssh_accept","ssh_offhours","ssh_unique_ipb","ssh_fail_rate",
        "job_count","job_bad_state","job_bad_rate","io_total_sum","io_total_mean","cpu_util_proxy","max_rss_mb_mean",
        "rb_score","ssh_score","job_score","fusion_score",
    ]
    uf.reset_index()[["user"] + feat_cols].to_csv(os.path.join(out_dir, "user_features.csv"), index=False)

    print("DONE.")
    print(f"Wrote: {os.path.join(out_dir, 'user_level_metrics.csv')}")
    print("Models:", ", ".join([m[0] for m in models]))


if __name__ == "__main__":
    main()