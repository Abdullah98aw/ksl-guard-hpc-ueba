#!/usr/bin/env python3
"""
CLI wrapper for the synthetic dataset generator.

Example:
  python scripts/generate_synth_dataset.py --out-dir data/synth --days 30 --n-users 500 --seed 42 --anomaly-rate 0.02
"""
from __future__ import annotations
import argparse
from ksl_guard.synth.generator import SynthConfig, generate

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/synth", help="Output directory for generated CSV/JSON files")
    p.add_argument("--start-utc", default="2026-01-01T00:00:00Z", help="Start timestamp (ISO 8601, Z)")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--n-users", type=int, default=500)
    p.add_argument("--target-ssh-events", type=int, default=1200000)
    p.add_argument("--target-jobs", type=int, default=240000)
    p.add_argument("--user-activity-skew", type=float, default=1.15)
    p.add_argument("--anomaly-rate", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = SynthConfig(
        start_utc=args.start_utc,
        days=args.days,
        n_users=args.n_users,
        target_ssh_events=args.target_ssh_events,
        target_jobs=args.target_jobs,
        user_activity_skew=args.user_activity_skew,
        anomaly_rate=args.anomaly_rate,
        out_dir=args.out_dir,
        seed=args.seed,
    )
    paths = generate(cfg)
    print("Generated:")
    for k, v in paths.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
