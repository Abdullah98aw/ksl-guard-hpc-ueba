import argparse
import os
import random
from datetime import datetime, timedelta

import pandas as pd


def _ts(dt: datetime) -> str:
    return dt.strftime("%b %d %H:%M:%S")


def make_auth_log(path: str, days: int = 2):
    random.seed(7)
    users = ["alice", "bob", "carol", "dave"]
    normal_ips = {
        "alice": ["10.2.5.11", "10.2.5.12"],
        "bob": ["10.2.7.20"],
        "carol": ["10.2.9.8"],
        "dave": ["10.2.3.44"],
    }

    start = datetime.utcnow() - timedelta(days=days)
    lines = []

    # normal behavior
    for i in range(800):
        u = random.choice(users)
        ip = random.choice(normal_ips[u])
        dt = start + timedelta(minutes=i * 3)
        if random.random() < 0.85:
            lines.append(f"{_ts(dt)} login sshd[1234]: Accepted publickey for {u} from {ip} port {random.randint(2200,65000)} ssh2")
            if random.random() < 0.60:
                lines.append(f"{_ts(dt)} login sshd[1234]: pam_unix(sshd:session): session opened for user {u} by (uid=0)")
        else:
            lines.append(f"{_ts(dt)} login sshd[1234]: Failed password for {u} from {ip} port {random.randint(2200,65000)} ssh2")

    # injected anomaly: bob from rare IP with many failures
    dt = start + timedelta(hours=6)
    for j in range(25):
        lines.append(f"{_ts(dt + timedelta(minutes=j))} login sshd[9999]: Failed password for bob from 203.0.113.77 port {random.randint(2200,65000)} ssh2")
    lines.append(f"{_ts(dt + timedelta(minutes=30))} login sshd[9999]: Accepted password for bob from 203.0.113.77 port 40222 ssh2")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def make_slurmctld_log(path: str):
    random.seed(11)
    users = ["alice", "bob", "carol", "dave"]
    start = datetime.utcnow() - timedelta(days=1)
    lines = []
    jobid = 10000

    for i in range(400):
        u = random.choice(users)
        dt = start + timedelta(minutes=i * 4)
        jid = jobid + i
        lines.append(f"{_ts(dt)} controller slurmctld: sched: job_submit JobId={jid} UserId={u}(100{i%10})")
        if random.random() < 0.9:
            lines.append(f"{_ts(dt + timedelta(seconds=25))} controller slurmctld: sched: job_start JobId={jid}")
            lines.append(f"{_ts(dt + timedelta(minutes=random.randint(2,120)))} controller slurmctld: sched: job_complete JobId={jid} WEXITSTATUS=0")
        else:
            lines.append(f"{_ts(dt + timedelta(minutes=1))} controller slurmctld: sched: job_complete JobId={jid} WEXITSTATUS=1")

    # injected anomaly: many rapid submit events (script abuse)
    dt = start + timedelta(hours=8)
    for k in range(35):
        jid = jobid + 900 + k
        lines.append(f"{_ts(dt + timedelta(seconds=k))} controller slurmctld: sched: job_submit JobId={jid} UserId=carol(1002)")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def make_sacct_csv(path: str):
    random.seed(13)
    rows = []
    users = ["alice", "bob", "carol", "dave"]

    for i in range(500):
        u = random.choice(users)
        jid = 10000 + i
        # normal ranges
        elapsed = random.randint(60, 7200)
        cpu = elapsed * random.uniform(0.2, 1.8)
        ncpu = random.choice([1, 2, 4, 8, 16])
        rss = random.uniform(0.5, 48.0) * 1024  # MB-ish
        read = random.uniform(0, 5e9)
        write = random.uniform(0, 2e9)

        rows.append({
            "User": u,
            "JobID": jid,
            "ElapsedRaw": elapsed,
            "TotalCPU": cpu,
            "NCPUS": ncpu,
            "MaxRSS": rss,
            "ReadBytes": read,
            "WriteBytes": write,
        })

    # injected anomaly: extremely high I/O job by dave
    rows.append({
        "User": "dave",
        "JobID": 77777,
        "ElapsedRaw": 5400,
        "TotalCPU": 2000,
        "NCPUS": 8,
        "MaxRSS": 1024,
        "ReadBytes": 8e12,
        "WriteBytes": 6e12,
    })

    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    make_auth_log(os.path.join(args.out_dir, "sample_auth.log"))
    make_slurmctld_log(os.path.join(args.out_dir, "sample_slurmctld.log"))
    make_sacct_csv(os.path.join(args.out_dir, "sample_sacct.csv"))

    print(f"Sample data written to {args.out_dir}")


if __name__ == "__main__":
    main()
