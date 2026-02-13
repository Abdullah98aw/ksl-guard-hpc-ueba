from typing import Tuple
import pandas as pd

from .parsing import parse_sshd_lines, parse_slurmctld_lines


def _read_text(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()


def ingest(sshd_path: str = "", slurmctld_path: str = "", sacct_path: str = "") -> Tuple[pd.DataFrame, pd.DataFrame]:
    ssh_events = parse_sshd_lines(_read_text(sshd_path)) if sshd_path else []
    sl_events = parse_slurmctld_lines(_read_text(slurmctld_path)) if slurmctld_path else []

    events = pd.DataFrame(ssh_events + sl_events)
    if not events.empty:
        events["ts"] = pd.to_datetime(events["ts"], errors="coerce", utc=True)
        events = events.sort_values("ts")

    sacct = pd.read_csv(sacct_path) if sacct_path else pd.DataFrame()
    return events, sacct
