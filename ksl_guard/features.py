from __future__ import annotations

import hashlib
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _coarse_ip(ip: str) -> str:
    # privacy-friendly IP bucketing
    if not ip or "." not in ip:
        return ""
    parts = ip.split(".")
    return ".".join(parts[:2])  # /16-ish bucket


def build_event_tokens(events: pd.DataFrame) -> Tuple[List[str], List[str]]:
    if events is None or events.empty:
        return [], []

    tokens: List[str] = []
    users: List[str] = []
    for _, r in events.iterrows():
        src = str(r.get("src", "na"))
        etype = str(r.get("etype", "na"))
        user = str(r.get("user", "na"))
        ipb = _coarse_ip(str(r.get("ip", "")))

        # Token captures type + coarse source info (privacy-aware)
        tok = f"{src}|{etype}|{ipb}"
        tokens.append(tok)
        users.append(user)

    return tokens, users


def fit_vocab(tokens: List[str], max_vocab: int = 8000) -> Dict[str, int]:
    c = Counter(tokens)
    most = c.most_common(max_vocab - 2)
    vocab: Dict[str, int] = {"[PAD]": 0, "[UNK]": 1}
    for i, (t, _) in enumerate(most, start=2):
        vocab[t] = i
    return vocab


def encode_tokens(tokens: List[str], vocab: Dict[str, int]) -> List[int]:
    return [vocab.get(t, vocab["[UNK]"]) for t in tokens]


def make_sequences(token_ids: List[int], seq_len: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(token_ids) - seq_len):
        X.append(token_ids[i : i + seq_len])
        y.append(token_ids[i + seq_len])
    return np.asarray(X, dtype=np.int64), np.asarray(y, dtype=np.int64)


def job_features(sacct: pd.DataFrame) -> pd.DataFrame:
    # Expect best-effort `sacct` export CSV. Column names vary across sites.
    if sacct is None or sacct.empty:
        return pd.DataFrame()

    df = sacct.copy()

    def pick(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    c_user = pick("User", "UserID", "Account", "Uid")
    c_job = pick("JobID", "JobId", "JobIDRaw", "JobIDRaw_")
    c_elapsed = pick("ElapsedRaw", "ElapsedSeconds", "Elapsed", "ElapsedRawSeconds")
    c_cpu = pick("TotalCPU", "CPUTimeRAW", "CPUTimeRaw", "CPUTime")
    c_ncpu = pick("NCPUS", "AllocCPUS", "ReqCPUS")
    c_rss = pick("MaxRSS", "AveRSS", "MaxRSSBytes", "AveRSSBytes")
    c_read = pick("Read", "ReadBytes", "AveRead", "AveReadBytes")
    c_write = pick("Write", "WriteBytes", "AveWrite", "AveWriteBytes")

    keep = [c for c in [c_user, c_job, c_elapsed, c_cpu, c_ncpu, c_rss, c_read, c_write] if c]
    df = df[keep].rename(columns={
        c_user: "user",
        c_job: "jobid",
        c_elapsed: "elapsed",
        c_cpu: "cpu",
        c_ncpu: "ncpu",
        c_rss: "rss",
        c_read: "read",
        c_write: "write",
    })

    # Coerce numerics
    for c in ["elapsed", "cpu", "ncpu", "rss", "read", "write"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df
