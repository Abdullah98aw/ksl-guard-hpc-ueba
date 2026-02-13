import argparse
import pickle
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import ModelConfig, RiskConfig
from .ingest import ingest
from .features import build_event_tokens, encode_tokens, make_sequences, job_features
from .models import NextTokenLSTM
from .utils import load_json, save_json


def _seq_anomaly_scores(model: NextTokenLSTM, X: np.ndarray, y: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        bx = torch.tensor(X).to(device)
        logits = model(bx)
        probs = F.softmax(logits, dim=1).cpu().numpy()

    true_prob = probs[np.arange(len(y)), y]
    return 1.0 - true_prob  # higher means more anomalous


def _job_anomaly_scores(job_pack: Dict[str, Any], job_df: pd.DataFrame) -> pd.DataFrame:
    if not job_pack or job_df is None or job_df.empty:
        return pd.DataFrame(columns=["user", "jobid", "job_anom"])

    cols = job_pack["cols"]
    feats = job_df[cols] if cols else job_df.drop(columns=["user", "jobid"], errors="ignore")

    X = job_pack["scaler"].transform(feats.values)
    # IsolationForest score_samples: higher = more normal
    normality = job_pack["model"].score_samples(X)
    a = (normality.max() - normality)
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)

    out = job_df[["user", "jobid"]].copy()
    out["job_anom"] = a
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sshd", type=str, default="")
    p.add_argument("--slurmctld", type=str, default="")
    p.add_argument("--sacct", type=str, default="")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    mcfg = ModelConfig()
    rcfg = RiskConfig()

    meta = load_json(f"{args.model}/meta.json")

    with open(f"{args.model}/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_model = NextTokenLSTM(
        vocab_size=meta["vocab_size"],
        emb_dim=meta["emb_dim"],
        hidden_dim=meta["hidden_dim"],
        n_layers=meta["n_layers"],
        dropout=meta["dropout"],
    ).to(device)
    seq_model.load_state_dict(torch.load(f"{args.model}/seq_model.pt", map_location=device))

    with open(f"{args.model}/job_model.pkl", "rb") as f:
        job_pack = pickle.load(f)

    events, sacct = ingest(args.sshd, args.slurmctld, args.sacct)

    # Sequence scoring
    tokens, users = build_event_tokens(events)
    token_ids = encode_tokens(tokens, vocab)

    X, y = make_sequences(token_ids, seq_len=meta["seq_len"])
    if len(y) == 0:
        save_json(args.out, {"alerts": [], "seq_events": [], "job_events": []})
        print(f"Wrote empty output to {args.out} (not enough events)")
        return

    seq_scores = _seq_anomaly_scores(seq_model, X, y, device=device)

    seq_df = pd.DataFrame({
        "ts": events["ts"].iloc[meta["seq_len"]:].astype(str).values,
        "user": users[meta["seq_len"] :],
        "seq_anom": seq_scores.tolist(),
    })

    thr = float(np.quantile(seq_df["seq_anom"].values, rcfg.alert_quantile))
    seq_df["seq_alert"] = (seq_df["seq_anom"] >= thr)

    # Job scoring
    job_df = job_features(sacct)
    job_scored = _job_anomaly_scores(job_pack, job_df)

    # Per-user risk fusion
    alerts: List[Dict[str, Any]] = []
    for user, g in seq_df.groupby("user"):
        risk_seq = float(g["seq_anom"].tail(rcfg.user_window_events).mean())
        jg = job_scored[job_scored["user"] == user] if not job_scored.empty else pd.DataFrame()
        risk_job = float(jg["job_anom"].tail(rcfg.user_window_jobs).mean()) if not jg.empty else 0.0

        risk = rcfg.seq_weight * risk_seq + rcfg.job_weight * risk_job

        alerts.append({
            "user": user,
            "risk": risk,
            "risk_seq": risk_seq,
            "risk_job": risk_job,
            "seq_alert_events": int(g["seq_alert"].sum()),
            "seq_threshold": thr,
        })

    payload = {
        "alerts": sorted(alerts, key=lambda x: x["risk"], reverse=True),
        "seq_events": seq_df.to_dict(orient="records"),
        "job_events": job_scored.to_dict(orient="records"),
    }
    save_json(args.out, payload)
    print(f"Wrote alerts to {args.out}")


if __name__ == "__main__":
    main()
