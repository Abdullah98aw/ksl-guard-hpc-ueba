import argparse
import pickle

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .config import ModelConfig
from .ingest import ingest
from .features import build_event_tokens, fit_vocab, encode_tokens, make_sequences, job_features
from .models import NextTokenLSTM, loss_fn
from .utils import set_seed, ensure_dir, save_json


def train_sequence_model(X, y, vocab_size: int, cfg: ModelConfig, device: str):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    model = NextTokenLSTM(vocab_size, cfg.emb_dim, cfg.hidden_dim, cfg.n_layers, cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    model.train()
    for ep in range(cfg.epochs):
        total = 0.0
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = model(bx)
            loss = loss_fn(logits, by)
            loss.backward()
            opt.step()
            total += loss.item() * bx.size(0)
        print(f"[seq] epoch {ep+1}/{cfg.epochs} loss={total/len(ds):.4f}")
    return model


def train_job_model(job_df: pd.DataFrame):
    if job_df is None or job_df.empty:
        return None

    feats = job_df.drop(columns=[c for c in ["user", "jobid"] if c in job_df.columns], errors="ignore")
    scaler = StandardScaler()
    X = scaler.fit_transform(feats.values)

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)

    return {"model": model, "scaler": scaler, "cols": list(feats.columns)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sshd", type=str, default="")
    p.add_argument("--slurmctld", type=str, default="")
    p.add_argument("--sacct", type=str, default="")
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    cfg = ModelConfig()
    set_seed(cfg.seed)
    ensure_dir(args.out)

    events, sacct = ingest(args.sshd, args.slurmctld, args.sacct)

    tokens, _users = build_event_tokens(events)
    vocab = fit_vocab(tokens, max_vocab=cfg.max_vocab)
    token_ids = encode_tokens(tokens, vocab)
    X, y = make_sequences(token_ids, seq_len=cfg.seq_len)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_model = train_sequence_model(X, y, vocab_size=len(vocab), cfg=cfg, device=device)

    torch.save(seq_model.state_dict(), f"{args.out}/seq_model.pt")
    with open(f"{args.out}/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    job_df = job_features(sacct)
    job_pack = train_job_model(job_df)
    with open(f"{args.out}/job_model.pkl", "wb") as f:
        pickle.dump(job_pack, f)

    save_json(f"{args.out}/meta.json", {
        "seq_len": cfg.seq_len,
        "vocab_size": len(vocab),
        "emb_dim": cfg.emb_dim,
        "hidden_dim": cfg.hidden_dim,
        "n_layers": cfg.n_layers,
        "dropout": cfg.dropout,
        "device_trained": device,
    })

    print(f"Saved model artifacts to: {args.out}")


if __name__ == "__main__":
    main()
