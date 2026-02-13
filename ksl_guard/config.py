from dataclasses import dataclass

@dataclass
class ModelConfig:
    seed: int = 42
    max_vocab: int = 8000
    seq_len: int = 32
    emb_dim: int = 64
    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 256
    epochs: int = 6
    lr: float = 1e-3

@dataclass
class RiskConfig:
    # Fusion weights
    seq_weight: float = 0.65
    job_weight: float = 0.35

    # Risk windows
    user_window_events: int = 250
    user_window_jobs: int = 50

    # Event alert threshold: percentile of seq anomaly scores
    alert_quantile: float = 0.98
