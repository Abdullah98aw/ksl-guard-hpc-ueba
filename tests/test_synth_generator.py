import os
from ksl_guard.synth.generator import SynthConfig, generate

def test_generate_small(tmp_path):
    out = tmp_path / "synth"
    cfg = SynthConfig(out_dir=str(out), days=1, n_users=10, target_ssh_events=2000, target_jobs=500, seed=1)
    paths = generate(cfg)
    for k, p in paths.items():
        assert os.path.exists(p), f"missing {k}: {p}"
