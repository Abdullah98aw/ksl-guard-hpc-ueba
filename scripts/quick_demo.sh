#!/usr/bin/env bash
set -euo pipefail

python scripts/make_sample_data.py --out_dir data
python -m ksl_guard.train --sshd data/sample_auth.log --slurmctld data/sample_slurmctld.log --sacct data/sample_sacct.csv --out outputs/model
python -m ksl_guard.score --sshd data/sample_auth.log --slurmctld data/sample_slurmctld.log --sacct data/sample_sacct.csv --model outputs/model --out outputs/alerts.json

echo "Done. Output: outputs/alerts.json"
