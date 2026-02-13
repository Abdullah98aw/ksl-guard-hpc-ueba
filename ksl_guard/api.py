from fastapi import FastAPI
from pydantic import BaseModel
import os
import tempfile
import subprocess
import json

app = FastAPI(title="KSL-Guard API", version="0.1")

class ScoreRequest(BaseModel):
    sshd_path: str = ""
    slurmctld_path: str = ""
    sacct_path: str = ""
    model_dir: str
    out_path: str = ""

@app.post("/score")
def score(req: ScoreRequest):
    out = req.out_path or os.path.join(tempfile.gettempdir(), "ksl_guard_alerts.json")
    cmd = [
        "python", "-m", "ksl_guard.score",
        "--sshd", req.sshd_path,
        "--slurmctld", req.slurmctld_path,
        "--sacct", req.sacct_path,
        "--model", req.model_dir,
        "--out", out
    ]
    # Filter out empty args to avoid CLI issues
    cmd2 = []
    for i, c in enumerate(cmd):
        if c == "" and i > 0 and cmd[i-1].startswith("--"):
            continue
        cmd2.append(c)
    subprocess.check_call(cmd2)
    with open(out, "r", encoding="utf-8") as f:
        return json.load(f)
