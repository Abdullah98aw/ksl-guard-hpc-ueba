import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(prog="ksl-guard", description="KSL-Guard helper CLI")
    sp = p.add_subparsers(dest="cmd", required=True)

    t = sp.add_parser("train", help="Train models")
    t.add_argument("--sshd", type=str, default="")
    t.add_argument("--slurmctld", type=str, default="")
    t.add_argument("--sacct", type=str, default="")
    t.add_argument("--out", type=str, required=True)

    s = sp.add_parser("score", help="Score + produce alerts")
    s.add_argument("--sshd", type=str, default="")
    s.add_argument("--slurmctld", type=str, default="")
    s.add_argument("--sacct", type=str, default="")
    s.add_argument("--model", type=str, required=True)
    s.add_argument("--out", type=str, required=True)

    args = p.parse_args()

    if args.cmd == "train":
        subprocess.check_call([sys.executable, "-m", "ksl_guard.train",
                               "--sshd", args.sshd,
                               "--slurmctld", args.slurmctld,
                               "--sacct", args.sacct,
                               "--out", args.out])
    elif args.cmd == "score":
        subprocess.check_call([sys.executable, "-m", "ksl_guard.score",
                               "--sshd", args.sshd,
                               "--slurmctld", args.slurmctld,
                               "--sacct", args.sacct,
                               "--model", args.model,
                               "--out", args.out])


if __name__ == "__main__":
    main()
