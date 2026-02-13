import re
from datetime import datetime
from typing import Dict, List, Optional

from dateutil import parser as dtparser


# --- SSH parsing (best-effort) ---
# Examples seen in typical Linux auth.log / secure files.
RE_ACCEPT = re.compile(r"Accepted (\w+) for (\S+) from (\S+) port (\d+)")
RE_FAIL = re.compile(r"Failed (\w+) for (invalid user )?(\S+) from (\S+) port (\d+)")
RE_SESSION_OPEN = re.compile(r"pam_unix\(sshd:session\): session opened for user (\S+)")
RE_SESSION_CLOSE = re.compile(r"pam_unix\(sshd:session\): session closed for user (\S+)")


def _parse_ts_prefix(line: str) -> Optional[datetime]:
    # Many Linux logs start with: "Jan 12 10:11:12 host ..."
    try:
        prefix = " ".join(line.split()[:3])
        return dtparser.parse(prefix)
    except Exception:
        return None


def parse_sshd_lines(lines: List[str]) -> List[Dict]:
    events: List[Dict] = []
    for ln in lines:
        ts = _parse_ts_prefix(ln) or datetime.utcnow()

        m = RE_ACCEPT.search(ln)
        if m:
            method, user, ip, port = m.group(1), m.group(2), m.group(3), int(m.group(4))
            events.append({
                "ts": ts.isoformat(),
                "src": "sshd",
                "etype": "ssh_accept",
                "user": user,
                "ip": ip,
                "port": port,
                "method": method,
            })
            continue

        m = RE_FAIL.search(ln)
        if m:
            method = m.group(1)
            user = m.group(3)
            ip = m.group(4)
            port = int(m.group(5))
            events.append({
                "ts": ts.isoformat(),
                "src": "sshd",
                "etype": "ssh_fail",
                "user": user,
                "ip": ip,
                "port": port,
                "method": method,
            })
            continue

        m = RE_SESSION_OPEN.search(ln)
        if m:
            events.append({
                "ts": ts.isoformat(),
                "src": "sshd",
                "etype": "ssh_session_open",
                "user": m.group(1),
            })
            continue

        m = RE_SESSION_CLOSE.search(ln)
        if m:
            events.append({
                "ts": ts.isoformat(),
                "src": "sshd",
                "etype": "ssh_session_close",
                "user": m.group(1),
            })
            continue

    return events


# --- Slurm parsing (generic, not site-specific) ---
# Slurm log formats vary. These regexes catch common patterns.
RE_JOB_SUBMIT = re.compile(r"job_submit\W+JobId=(\d+)\W+UserId=([^\s]+)")
RE_JOB_START = re.compile(r"job_start\W+JobId=(\d+)")
RE_JOB_END = re.compile(r"job_complete\W+JobId=(\d+)\W+WEXITSTATUS=(\d+)")


def parse_slurmctld_lines(lines: List[str]) -> List[Dict]:
    events: List[Dict] = []
    for ln in lines:
        ts = _parse_ts_prefix(ln) or datetime.utcnow()

        m = RE_JOB_SUBMIT.search(ln)
        if m:
            jobid = int(m.group(1))
            user = m.group(2).split("(")[0]
            events.append({
                "ts": ts.isoformat(),
                "src": "slurmctld",
                "etype": "job_submit",
                "user": user,
                "jobid": jobid,
            })
            continue

        m = RE_JOB_START.search(ln)
        if m:
            events.append({
                "ts": ts.isoformat(),
                "src": "slurmctld",
                "etype": "job_start",
                "jobid": int(m.group(1)),
            })
            continue

        m = RE_JOB_END.search(ln)
        if m:
            events.append({
                "ts": ts.isoformat(),
                "src": "slurmctld",
                "etype": "job_end",
                "jobid": int(m.group(1)),
                "exit": int(m.group(2)),
            })
            continue

    return events
