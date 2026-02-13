# Limitations (Clear & Honest)

KSL‑Guard is a research prototype. The following limitations are expected and should be stated explicitly.

1) **Log format variability**
HPC sites differ in SSH/Slurm log formats and field naming. Parsers in this repo are best‑effort and need local adaptation.

2) **No guarantee of ground truth**
Self‑supervised anomaly detection learns “normal” from data. If the training window contains attacks or abnormal operations, the model may treat them as normal.

3) **False positives**
Behavior changes (new projects, conferences, deadlines, new partitions) can create spikes. Threshold calibration and allow-lists are required.

4) **Limited explainability**
The system provides scores and some feature hints, but deep explanations require additional instrumentation (template mining, attribution, enrichment).

5) **Not a full IDS**
KSL‑Guard is identity + scheduler focused. It does not replace network IDS, endpoint telemetry, or file-integrity monitoring.

6) **Privacy constraints**
Using fine-grained IP/user mapping may increase privacy risk. Default implementation uses coarse IP bucketing and recommends pseudonymization.
