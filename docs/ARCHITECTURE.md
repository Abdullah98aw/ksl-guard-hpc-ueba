# Architecture Diagram

This document shows how KSL‑Guard fits into an HPC environment (batch-first, streaming-ready).

## High-level architecture (Mermaid)
```mermaid
flowchart LR
  subgraph Sources[Telemetry Sources]
    A1[SSH auth.log / secure] 
    A2[Slurm slurmctld.log]
    A3[sacct CSV export]
  end

  subgraph Ingest[Ingestion + Normalization]
    B1[Parsing & normalization\n(ksl_guard/parsing.py)]
    B2[Event store / feature table\n(DataFrames, Parquet optional)]
  end

  subgraph Models[ML Models]
    C1[Self-supervised sequence model\nNext-token LSTM]
    C2[Job outlier model\nIsolation Forest]
  end

  subgraph Risk[Risk & Alerting]
    D1[Fusion: per-user risk score\n(seq + job)]
    D2[Alert rules\n(quantile threshold + ranking)]
    D3[JSON alerts\n(outputs/alerts.json)]
  end

  subgraph Integrations[Integrations]
    E1[SIEM/SOAR\n(Splunk/Elastic/etc.)]
    E2[Ticketing\n(Jira/ServiceNow)]
    E3[Dashboards\n(Kibana/Grafana)]
  end

  Sources --> B1 --> B2
  B2 --> C1
  B2 --> C2
  C1 --> D1
  C2 --> D1
  D1 --> D2 --> D3
  D3 --> E1
  D3 --> E2
  D3 --> E3
```

## Notes
- Batch mode is the simplest: export `sacct` daily and process rotated logs.
- Streaming mode can be added later by tailing logs into a queue and scoring via API.
