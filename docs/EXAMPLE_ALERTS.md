# Example Alerts (JSON)

This is an example of what `outputs/alerts.json` looks like.

```json
{
  "alerts": [
    {
      "user": "bob",
      "risk": 0.612,
      "risk_seq": 0.743,
      "risk_job": 0.213,
      "seq_alert_events": 8,
      "seq_threshold": 0.935
    }
  ],
  "seq_events": [
    {
      "ts": "2026-02-12 11:20:00+00:00",
      "user": "bob",
      "seq_anom": 0.981,
      "seq_alert": true
    }
  ],
  "job_events": [
    {
      "user": "dave",
      "jobid": 77777,
      "job_anom": 0.997
    }
  ]
}
```

## How a SOC might use it
- **Risk list**: start with the highest risk users.
- **Seq anomalies**: identify suspicious authentication bursts, rare sources, or unusual sequences.
- **Job anomalies**: spot abnormal I/O or runtime behavior that may indicate abuse or exfil signals.
