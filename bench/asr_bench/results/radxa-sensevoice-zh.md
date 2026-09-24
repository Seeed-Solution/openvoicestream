# ASR bench: sensevoice / zh

- Target: `ws://100.77.150.16:8622`
- Metric: final latency = audio-end -> `is_final` received (ms); RTF = final_latency / audio_duration

| Concurrency | Segments | OK | Errors | p50 latency (ms) | p95 latency (ms) | RTF p50 | RTF p95 | Throughput (seg/s) | Mean error rate |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 20 | 20 | 0 | 1292.3 | 1542.7 | 0.216 | 0.464 | 0.14 | 0.0599 |
| 2 | 20 | 1 | 19 | 1276.0 | 1276.0 | 0.213 | 0.213 | 0.14 | 0.0667 |
| 4 | 20 | 1 | 19 | 1424.5 | 1424.5 | 0.449 | 0.449 | 0.21 | 0.0000 |
| 8 | 20 | 1 | 19 | 1279.1 | 1279.1 | 0.496 | 0.496 | 0.26 | 0.1111 |
