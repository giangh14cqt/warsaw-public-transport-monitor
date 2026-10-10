# 02_error_handling.md: Circuit Breakers and Resilience Standards

## 1. Deduplication
- Persist telemetry updates only when the state tuple `(trip_id, stop_sequence, rt_arrival_time)` has mutated relative to the in-memory cache.
- Discard duplicate stationary observations to save storage and prevent biased downstream regression weights.

## 2. Circuit Breaker Strategy
- **Transient Network Hiccups**: Apply exponential backoff with full jitter on network timeouts and transient HTTP 5xx errors ($3\text{s}, 6\text{s}, 12\text{s}, 24\text{s}$ up to max 60s).
- **HTTP 429 (Rate Limiting)**: If the upstream endpoint (`zbiorkom.live`) returns HTTP 429, trigger an immediate circuit-breaker pause for 5 minutes (`300s`) before resuming polling.
- **Persistent Outages**: If continuous connection failures exceed 15 minutes, trigger an alert / dead-man's switch ping and log a critical failure.
- **Protobuf Schema Incompatibilities**: If `DecodeError` occurs while parsing GTFS-RT feed bytes, immediately stop polling to prevent corrupt state ingestion and raise a fatal alarm.

## 3. Remote Self-Host Daemon Stability
- Trapped Signals: Trap `SIGINT` and `SIGTERM` in collector daemons to guarantee a final in-memory buffer flush to disk before container or process termination.
- Health Monitoring: Maintain a rotating heartbeat log (`logs/collector_health.log`) reporting cycle duration, batch sizes, and error counts.
