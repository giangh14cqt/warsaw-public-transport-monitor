# Warsaw Transit Telemetry & Delay Attribution System

## Operational Infrastructure & Deployment Context

> **IMPORTANT**: The live transit delay telemetry ingestion pipeline is **actively running in production on a remote self-hosted server** and backed up daily to Google Drive.
> For private, host-specific paths and remote credentials on this workstation, refer to `LOCAL_DEPLOYMENT.md` and `.env` (both git-ignored).

### 1. Remote Self-Host Server
- **Host Environment**: Remote Linux server running Docker Compose or native Python daemon. (Specific server paths and credentials are documented in git-ignored `LOCAL_DEPLOYMENT.md`).
- **Daemon Process**: Continuous ingestion daemon running via Docker Compose (`container_name: warsaw_telemetry_daemon`) or `python3 collector_daemon.py`.
- **Ingestion Target**: GTFS-RT Warsaw endpoint (`https://cdn.zbiorkom.live/gtfs-rt/warsaw.pb`) polled every 30 seconds.
- **Dead Man's Switch**: Sends 15-minute periodic heartbeats and failure alerts to [Healthchecks.io](https://healthchecks.io) (`HEARTBEAT_URL` in `.env`).
- **Code Stability Constraint**: Any refactoring or code pushed to `main` must maintain 100% backward compatibility with `collector_daemon.py`, `run_preflight.py`, `Dockerfile`, `docker-compose.yml`, and `src/` root module proxies (`src/fetcher.py`, `src/gtfs_manager.py`, `src/storage.py`) so remote `git pull` operations never disrupt the running daemon.

### 2. Daily Google Drive Backup
- **Tool**: `rclone`
- **Configured Remote**: `${RCLONE_REMOTE:-gdrive}` (configured in `.env` or `LOCAL_DEPLOYMENT.md`)
- **Destination**: `${RCLONE_REMOTE}:${DEST_FOLDER}` (default: `gdrive:WarsawDelayTelemetry/data/raw`)
- **Source**: `data/raw/` (Hive-partitioned Apache Parquet partitions: `year=YYYY/month=MM/day=DD/batch_*.parquet`)
- **Automated Schedule**: Cron job running daily at 03:00 UTC:
  ```cron
  0 3 * * * /path/to/warsaw-public-transport-monitor/backup_to_gdrive.sh > /dev/null 2>&1
  ```
- **Backup Script**: [`backup_to_gdrive.sh`](backup_to_gdrive.sh)
- **Audit Logs**: `logs/backup.log`

### 3. Local Analytics & Google Drive Pull Workflow
- **Purpose**: When performing local analysis, feature mart fusion, or training models on your workstation, pull the latest Parquet partitions down from Google Drive without touching the remote server.
- **Pull Script**: [`pull_from_gdrive.sh`](pull_from_gdrive.sh)
- **Usage**:
  ```bash
  ./pull_from_gdrive.sh
  ```
- **Sync Behavior**: Runs `rclone copy` with `--update`, fetching only new/updated daily partitions into `data/raw/` while preserving existing local files.
- **Audit Logs**: `logs/pull_gdrive.log`

### 4. Architecture & Data Flow
1. **Ingestion**: `src/ingestion/` (GTFS-RT protobuf poller + weekly static GTFS archiver + partitioned Parquet writer).
2. **Exogenous**: `src/exogenous/` (IMGW-PIB weather telemetry + OSMnx road corridor topology).
3. **Fusion**: `src/fusion/` (trajectory delay decomposition + multi-source feature mart assembly).
4. **Modeling**: `src/models/` (purged temporal block splitting + TWFE panel regression + LightGBM/CatBoost).
5. **xAI**: `src/xai/` (TreeSHAP global/local attribution + ALE threshold detection + DiCE counterfactuals).

### 5. Project Tracking & GitHub Projects Integration
- **Active Project Board**: [GitHub Project #1](https://github.com/users/giangh14cqt/projects/1)
- **Repository**: [`giangh14cqt/warsaw-public-transport-monitor`](https://github.com/giangh14cqt/warsaw-public-transport-monitor)
- **GitHub Milestones**: [Milestones View](https://github.com/giangh14cqt/warsaw-public-transport-monitor/milestones)
  - **M1: Ingestion Engine** (Issues #1, #2, #3)
  - **M2: Exogenous Feature Mart** (Issues #4, #5)
  - **M3: Merged Feature Store** (Issues #6, #7)
  - **M4: Benchmark Models POC** (Issues #8, #9, #10)
  - **M5: xAI Diagnostics POC** (Issues #11, #12, #13, #14)
  - **M6: Final Deliverables & POC Report** (Issues #15, #16)
- **Task Delivery Protocol**:
  - All engineering tasks, subtasks, equations, and acceptance criteria are tracked directly in the GitHub Issues above.
  - When completing work in new sessions, reference the relevant issue number (`#N`), verify its acceptance criteria, and update the issue status on GitHub Project 1.
