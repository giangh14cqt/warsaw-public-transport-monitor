# 04_remote_infrastructure.md: Remote Server & Cloud Backup Operations

## 1. Remote Production Server Context
- **Deployment**: Live ingestion runs 24/7 on a remote self-hosted server running Docker Compose or native Python daemon. (Specific server paths and credentials are documented in git-ignored `LOCAL_DEPLOYMENT.md`).
- **Runtime**: Runs as a daemon container (`container_name: warsaw_telemetry_daemon`) via Docker Compose or native Python systemd/tmux service.
- **Monitoring**: Heartbeat pings sent every 15 minutes to Healthchecks.io via `HEARTBEAT_URL` in `.env`.
- **Golden Rule**: Never push breaking structural or import changes to `main` that would cause `collector_daemon.py`, `run_preflight.py`, `Dockerfile`, or `docker-compose.yml` to fail when `git pull` is run on the server. Always maintain backwards-compatible import shims in `src/`.

## 2. Daily Google Drive Incremental Backups
- **Synchronization Tool**: `rclone copy` (copies only new/modified Parquet files; never deletes files).
- **Remote Profile**: Configured via `${RCLONE_REMOTE:-gdrive}` in `.env` (e.g. `gdrive`).
- **Destination Path**: `${RCLONE_REMOTE}:${DEST_FOLDER}` (default: `gdrive:WarsawDelayTelemetry/data/raw`).
- **Source Path**: `data/raw/`
- **Cron Trigger**: Daily at 03:00 UTC (`0 3 * * * .../backup_to_gdrive.sh`).
- **Logs**: Local backup execution history is tracked in `logs/backup.log`.

## 3. Pulling Partitions for Local Analytics
- **Script**: `pull_from_gdrive.sh`
- **Execution**: Run `./pull_from_gdrive.sh` to pull recent Parquet partitions from Google Drive down to local `data/raw/` with `--update` caching.
- **Log**: Operation logged to `logs/pull_gdrive.log`.
