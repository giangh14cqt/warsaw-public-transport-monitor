# 03_code_style.md: Python Standards & Development Conventions

## 1. Environment & Package Management
- Use `uv` for ultra-fast virtual environment creation and deterministic dependency resolution:
  ```bash
  uv venv --python 3.12
  uv pip install -r requirements.txt
  ```
- Package configurations and dependencies are defined in `pyproject.toml`.

## 2. Code Quality & Formatting
- **Linter & Formatter**: Use `ruff` (`ruff check .`, `ruff format .`).
- **Python Version**: Minimum Python 3.12.
- **Type Annotations**: Enforce PEP 484 type hints across all module function signatures (`def func(param: str) -> Optional[dict]:`).
- **Docstrings**: Provide Google-style docstrings describing parameters, return values, and raised exceptions.

## 3. Backwards Compatibility
- The root files `collector_daemon.py`, `run_preflight.py`, `backup_to_gdrive.sh`, `Dockerfile`, and `docker-compose.yml` must remain runnable at all times to avoid breaking the production daemon on remote self-hosted servers when `git pull` is executed.
- Expose re-export shims in `src/__init__.py` and root module proxies if legacy import paths are referenced.
