#!/usr/bin/env bash
# Launch the diet-opt FastAPI server.
# Reads optional config from deploy/diet-opt.env (see diet-opt.env.example).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

ENV_FILE="$REPO_DIR/deploy/diet-opt.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

PYTHON_BIN="${DIET_OPT_PYTHON:-$HOME/Documents/py_venv/bin/python}"
HOST="${DIET_OPT_HOST:-127.0.0.1}"
PORT="${DIET_OPT_PORT:-8000}"
WORKERS="${DIET_OPT_WORKERS:-1}"

exec "$PYTHON_BIN" -m uvicorn diet_opt.web.app:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --proxy-headers \
    --forwarded-allow-ips="${DIET_OPT_TRUSTED_PROXIES:-127.0.0.1}"
