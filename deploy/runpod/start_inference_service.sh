#!/usr/bin/env bash
set -euo pipefail

PORT="${RUNPOD_INFERENCE_PORT:-8001}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT_DIR="${RUNPOD_REMOTE_ROOT:-/workspace/aneurysm}"
APP_DIR="${RUNPOD_REMOTE_APP:-$REPO_DIR}"
WORKSPACE_DIR="${RUNPOD_REMOTE_WORKSPACE:-$ROOT_DIR/workspace}"
LOG_DIR="$WORKSPACE_DIR/logs"
LOG_FILE="$LOG_DIR/cloud_inference_${PORT}.log"
PID_FILE="$ROOT_DIR/cloud_inference_${PORT}.pid"
SERVICE_SCRIPT="$APP_DIR/python/segmend_runpod_service.py"

if [[ ! -f "$SERVICE_SCRIPT" ]]; then
  if [[ -f "$APP_DIR/python/nnunet_microservice.py" ]]; then
    SERVICE_SCRIPT="$APP_DIR/python/nnunet_microservice.py"
  elif [[ -f "$ROOT_DIR/python/segmend_runpod_service.py" ]]; then
    APP_DIR="$ROOT_DIR"
    SERVICE_SCRIPT="$APP_DIR/python/segmend_runpod_service.py"
  elif [[ -f "$ROOT_DIR/python/nnunet_microservice.py" ]]; then
    APP_DIR="$ROOT_DIR"
    SERVICE_SCRIPT="$APP_DIR/python/nnunet_microservice.py"
  else
    echo "Could not find SegMend service script under $APP_DIR or $ROOT_DIR" >&2
    exit 2
  fi
fi

VENV_DIR="${VENV_DIR:-$WORKSPACE_DIR/.venv_nnunet}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "$VENV_DIR/bin/python" ]]; then
  PYTHON_BIN="$VENV_DIR/bin/python"
fi

export NNUNET_RAW_DATA_BASE="${NNUNET_RAW_DATA_BASE:-$WORKSPACE_DIR/nnUNet_raw}"
export NNUNET_PREPROCESSED="${NNUNET_PREPROCESSED:-$WORKSPACE_DIR/nnUNet_preprocessed}"
export NNUNET_RESULTS="${NNUNET_RESULTS:-$WORKSPACE_DIR/nnUNet_results}"
export nnUNet_raw="$NNUNET_RAW_DATA_BASE"
export nnUNet_preprocessed="$NNUNET_PREPROCESSED"
export nnUNet_results="$NNUNET_RESULTS"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$WORKSPACE_DIR/inference/.matplotlib}"
export NNUNET_USE_MIRRORING="${NNUNET_USE_MIRRORING:-0}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-4}"
export TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-2}"

mkdir -p "$NNUNET_RAW_DATA_BASE" "$NNUNET_PREPROCESSED" "$NNUNET_RESULTS" "$MPLCONFIGDIR" "$LOG_DIR"

echo "Starting SegMend nnU-Net inference service"
echo "  app:     $APP_DIR"
echo "  python:  $PYTHON_BIN"
echo "  results: $NNUNET_RESULTS"
echo "  port:    $PORT"
echo "  script:  $SERVICE_SCRIPT"

if ! "$PYTHON_BIN" -c "import flask, nnunetv2" >/dev/null 2>&1; then
  "$PYTHON_BIN" -m pip install --no-cache-dir --ignore-installed \
    blinker flask nnunetv2 SimpleITK nibabel scikit-image opencv-python-headless
fi

if pgrep -f "segmend_runpod_service.py|nnunet_microservice.py.*--server" >/dev/null 2>&1; then
  pkill -f "segmend_runpod_service.py|nnunet_microservice.py.*--server" || true
  sleep 2
fi

cd "$APP_DIR"
if [[ "$(basename "$SERVICE_SCRIPT")" == "segmend_runpod_service.py" ]]; then
  nohup "$PYTHON_BIN" "$SERVICE_SCRIPT" --host 0.0.0.0 --port "$PORT" > "$LOG_FILE" 2>&1 &
else
  nohup "$PYTHON_BIN" "$SERVICE_SCRIPT" --server --host 0.0.0.0 --port "$PORT" > "$LOG_FILE" 2>&1 &
fi
echo "$!" > "$PID_FILE"

echo "Started PID $(cat "$PID_FILE"). Waiting for health..."
for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/tmp/segmend_health.json 2>/dev/null; then
    cat /tmp/segmend_health.json
    echo
    echo "Healthy. Use the RunPod HTTP URL for port $PORT in SegMend."
    exit 0
  fi
  sleep 1
done

echo "Service did not become healthy. Last log lines:" >&2
tail -n 80 "$LOG_FILE" >&2 || true
exit 1
