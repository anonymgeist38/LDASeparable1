#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -f "$SCRIPT_DIR/prepare_real_mask_nnunet_dataset.py" ]; then
  HELPER_DIR="$SCRIPT_DIR"
  REPO_ROOT="$SCRIPT_DIR"
else
  HELPER_DIR="$REPO_ROOT/python"
fi

IMAGES_DIR=""
MASKS_DIR=""
DATASET_ID="${DATASET_ID:-1000}"
DATASET_NAME="${DATASET_NAME:-AneurysmSeg}"
TRAIN_CONFIG="${TRAIN_CONFIG:-2d}"
TRAINER="${TRAINER:-nnUNetTrainer_20epochs}"
FOLD="${FOLD:-0}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}"
THREADS="${THREADS:-4}"
WORKERS="${WORKERS:-2}"
TRAIN_SPLIT="${TRAIN_SPLIT:-0.8}"
MAX_CASES="${MAX_CASES:-}"
CHANNEL_NAME="${CHANNEL_NAME:-MRA}"
MIN_NONZERO="${MIN_NONZERO:-100}"
MIN_FRACTION="${MIN_FRACTION:-0.000001}"
NO_TRAIN="false"
PYTHON_BIN="${PYTHON_BIN:-python3}"

NNUNET_RAW="${NNUNET_RAW_DATA_BASE:-${nnUNet_raw:-$HOME/nnUNet_raw_data_base}}"
NNUNET_PREPROCESSED_DIR="${NNUNET_PREPROCESSED:-${nnUNet_preprocessed:-$HOME/nnUNet_preprocessed}}"
NNUNET_RESULTS_DIR="${NNUNET_RESULTS:-${nnUNet_results:-$HOME/nnUNet_results}}"

usage() {
  cat <<EOF
Usage: $(basename "$0") --images DIR --masks DIR [options]

Required:
  --images DIR          Folder with image .nii/.nii.gz files
  --masks DIR           Folder with matching dense mask .nii/.nii.gz files

Options:
  --dataset-id ID       nnU-Net dataset id (default: $DATASET_ID)
  --dataset-name NAME   nnU-Net dataset name suffix (default: $DATASET_NAME)
  --config CONFIG       Training config: 2d, 3d_fullres, 3d_lowres (default: $TRAIN_CONFIG)
  --trainer NAME        nnU-Net trainer (default: $TRAINER)
  --fold FOLD           Fold id (default: $FOLD)
  --gpu GPU             CUDA_VISIBLE_DEVICES value (default: $GPU_ID)
  --device DEVICE       Training device: cuda, cpu, or mps (default: $DEVICE)
  --batch-size N        Override planned batch size after preprocessing (default: $BATCH_SIZE)
  --threads N|auto      Torch/BLAS CPU threads for training (default: $THREADS)
  --workers N|auto      nnU-Net data augmentation workers (default: $WORKERS)
  --train-split FLOAT   Train/test split (default: $TRAIN_SPLIT)
  --max-cases N         Limit paired cases for tiny smoke runs
  --channel-name NAME    nnU-Net channel name, for example CT or MRA (default: $CHANNEL_NAME)
  --min-nonzero N       Minimum foreground voxels per training mask (default: $MIN_NONZERO)
  --min-fraction FLOAT  Minimum foreground fraction per training mask (default: $MIN_FRACTION)
  --raw-base DIR        nnU-Net raw data base (default: $NNUNET_RAW)
  --preprocessed DIR    nnU-Net preprocessed dir (default: $NNUNET_PREPROCESSED_DIR)
  --results DIR         nnU-Net results dir (default: $NNUNET_RESULTS_DIR)
  --python PATH         Python executable for helper scripts (default: $PYTHON_BIN)
  --no-train            Prepare and audit dataset, but do not start training
  -h, --help            Show this message
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --images) IMAGES_DIR="$2"; shift 2 ;;
    --masks) MASKS_DIR="$2"; shift 2 ;;
    --dataset-id) DATASET_ID="$2"; shift 2 ;;
    --dataset-name) DATASET_NAME="$2"; shift 2 ;;
    --config) TRAIN_CONFIG="$2"; shift 2 ;;
    --trainer) TRAINER="$2"; shift 2 ;;
    --fold) FOLD="$2"; shift 2 ;;
    --gpu) GPU_ID="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --train-split) TRAIN_SPLIT="$2"; shift 2 ;;
    --max-cases) MAX_CASES="$2"; shift 2 ;;
    --channel-name) CHANNEL_NAME="$2"; shift 2 ;;
    --min-nonzero) MIN_NONZERO="$2"; shift 2 ;;
    --min-fraction) MIN_FRACTION="$2"; shift 2 ;;
    --raw-base) NNUNET_RAW="$2"; shift 2 ;;
    --preprocessed) NNUNET_PREPROCESSED_DIR="$2"; shift 2 ;;
    --results) NNUNET_RESULTS_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --no-train) NO_TRAIN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [ -z "$IMAGES_DIR" ] || [ -z "$MASKS_DIR" ]; then
  usage
  exit 1
fi

INFERENCE_DIR="${NNUNET_INFERENCE_DIR:-$REPO_ROOT/inference}"
mkdir -p "$NNUNET_RAW" "$NNUNET_PREPROCESSED_DIR" "$NNUNET_RESULTS_DIR" "$INFERENCE_DIR"
mkdir -p "$INFERENCE_DIR/.matplotlib"

detect_cpu_count() {
  if command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.ncpu 2>/dev/null && return
  fi
  if command -v getconf >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN 2>/dev/null && return
  fi
  echo 4
}

CPU_COUNT="$(detect_cpu_count)"
if [ "$THREADS" = "auto" ]; then
  THREADS="$CPU_COUNT"
fi
if [ "$WORKERS" = "auto" ]; then
  if [ "$CPU_COUNT" -gt 2 ]; then
    WORKERS=$((CPU_COUNT / 2))
  else
    WORKERS=1
  fi
fi

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export TORCH_NUM_THREADS="$THREADS"
export TORCH_NUM_INTEROP_THREADS="$THREADS"
export nnUNet_n_proc_DA="$WORKERS"
export MPLCONFIGDIR="$INFERENCE_DIR/.matplotlib"

export nnUNet_raw="$NNUNET_RAW"
export nnUNet_preprocessed="$NNUNET_PREPROCESSED_DIR"
export nnUNet_results="$NNUNET_RESULTS_DIR"
export NNUNET_RAW_DATA_BASE="$NNUNET_RAW"
export NNUNET_PREPROCESSED="$NNUNET_PREPROCESSED_DIR"
export NNUNET_RESULTS="$NNUNET_RESULTS_DIR"

DATASET_FOLDER="$NNUNET_RAW/Dataset$(printf '%03d' "$DATASET_ID")_${DATASET_NAME}"

echo "[1/4] Preparing paired dense-mask nnU-Net dataset"
PREPARE_ARGS=(
  --images "$IMAGES_DIR"
  --masks "$MASKS_DIR"
  --output-base "$NNUNET_RAW"
  --dataset-id "$DATASET_ID"
  --dataset-name "$DATASET_NAME"
  --train-split "$TRAIN_SPLIT"
  --channel-name "$CHANNEL_NAME"
  --min-nonzero "$MIN_NONZERO"
)
if [ -n "$MAX_CASES" ]; then
  PREPARE_ARGS+=(--max-cases "$MAX_CASES")
fi
"$PYTHON_BIN" "$HELPER_DIR/prepare_real_mask_nnunet_dataset.py" \
  "${PREPARE_ARGS[@]}"

echo "[2/4] Auditing labels"
"$PYTHON_BIN" "$HELPER_DIR/audit_nnunet_dataset.py" \
  --dataset "$DATASET_FOLDER" \
  --output "$INFERENCE_DIR/Dataset$(printf '%03d' "$DATASET_ID")_${DATASET_NAME}_label_audit.json" \
  --min-nonzero "$MIN_NONZERO" \
  --min-fraction "$MIN_FRACTION"

echo "[3/4] Planning and preprocessing"
nnUNetv2_plan_and_preprocess \
  -d "$DATASET_ID" \
  --verify_dataset_integrity

if [ -n "$BATCH_SIZE" ]; then
  echo "[3/4] Setting $TRAIN_CONFIG batch size to $BATCH_SIZE"
  "$PYTHON_BIN" -c "import json; from pathlib import Path; p=Path('$NNUNET_PREPROCESSED_DIR') / 'Dataset$(printf '%03d' "$DATASET_ID")_${DATASET_NAME}' / 'nnUNetPlans.json'; d=json.loads(p.read_text()); d['configurations']['$TRAIN_CONFIG']['batch_size']=int('$BATCH_SIZE'); p.write_text(json.dumps(d, indent=4) + '\n')"
fi

if [ "$NO_TRAIN" = "true" ]; then
  echo "[4/4] Skipping training (--no-train)"
  exit 0
fi

echo "[4/4] Training nnU-Net"
echo "Using CPU threads: $THREADS"
echo "Using nnU-Net data augmentation workers: $WORKERS"
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  nnUNetv2_train "$DATASET_ID" "$TRAIN_CONFIG" "$FOLD" -tr "$TRAINER" -device "$DEVICE"
