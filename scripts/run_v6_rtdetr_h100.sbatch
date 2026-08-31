#!/bin/bash
#SBATCH --job-name=amia_v6_raw
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --qos=standard
#SBATCH --output=amia_v6_raw_%j.out
#SBATCH --error=amia_v6_raw_%j.err

set -euo pipefail

PROJECT_DIR="/scratch/fanm01/CV/amia_slurm_project_curta"
PYTHON_BIN="/scratch/fanm01/dms_predict/envs/CV/bin/python"

cd "$PROJECT_DIR"

export AMIA_DATA_DIR="/scratch/fanm01/CV/amia_slurm_project_curta/data/amia-public-challenge-2026"
export AMIA_WEIGHTS_DIR="/scratch/fanm01/CV/amia_slurm_project_curta/weights"
export AMIA_V6_WORK_DIR="/scratch/xiaolil02/amia_2026_work/v6_rtdetr_raw640"

# Controlled v6 experiment
export AMIA_V6_IMG_SIZE=640
export AMIA_V6_EPOCHS=220
export AMIA_V6_PATIENCE=30

# Single-GPU Ultralytics AutoBatch target.
# 0.75 means profile and choose a batch intended to use ~75% of available VRAM.
export AMIA_V6_BATCH_FRACTION=0.75

export AMIA_V6_WORKERS=8
export AMIA_V6_PRED_CONF_FLOOR=0.02
export AMIA_V6_PRED_IOU=0.40
export AMIA_V6_PRED_BATCH=16
export AMIA_V6_PRED_CHUNK=128

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========================================================================"
echo "AMIA v6 H100 job"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "========================================================================"

nvidia-smi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "Allocated GPU: ${GPU_NAME}"

if ! echo "$GPU_NAME" | grep -qi "H100"; then
    echo "ERROR: expected H100 but got ${GPU_NAME}"
    exit 1
fi

"$PYTHON_BIN" --version
"$PYTHON_BIN" - <<'PY'
import torch, ultralytics
print("torch:", torch.__version__)
print("ultralytics:", ultralytics.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
PY

# GPU telemetry every 60 s so utilization can be checked from the log.
(
  while true; do
    echo "[GPU MONITOR] $(date '+%F %T')"
    nvidia-smi \
      --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
      --format=csv,noheader,nounits
    sleep 60
  done
) &
MONITOR_PID=$!

cleanup_monitor() {
    kill "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup_monitor EXIT

srun "$PYTHON_BIN" -u "${PROJECT_DIR}/amia_v6_rtdetr_raw_h100.py"

echo "========================================================================"
echo "Finished: $(date)"
echo "========================================================================"
