#!/bin/bash
set -euo pipefail

# --- workdir + environment -------------------------------------------------
cd /dkfz/cluster/gpu/data/OE0441/t006d/Code/gmae      # <-- adjust to the path on the cluster
source .venv/bin/activate                           # uv venv created in gmae/

# --- paths -----------------------------------------------------------------
export IMAGENET_DIR=/dkfz/cluster/gpu/data/common/imagenet/ILSVRC/Data/CLS-LOC
export CKPT=/dkfz/cluster/gpu/checkpoints/OE0441/t006d/generalized_mim/gmae/mae_vitl_800e/checkpoint-799.pth
export OUT=/dkfz/cluster/gpu/checkpoints/OE0441/t006d/generalized_mim/gmae/linprobe_vitl_800e

# --- sanity: print which host + GPUs we actually landed on ------------------
echo "HOST=$(hostname)"; nvidia-smi -L
export NCCL_DEBUG=WARN

# --- MAE ViT-L linear probe (A100 40GB) ------------------------------------
# Keep the recipe effective batch of 16384 fixed; trade per-GPU batch for
# gradient accumulation so it fits in 40GB. ACCUM is derived from GPUS so the
# math stays correct whether you land on a 4- or 8-GPU node.
#   eff_batch = PER_GPU_BS * GPUS * ACCUM = 16384
#   lr = blr * eff_batch / 256 = 0.1 * 16384 / 256 = 6.4   (unchanged)
GPUS=8                 # <-- set to the #GPUs you actually requested (4 or 8)
PER_GPU_BS=256         # safe for ViT-L linprobe on a 40GB A100
TARGET_EFF=16384
ACCUM=$(( TARGET_EFF / (PER_GPU_BS * GPUS) ))
echo "GPUS=${GPUS} PER_GPU_BS=${PER_GPU_BS} ACCUM=${ACCUM} eff_batch=$(( PER_GPU_BS * GPUS * ACCUM ))"

# linprobe uses the CLS token + frozen BN head (global_pool=False, the default).
torchrun --standalone --nproc_per_node="${GPUS}" main_linprobe.py \
    --model vit_large_patch16 --cls_token \
    --finetune "${CKPT}" \
    --nb_classes 1000 --data_path "${IMAGENET_DIR}" \
    --batch_size "${PER_GPU_BS}" --accum_iter "${ACCUM}" \
    --epochs 50 --blr 0.1 --weight_decay 0.0 \
    --dist_eval \
    --output_dir "${OUT}" --log_dir "${OUT}" \
    --num_workers 16 --pin_mem
