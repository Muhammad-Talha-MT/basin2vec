#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Basin2Vec Round-2 ablation embedding extraction
# ------------------------------------------------------------
# Reads  emb/src/checkpoints/ablation_round2/Basin2Vec_<base>_<variant>/seed_<seed>/epoch_<EPOCH>.pt
# Writes emb/src/evaluation_outputs/ablation_round2/Basin2Vec_<base>_<variant>/seed_<seed>/
#        basin_year_embeddings.parquet, basin_embeddings.parquet (Eq. 14), extraction_manifest.json
#
# Examples:
#   ./run_ablation_extraction.sh
#
#   # Settings of the shipped Round-2 results (see README, "Round-2 ablations"):
#   EPOCH=5 ./run_ablation_extraction.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BASE_CONFIG="${BASE_CONFIG:-paper}"
EPOCH="${EPOCH:-100}"
VARIANTS="${VARIANTS:-no_monthly_branch,no_static_branch,no_metadata_encoder,metadata_concat,no_meta_denominator,meta_as_positive,no_negative_margin,instance_contrastive}"
SEEDS="${SEEDS:-42,123,456}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SCRIPT_DIR}/checkpoints/ablation_round2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/evaluation_outputs/ablation_round2}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

VISIBLE_GPU_COUNT=$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)

if [[ "${VISIBLE_GPU_COUNT}" -lt "${NPROC_PER_NODE}" ]]; then
  echo "[ERROR] Requested NPROC_PER_NODE=${NPROC_PER_NODE}, but PyTorch sees ${VISIBLE_GPU_COUNT} CUDA device(s)." >&2
  echo "[ERROR] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi

echo "============================================================"
echo "Basin2Vec Round-2 embedding extraction"
echo "============================================================"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PyTorch visible GPUs=${VISIBLE_GPU_COUNT}"
echo "nproc=${NPROC_PER_NODE}"
echo "base_config=${BASE_CONFIG}"
echo "epoch=${EPOCH}"
echo "variants=${VARIANTS}"
echo "seeds=${SEEDS}"
echo "batch/rank=${BATCH_SIZE}"
echo "workers/rank=${NUM_WORKERS}"
echo "checkpoint_root=${CHECKPOINT_ROOT}"
echo "output_root=${OUTPUT_ROOT}"
echo "============================================================"

exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  "${SCRIPT_DIR}/extract_basin2vec_ablation_embeddings.py" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --base-config "${BASE_CONFIG}" \
  --epoch "${EPOCH}" \
  --variants "${VARIANTS}" \
  --seeds "${SEEDS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}"
