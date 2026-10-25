#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Basin2Vec Round-2 ablation pretraining launcher
# ------------------------------------------------------------
# Checkpoints go to emb/src/checkpoints/ablation_round2/Basin2Vec_<base>_<variant>/seed_<seed>/.
#
# Examples:
#   NPROC_PER_NODE=4 BASE_CONFIG=paper EPOCHS=100 ./run_ablation_pretraining.sh
#
#   # Settings of the shipped Round-2 results (see README, "Round-2 ablations"):
#   NPROC_PER_NODE=4 BASE_CONFIG=paper EPOCHS=8 ./run_ablation_pretraining.sh
#
# Debug mode (recommended for the first test after a failure):
#   DEBUG_DISTRIBUTED=1 NPROC_PER_NODE=2 EPOCHS=1 NUM_WORKERS=0 \
#     VARIANTS_OVERRIDE="full" SEEDS_OVERRIDE="42" \
#     ./run_ablation_pretraining.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
BASE_CONFIG="${BASE_CONFIG:-paper}"
EPOCHS="${EPOCHS:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEBUG_DISTRIBUTED="${DEBUG_DISTRIBUTED:-0}"
FIND_UNUSED_PARAMETERS="${FIND_UNUSED_PARAMETERS:-0}"
DETECT_ANOMALY="${DETECT_ANOMALY:-0}"

if [[ "${DEBUG_DISTRIBUTED}" == "1" ]]; then
  export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
  export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  export PYTHONFAULTHANDLER="1"
fi

DEFAULT_SEEDS=(42 123 456)
# The eight variants with shipped results. The trainer also supports
# full, no_temporal_positive, no_overlap_positive and no_area_regularization.
DEFAULT_VARIANTS=(
  no_monthly_branch
  no_static_branch
  no_metadata_encoder
  metadata_concat
  no_meta_denominator
  meta_as_positive
  no_negative_margin
  instance_contrastive
)

if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then
  read -r -a SEEDS <<< "${SEEDS_OVERRIDE}"
else
  SEEDS=("${DEFAULT_SEEDS[@]}")
fi

if [[ -n "${VARIANTS_OVERRIDE:-}" ]]; then
  read -r -a VARIANTS <<< "${VARIANTS_OVERRIDE}"
else
  VARIANTS=("${DEFAULT_VARIANTS[@]}")
fi

EXTRA_ARGS=()
if [[ "${FIND_UNUSED_PARAMETERS}" == "1" ]]; then
  EXTRA_ARGS+=(--find-unused-parameters)
fi
if [[ "${DETECT_ANOMALY}" == "1" ]]; then
  EXTRA_ARGS+=(--detect-anomaly)
fi

for variant in "${VARIANTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "============================================================"
    echo "variant=${variant} seed=${seed} base_config=${BASE_CONFIG}"
    echo "gpus=${NPROC_PER_NODE} epochs=${EPOCHS} workers/rank=${NUM_WORKERS}"
    echo "============================================================"

    torchrun \
      --standalone \
      --nproc_per_node="${NPROC_PER_NODE}" \
      "${SCRIPT_DIR}/train_basin2vec_ablation_suite.py" \
      --variant "${variant}" \
      --base-config "${BASE_CONFIG}" \
      --seed "${seed}" \
      --epochs "${EPOCHS}" \
      --num-workers "${NUM_WORKERS}" \
      "${EXTRA_ARGS[@]}"
  done
done
