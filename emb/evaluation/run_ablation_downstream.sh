#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Basin2Vec Round-2 downstream ablation launcher
#
# Evaluates each extracted Eq. 14 descriptor with the LSTM, L=365
# temporal and spatial protocols of temporal_split.py and
# spatial_split.py. The pretraining seed is also the downstream seed.
#
# Reads  emb/src/evaluation_outputs/ablation_round2/Basin2Vec_<base>_<variant>/seed_<seed>/basin_embeddings.parquet
# Writes emb/evaluation/evaluation_outputs/ablation_round2/{temporal,spatial}/<variant>/seed_<seed>/
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

BASE_CONFIG="${BASE_CONFIG:-paper}"
EMBEDDING_ROOT="${EMBEDDING_ROOT:-${SCRIPT_DIR}/../src/evaluation_outputs/ablation_round2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/evaluation_outputs/reproduced_ablation_round2}"

VARIANTS="${VARIANTS:-instance_contrastive,meta_as_positive,metadata_concat,no_metadata_encoder,no_meta_denominator,no_monthly_branch,no_negative_margin,no_static_branch}"
SEEDS="${SEEDS:-42,123,456}"

# Any combination of: temporal,spatial
SPLITS="${SPLITS:-temporal,spatial}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

IFS=',' read -r -a VARIANT_ARRAY <<< "${VARIANTS}"
IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"
IFS=',' read -r -a SPLIT_ARRAY <<< "${SPLITS}"

echo "============================================================"
echo "Basin2Vec Round-2 downstream ablation suite"
echo "============================================================"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "nproc=${NPROC_PER_NODE}"
echo "base_config=${BASE_CONFIG}"
echo "embedding_root=${EMBEDDING_ROOT}"
echo "output_root=${OUTPUT_ROOT}"
echo "variants=${VARIANTS}"
echo "seeds=${SEEDS}"
echo "splits=${SPLITS}"
echo "============================================================"

for variant in "${VARIANT_ARRAY[@]}"; do
    variant="$(echo "${variant}" | xargs)"
    [[ -z "${variant}" ]] && continue

    for seed in "${SEED_ARRAY[@]}"; do
        seed="$(echo "${seed}" | xargs)"
        [[ -z "${seed}" ]] && continue

        PARQUET="${EMBEDDING_ROOT}/Basin2Vec_${BASE_CONFIG}_${variant}/seed_${seed}/basin_embeddings.parquet"

        if [[ ! -f "${PARQUET}" ]]; then
            echo "[ERROR] Missing embedding parquet:"
            echo "        ${PARQUET}"
            exit 1
        fi

        for split in "${SPLIT_ARRAY[@]}"; do
            split="$(echo "${split}" | xargs)"
            [[ -z "${split}" ]] && continue

            case "${split}" in
                temporal|spatial)
                    script="${SCRIPT_DIR}/${split}_split.py"
                    ;;
                *)
                    echo "[ERROR] Unknown split '${split}'. Use temporal and/or spatial."
                    exit 1
                    ;;
            esac

            echo
            echo "============================================================"
            echo "split=${split} variant=${variant} seed=${seed}"
            echo "parquet=${PARQUET}"
            echo "============================================================"

            torchrun \
                --standalone \
                --nproc_per_node="${NPROC_PER_NODE}" \
                "${script}" \
                --models lstm \
                --static-sources basin2vec \
                --seq-lengths 365 \
                --basin2vec-parquet "${PARQUET}" \
                --seed "${seed}" \
                --out-dir "${OUTPUT_ROOT}/${split}/${variant}/seed_${seed}"
        done
    done
done

echo
echo "[COMPLETE] Requested downstream ablation runs finished."
echo
echo "Aggregate results with:"
echo "python ${SCRIPT_DIR}/aggregate_ablation_results.py \\"
echo "  --root ${OUTPUT_ROOT} \\"
echo "  --variants ${VARIANTS} \\"
echo "  --seeds ${SEEDS}"
