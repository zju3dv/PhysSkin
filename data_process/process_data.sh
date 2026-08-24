#!/usr/bin/env bash
set -euo pipefail

# Existing outputs are skipped, so interrupted jobs can be resumed.

if [[ $# -lt 2 ]]; then
    echo "Usage: bash $0 <category_id> <gpu_id> [gpu_id ...]"
    echo "Example: bash $0 00000001 0 1"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${DATASET_ROOT:?Set DATASET_ROOT to the dataset root}"
: "${MICHELANGELO_ROOT:?Set MICHELANGELO_ROOT to the package root}"
: "${ENCODER_CHECKPOINT:?Set ENCODER_CHECKPOINT to the encoder checkpoint}"
SAMPLING_PYTHON="${SAMPLING_PYTHON:-python}"
RAYTRACING_PYTHON="${RAYTRACING_PYTHON:-python}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"
FILL_RESOLUTION="${FILL_RESOLUTION:-64}"

CATEGORY_ID="$1"
shift
GPU_IDS=("$@")
BASE_DIR="${DATASET_ROOT}/${CATEGORY_ID}"

if [[ ! -d "${BASE_DIR}" ]]; then
    echo "ERROR: category directory does not exist: ${BASE_DIR}" >&2
    exit 1
fi
if ! command -v "${SAMPLING_PYTHON}" >/dev/null 2>&1 && [[ ! -x "${SAMPLING_PYTHON}" ]]; then
    echo "ERROR: SAMPLING_PYTHON is not executable: ${SAMPLING_PYTHON}" >&2
    exit 1
fi
if ! command -v "${RAYTRACING_PYTHON}" >/dev/null 2>&1 && [[ ! -x "${RAYTRACING_PYTHON}" ]]; then
    echo "ERROR: RAYTRACING_PYTHON is not executable: ${RAYTRACING_PYTHON}" >&2
    exit 1
fi

export DATASET_ROOT
export PYTHON_BIN="${RAYTRACING_PYTHON}"
export MAX_TASKS_PER_GPU
bash "${SCRIPT_DIR}/process_wtmesh.sh" "${CATEGORY_ID}" "${GPU_IDS[@]}"

mapfile -t MODEL_DIRS < <(find "${BASE_DIR}" -mindepth 1 -maxdepth 1 -type d | sort -V)

process_samples() {
    local model_dir="$1"
    local gpu_id="$2"
    local model_root="${model_dir}/models"
    local wt_dir="${model_root}/watertight"
    local wt_mesh="${wt_dir}/wt_model.obj"
    local sample_dir="${model_root}/samples"

    if [[ ! -f "${wt_mesh}" ]]; then
        echo "SKIP: missing ${wt_mesh}"
        return
    fi
    mkdir -p "${sample_dir}"

    if [[ ! -f "${sample_dir}/sample_surface.ply" ]]; then
        echo "SURFACE: $(basename "${model_dir}") on GPU ${gpu_id}"
        CUDA_VISIBLE_DEVICES="${gpu_id}" "${SAMPLING_PYTHON}" \
            "${SCRIPT_DIR}/mesh_to_relaxed_sdf/mesh_sample.py" \
            input.mesh_path="${wt_mesh}" input.mesh_id=wt_model \
            sampling.point_number=200000 sampling.num_split=1 \
            output.save_dir="${sample_dir}"
    fi

    if [[ ! -f "${sample_dir}/internal_filled.npz" ]]; then
        echo "FILL: $(basename "${model_dir}") on GPU ${gpu_id}"
        "${RAYTRACING_PYTHON}" "${SCRIPT_DIR}/obj2ply_org.py" \
            --input_obj "${wt_mesh}" --output_ply "${wt_dir}/wt_model.ply"
        CUDA_VISIBLE_DEVICES="${gpu_id}" "${RAYTRACING_PYTHON}" \
            "${SCRIPT_DIR}/fill_raytracing.py" \
            --mesh_file "${wt_dir}/wt_model.ply" \
            --resolution "${FILL_RESOLUTION}" \
            --surface_path "${sample_dir}/sample_surface.ply" \
            --save_dir "${model_root}"
    fi

    if [[ ! -f "${sample_dir}/norm_pc_normals.npz" ]]; then
        echo "LATENT POINTS: $(basename "${model_dir}") on GPU ${gpu_id}"
        CUDA_VISIBLE_DEVICES="${gpu_id}" "${SAMPLING_PYTHON}" \
            "${SCRIPT_DIR}/mesh_to_relaxed_sdf/sample_latent_pc.py" \
            input.mesh_path="${wt_mesh}" input.mesh_id=wt_model \
            sampling.point_number=8192 sampling.num_split=1 \
            output.save_dir="${sample_dir}"
    fi
}

batch_size=$((MAX_TASKS_PER_GPU * ${#GPU_IDS[@]}))
for index in "${!MODEL_DIRS[@]}"; do
    gpu_id="${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}"
    process_samples "${MODEL_DIRS[index]}" "${gpu_id}" &
    if (( (index + 1) % batch_size == 0 )); then
        wait
    fi
done
wait

CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" "${SAMPLING_PYTHON}" \
    "${SCRIPT_DIR}/encode_latents.py" \
    --dataset-root "${DATASET_ROOT}" \
    --category "${CATEGORY_ID}" \
    --michelangelo-root "${MICHELANGELO_ROOT}" \
    --encoder-checkpoint "${ENCODER_CHECKPOINT}" \
    --device cuda

echo "All PhysSkin preprocessing stages complete."
