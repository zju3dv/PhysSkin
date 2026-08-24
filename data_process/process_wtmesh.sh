#!/usr/bin/env bash
set -euo pipefail

# Convert every model_normalized.obj in one category into a watertight mesh.

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

if [[ $# -lt 2 ]]; then
    echo "Usage: bash $0 <category_id> <gpu_id> [gpu_id ...]"
    echo "Example: bash $0 00000001 0 1"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${DATASET_ROOT:?Set DATASET_ROOT to the dataset root}"
PYTHON_EXECUTABLE="${PYTHON_BIN:-python}"

CATEGORY_ID="$1"
shift
GPU_IDS=("$@")
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"
RESOLUTION="${RESOLUTION:-512}"
BASE_DIR="${DATASET_ROOT}/${CATEGORY_ID}"
WATERTIGHT_SCRIPT="${SCRIPT_DIR}/mesh_to_relaxed_sdf/to_watertight_mesh.py"

if [[ ! -d "${BASE_DIR}" ]]; then
    echo "ERROR: category directory does not exist: ${BASE_DIR}" >&2
    exit 1
fi
if [[ ! -f "${WATERTIGHT_SCRIPT}" ]]; then
    echo "ERROR: converter not found: ${WATERTIGHT_SCRIPT}" >&2
    exit 1
fi

mapfile -t MODEL_DIRS < <(find "${BASE_DIR}" -mindepth 1 -maxdepth 1 -type d | sort -V)
echo "Dataset root: ${DATASET_ROOT}"
echo "Category: ${CATEGORY_ID}"
echo "GPUs: ${GPU_IDS[*]}"
echo "Models: ${#MODEL_DIRS[@]}"

process_model() {
    local model_dir="$1"
    local gpu_id="$2"
    local input_mesh="${model_dir}/models/model_normalized.obj"
    local output_dir="${model_dir}/models/watertight"
    local output_mesh="${output_dir}/wt_model.obj"

    if [[ ! -f "${input_mesh}" ]]; then
        echo "SKIP: missing ${input_mesh}"
        return
    fi
    if [[ -f "${output_mesh}" ]]; then
        echo "SKIP: ${output_mesh} already exists"
        return
    fi

    mkdir -p "${output_dir}"
    echo "PROCESS: $(basename "${model_dir}") on GPU ${gpu_id}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_EXECUTABLE}" "${WATERTIGHT_SCRIPT}" \
        input.mesh_path="${input_mesh}" \
        input.mesh_id=wt_model \
        output.save_dir="${output_dir}" \
        processing.resolution="${RESOLUTION}" \
        processing.scale=1.0
}

batch_size=$((MAX_TASKS_PER_GPU * ${#GPU_IDS[@]}))
for index in "${!MODEL_DIRS[@]}"; do
    gpu_id="${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}"
    process_model "${MODEL_DIRS[index]}" "${gpu_id}" &
    if (( (index + 1) % batch_size == 0 )); then
        wait
    fi
done
wait
echo "Watertight processing complete."
