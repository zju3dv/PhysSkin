#!/usr/bin/env bash
set -euo pipefail

# Sample normalized points and normals for latent encoding.
if [[ $# -lt 1 ]]; then
    echo "Usage: bash $0 <category_id> [max_jobs]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${DATASET_ROOT:?Set DATASET_ROOT to the dataset root}"
SAMPLING_PYTHON="${SAMPLING_PYTHON:-python}"
CATEGORY_ID="$1"
MAX_JOBS="${2:-32}"
BASE_DIR="${DATASET_ROOT}/${CATEGORY_ID}"

if [[ ! -d "${BASE_DIR}" ]]; then
    echo "ERROR: category directory does not exist: ${BASE_DIR}" >&2
    exit 1
fi

mapfile -t MODEL_DIRS < <(find "${BASE_DIR}" -mindepth 1 -maxdepth 1 -type d | sort -V)
sample_one() {
    local model_dir="$1"
    local wt_mesh="${model_dir}/models/watertight/wt_model.obj"
    local sample_dir="${model_dir}/models/samples"
    if [[ -f "${sample_dir}/norm_pc_normals.npz" ]]; then
        echo "SKIP: $(basename "${model_dir}")"
        return
    fi
    mkdir -p "${sample_dir}"
    "${SAMPLING_PYTHON}" "${SCRIPT_DIR}/mesh_to_relaxed_sdf/sample_latent_pc.py" \
        input.mesh_path="${wt_mesh}" input.mesh_id=wt_model \
        sampling.point_number=8192 sampling.num_split=1 \
        output.save_dir="${sample_dir}"
}
for index in "${!MODEL_DIRS[@]}"; do
    sample_one "${MODEL_DIRS[index]}" &
    if (( (index + 1) % MAX_JOBS == 0 )); then
        wait
    fi
done
wait
