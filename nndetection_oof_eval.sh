#!/bin/bash
#SBATCH --job-name=picai-nndetection-oof-eval
#SBATCH --partition=teaching
#SBATCH --chdir=/home/ad.msoe.edu/kwaterskip/Research/picai/baselines/picai_nndetection_gc_algorithm
#SBATCH --output=logs/oof_eval_%j.log
#SBATCH --error=logs/oof_eval_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

# Slurm copies batch scripts into /var/spool/slurm/...; rely on --chdir above.
SCRIPTPATH="$(pwd)"
PROJECT_ROOT="$(cd "${SCRIPTPATH}/../.." && pwd)"

PREDICTIONS_DIR="${PREDICTIONS_DIR:-${SCRIPTPATH}/results/oof/combined}"
LABELS_ROOT="${LABELS_ROOT:-${PROJECT_ROOT}/data/picai_labels}"
SPLITS_DIR="${SPLITS_DIR:-${SCRIPTPATH}/data/splits/picai_nnunet}"
OUTPUT="${OUTPUT:-${PREDICTIONS_DIR}/metrics.json}"
VENV="${VENV:-${SCRIPTPATH}/.venv}"

mkdir -p "${SCRIPTPATH}/logs"

echo "Predictions: ${PREDICTIONS_DIR}"
echo "Labels:      ${LABELS_ROOT}"
echo "Splits:      ${SPLITS_DIR}"
echo "Output:      ${OUTPUT}"

if [[ ! -d "${PREDICTIONS_DIR}" ]]; then
  echo "Missing predictions directory: ${PREDICTIONS_DIR}" >&2
  echo "Run nndetection_oof_validate.sh first." >&2
  exit 1
fi

if [[ -x "${VENV}/bin/python" ]]; then
  PYTHON="${VENV}/bin/python"
else
  PYTHON="python3"
fi

"${PYTHON}" "${SCRIPTPATH}/evaluate_oof.py" \
  --predictions-dir "${PREDICTIONS_DIR}" \
  --labels-root "${LABELS_ROOT}" \
  --splits-dir "${SPLITS_DIR}" \
  --output "${OUTPUT}"

echo "Evaluation complete."
