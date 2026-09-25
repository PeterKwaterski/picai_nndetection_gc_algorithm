#!/bin/bash
#SBATCH --job-name=picai-nndetection-oof
#SBATCH --partition=teaching
#SBATCH --chdir=/home/ad.msoe.edu/kwaterskip/Research/picai/baselines/picai_nndetection_gc_algorithm
#SBATCH --output=logs/oof_%A_%a.log
#SBATCH --error=logs/oof_%A_%a.err
#SBATCH --array=0-4
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

SCRIPTPATH="$(pwd)"
PROJECT_ROOT="$(cd "${SCRIPTPATH}/../.." && pwd)"

GC_CASES_DIR="${GC_CASES_DIR:-${PROJECT_ROOT}/data/gc_cases}"
SPLITS_DIR="${SPLITS_DIR:-${SCRIPTPATH}/data/splits/picai_nnunet}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPTPATH}/results}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPTPATH}/results/oof}"
COMBINED_DIR="${COMBINED_DIR:-${OUTPUT_DIR}/combined}"
SIF="${SIF:-${SCRIPTPATH}/picai_baseline_nndetection_processor.sif}"

FOLD="${SLURM_ARRAY_TASK_ID:?Set SLURM_ARRAY_TASK_ID via --array}"

mkdir -p "${SCRIPTPATH}/logs" "${OUTPUT_DIR}" "${COMBINED_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID:-$$}-${FOLD}"
mkdir -p "${MPLCONFIGDIR}"

# Single scratch tree so nndet can rename between results/ and nndet/output/
JOB_SCRATCH="/tmp/nndet-oof-${SLURM_JOB_ID:-$$}-${FOLD}"
mkdir -p \
  "${JOB_SCRATCH}/results" \
  "${JOB_SCRATCH}/nndet/input" \
  "${JOB_SCRATCH}/nndet/output" \
  "${JOB_SCRATCH}/det_data" \
  "${JOB_SCRATCH}/work/nnDet_raw_data"

export MKL_THREADING_LAYER=GNU

echo "Project root:  ${PROJECT_ROOT}"
echo "Fold:          ${FOLD}"
echo "GC cases:      ${GC_CASES_DIR}"
echo "Splits:        ${SPLITS_DIR}"
echo "Job scratch:   ${JOB_SCRATCH}"
echo "Output:        ${OUTPUT_DIR}/fold_${FOLD}"
echo "Combined:      ${COMBINED_DIR}"

module purge
module load singularity/3.10.0
module load cuda/12.9

cp -a "${RESULTS_DIR}/." "${JOB_SCRATCH}/results/"

singularity exec --nv --no-home \
  --env MKL_THREADING_LAYER=GNU \
  --env det_data=/scratch/det_data \
  --env det_models=/scratch/results/nnDet \
  --bind "${GC_CASES_DIR}:/gc_cases:ro" \
  --bind "${SPLITS_DIR}:/splits:ro" \
  --bind "${JOB_SCRATCH}:/scratch" \
  --bind "${OUTPUT_DIR}:/output" \
  --bind "${COMBINED_DIR}:/combined" \
  --bind "${SCRIPTPATH}/run_oof_fold.py:/opt/algorithm/run_oof_fold.py:ro" \
  "${SIF}" \
  python3 /opt/algorithm/run_oof_fold.py \
    --fold "${FOLD}" \
    --gc-cases-dir /gc_cases \
    --splits-dir /splits \
    --results-dir /scratch/results \
    --output-dir /output \
    --combined-dir /combined

echo "Fold ${FOLD} complete."
