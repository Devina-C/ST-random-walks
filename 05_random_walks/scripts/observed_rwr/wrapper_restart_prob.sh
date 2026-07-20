#!/bin/bash
#SBATCH --job-name=ccc_sweep
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ccc_sweep_%j.log
#SBATCH --error=logs/ccc_sweep_%j.err

# Controller job: sweeps over r values, submitting and waiting on sub-jobs.
# Generous 24h walltime since most of its life is spent idle waiting in
# polling loops for child jobs to finish.

set -e   # exit on any error

mkdir -p logs

cd /scratch/users/k22026807/masters/project/random_walks

source /software/spackages_v0_21_prod/apps/linux-ubuntu22.04-zen3/gcc-13.2.0/anaconda3-2022.10-askrqkr7fpl3uzrxqevlyoc7hanrxyu5/etc/profile.d/conda.sh
conda activate random_walks

# r values to sweep (edit this list as needed)
R_VALUES=(0.1 0.3 0.5 0.05)

echo "==========================================="
echo "Sweep started at $(date)"
echo "Controller job ID: $SLURM_JOB_ID"
echo "R values: ${R_VALUES[*]}"
echo "==========================================="

wait_for_job() {
    local jobid=$1
    local label=$2
    echo "  Waiting for ${label} (job ${jobid})..."
    while squeue -j "$jobid" 2>/dev/null | tail -n +2 | grep -q "$jobid"; do
        sleep 30
    done
    echo "  [${label}] finished at $(date +%H:%M:%S)"
}

wait_for_array() {
    local jobid=$1
    local label=$2
    echo "  Waiting for array ${label} (job ${jobid})..."
    while squeue -j "$jobid" 2>/dev/null | tail -n +2 | grep -q "$jobid"; do
        local n=$(squeue -j "$jobid" 2>/dev/null | tail -n +2 | wc -l)
        echo "    $(date +%H:%M:%S) - ${n} tasks remaining"
        sleep 60
    done
    echo "  [${label}] array finished at $(date +%H:%M:%S)"
}

for R in "${R_VALUES[@]}"; do
    R_TAG=$(python -c "print(f'r{int($R * 1000):04d}')")
    echo ""
    echo "==========================================="
    echo "STARTING r=${R} (tag=${R_TAG}) at $(date)"
    echo "==========================================="

    # Step 1: set r in all scripts
    echo "Step 1: setting r=${R} in all scripts"
    ./scripts/set_r.sh "$R"
    grep "^RWR_RESTART\|^RWR_R " scripts/update_seeds.py scripts/run_rwr.py \
                                  scripts/lr_score.py scripts/permutation_test.py \
                                  scripts/merge_and_BH_correction.py

    # Step 2: observed CCC pipeline
    echo ""
    echo "Step 2: observed CCC pipeline"
    OBS_JOB=$(sbatch --parsable scripts/slurm/run_obs_pipeline.sh)
    echo "  Submitted obs pipeline as job ${OBS_JOB}"
    wait_for_job "$OBS_JOB" "obs_pipeline_${R_TAG}"

    if [ ! -f "results/ccc_results/ccc_all_lr_pairs_${R_TAG}.csv" ]; then
        echo "ERROR: ccc_all_lr_pairs_${R_TAG}.csv not produced. Aborting."
        exit 1
    fi
    echo "  Verified: ccc_all_lr_pairs_${R_TAG}.csv exists"

    # Step 3: permutation array
    echo ""
    echo "Step 3: permutation array"
    PERM_JOB=$(sbatch --parsable scripts/slurm/run_perm.sh)
    echo "  Submitted perm array as job ${PERM_JOB}"
    wait_for_array "$PERM_JOB" "perm_${R_TAG}"

    N_CHUNKS=$(ls results/ccc_results/permutation_test_results_chunk*_${R_TAG}.csv 2>/dev/null | wc -l)
    if [ "$N_CHUNKS" -ne 16 ]; then
        echo "ERROR: expected 16 chunk files for ${R_TAG}, found ${N_CHUNKS}. Aborting."
        exit 1
    fi
    echo "  Verified: 16 chunk files exist"

    # Step 4: merge + BH
    echo ""
    echo "Step 4: merge + BH correction"
    python scripts/merge_and_BH_correction.py

    if [ ! -f "results/ccc_results/permutation_test_results_${R_TAG}.csv" ]; then
        echo "ERROR: merged file not produced. Aborting."
        exit 1
    fi

    echo ""
    echo "DONE r=${R} at $(date)"
done

echo ""
echo "==========================================="
echo "ALL r VALUES COMPLETE at $(date)"
echo "==========================================="