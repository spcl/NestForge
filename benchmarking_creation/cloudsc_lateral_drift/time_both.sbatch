#!/bin/bash
#SBATCH --account=g34
#SBATCH --job-name=bench-b1-both
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --uenv=prgenv-gnu/25.11:v1
#SBATCH --view=default

# Baseline and reference of cloudsc_lateral_drift, preset M, one GPU.
unset PYTHONPATH
export PYTHONPATH=/ritom/scratch/cscs/nribeiro/thesis/bench/NestForge
# This uenv has libgomp and no libomp. The linked program looks up libomp on LIBRARY_PATH.
export LIBRARY_PATH=/ritom/scratch/cscs/nribeiro/thesis/omp-env/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export LD_LIBRARY_PATH=/ritom/scratch/cscs/nribeiro/thesis/omp-env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
source /ritom/scratch/cscs/nribeiro/thesis/venv-bc/bin/activate
BENCH=/ritom/scratch/cscs/nribeiro/thesis/bench/lateral
export TMPDIR=$BENCH/tmp-$SLURM_JOB_ID
mkdir -p "$TMPDIR"
cd "$BENCH"
echo "start $(date -Is) host=$(hostname)"
python -c "import dace,nestforge,hpcagent_bench; print('dace', dace.__file__); print('nestforge', nestforge.__file__)"
nvidia-smi -L
python "$PYTHONPATH/benchmarking_creation/cloudsc_lateral_drift/time_both.py"
echo "end $(date -Is)"
