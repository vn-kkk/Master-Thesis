#!/bin/bash
#SBATCH --job-name=TOV4_create_dataset
#SBATCH --output=TOV4_create_dataset%j.out
#SBATCH --error=TOV4_create_dataset%j.err
#SBATCH --nodes=1                   # Use all 4 nodes: 135,136,137,138
#SBATCH --ntasks=1                  # Total MPI ranks (32+16+16+8)
#SBATCH --cpus-per-task=32
#SBATCH --time=96:00:00             # Walltime limit
#SBATCH --mem=0                     # Use all available memory on each node
#SBATCH --partition=all		        # Partition to submit the job to

# Set OMP_NUM_THREADS to match requested CPUs-per-task for efficient CPU usage
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print some info
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on: $(hostname)"
echo "Starting at: $(date)"

echo "Starting Python training script:TOV4_create_dataset.py"

python TOV4_create_dataset.py

echo "Job finished at: $(date)"