#!/bin/bash

#SBATCH -J heat_diffusion
#SBATCH -o ./output/out/%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:10:00

set -e

mkdir -p "./output/data/${SLURM_JOB_ID}"
cd "./output/data/${SLURM_JOB_ID}"

echo "Running: srun ./build/src/heat_diffusion $@"
srun "../../../build/src/heat_diffusion" "$@"