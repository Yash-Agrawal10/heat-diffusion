#!/bin/bash

#SBATCH -J heat_diffusion_output_shared_gpu
#SBATCH -o ./output/output/shared_gpu/output/%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -p mi2104x

set -e

mkdir -p "./output/output/shared_gpu/data/${SLURM_JOB_ID}"
cd "./output/output/shared_gpu/data/${SLURM_JOB_ID}"

echo "Running: srun ./build/src/heat_diffusion $@ --kernel shared_gpu --mode output"
srun "../../../../../build/src/heat_diffusion" "$@" --kernel shared_gpu --mode output