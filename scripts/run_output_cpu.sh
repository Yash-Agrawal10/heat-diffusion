#!/bin/bash

#SBATCH -J heat_diffusion_output_cpu
#SBATCH -o ./output/output/cpu/output/%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -p mi2104x

set -e

mkdir -p "./output/output/cpu/data/${SLURM_JOB_ID}"
cd "./output/output/cpu/data/${SLURM_JOB_ID}"

echo "Running: srun ./build/src/heat_diffusion $@ --kernel cpu --mode output"
srun "../../../../../build/src/heat_diffusion" "$@" --kernel cpu --mode output