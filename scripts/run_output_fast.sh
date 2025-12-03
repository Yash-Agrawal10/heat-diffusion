#!/bin/bash

#SBATCH -J heat_diffusion_output_fast
#SBATCH -o ./output/output/fast/output/%j.out
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 00:10:00
#SBATCH -p mi2104x

set -e

mkdir -p "./output/output/fast/data/${SLURM_JOB_ID}"
cd "./output/output/fast/data/${SLURM_JOB_ID}"

echo "Running: srun ./build/src/heat_diffusion $@ --kernel fast --mode output"
srun "../../../../../build/src/heat_diffusion" "$@" --kernel fast --mode output