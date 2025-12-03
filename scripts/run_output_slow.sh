#!/bin/bash

#SBATCH -J heat_diffusion_output_slow
#SBATCH -o ./output/output/slow/output/%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH -p mi2104x

set -e

mkdir -p "./output/output/slow/data/${SLURM_JOB_ID}"
cd "./output/output/slow/data/${SLURM_JOB_ID}"

echo "Running: srun ./build/src/heat_diffusion $@ --kernel slow --mode output"
srun "../../../../../build/src/heat_diffusion" "$@" --kernel slow --mode output