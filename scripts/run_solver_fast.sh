#!/bin/bash

#SBATCH -J heat_diffusion_solver_fast
#SBATCH -o ./output/solver/fast/output/%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH -p mi2104x

set -e

mkdir -p "./output/solver/fast/data/${SLURM_JOB_ID}"
cd "./output/solver/fast/data/${SLURM_JOB_ID}"

echo "Running: srun ./build/src/heat_diffusion_solver $@ --mode fast"
srun "../../../../../build/src/heat_diffusion_solver" "$@" --mode fast