#!/bin/bash

#SBATCH -J heat_diffusion_solver_slow
#SBATCH -o ./output/solver/slow/output/%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH -p mi2104x

set -e

mkdir -p "./output/solver/slow/data/${SLURM_JOB_ID}"
cd "./output/solver/slow/data/${SLURM_JOB_ID}"

echo "Running: srun ./build/src/heat_diffusion_solver $@ --mode slow"
srun "../../../../../build/src/heat_diffusion_solver" "$@" --mode slow