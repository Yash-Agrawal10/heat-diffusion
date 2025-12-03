#!/bin/bash

#SBATCH -J heat_diffusion_output_distributed_gpu
#SBATCH -o ./output/output/distributed_gpu/output/%j.out
#SBATCH -N 4
#SBATCH -n 16
#SBATCH -t 00:10:00
#SBATCH -p mi2104x

set -e

mkdir -p "./output/output/distributed_gpu/data/${SLURM_JOB_ID}"
cd "./output/output/distributed_gpu/data/${SLURM_JOB_ID}"

echo "Running: srun ./build/src/heat_diffusion $@ --kernel distributed_gpu --mode output"
srun "../../../../../build/src/heat_diffusion" "$@" --kernel distributed_gpu --mode output