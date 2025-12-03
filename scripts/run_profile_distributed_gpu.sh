#!/bin/bash

#SBATCH -J heat_diffusion_profile_distributed_gpu
#SBATCH -o ./output/profile/distributed_gpu/%j.out
#SBATCH -N 4
#SBATCH -n 16
#SBATCH -t 00:10:00
#SBATCH -p mi2104x

set -e

echo "Running: srun ./build/src/heat_diffusion $@ --kernel distributed_gpu --mode profile"
prun "./build/src/heat_diffusion" "$@" --kernel distributed_gpu --mode profile