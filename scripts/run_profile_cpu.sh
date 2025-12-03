#!/bin/bash

#SBATCH -J heat_diffusion_profile_cpu
#SBATCH -o ./output/profile/cpu/%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH -p mi2104x

set -e

echo "Running: srun ./build/src/heat_diffusion $@ --kernel cpu --mode profile"
srun "./build/src/heat_diffusion" "$@" --kernel cpu --mode profile