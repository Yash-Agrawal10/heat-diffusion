#!/bin/bash

#SBATCH -J heat_diffusion_profile_fast
#SBATCH -o ./output/profile/fast/%j.out
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 00:10:00
#SBATCH -p mi2104x

set -e

echo "Running: srun ./build/src/heat_diffusion_profile $@ --mode fast"
srun "./build/src/heat_diffusion_profile" "$@" --mode fast