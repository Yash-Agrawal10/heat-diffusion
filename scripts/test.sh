#!/bin/bash

#SBATCH -J heat_diffusion_test
#SBATCH -o ./output/test/%j.out
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 00:10:00
#SBATCH -p mi2104x

set -e

cd ./build
ctest