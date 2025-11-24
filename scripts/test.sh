#!/bin/bash

#SBATCH -J test
#SBATCH -o ./output/test/%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:10:00

cd ./build
ctest