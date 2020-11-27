#!/bin/bash

#SBATCH --job-name=quad-figure3
#SBATCH --ntasks=1
#SBATCH --partition=htc

# Load the default OpenMPI module.
export MODULEPATH=$MODULEPATH:/mnt/exports/data/module_files
module load mpi/openmpi
module load gcc-9.2.0
module load gsl-2.4
module load fftw-2.1.5

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Run the mpi program with mpirun. The -n flag is not required;
# mpirun will automatically figure out the best configuration from the
# Slurm environment variables.
# mpirun ./N-GenIC planck1_10_128.param 
python quad-figure3.py

