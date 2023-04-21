#!/bin/bash

# Request resources:
#SBATCH -c 1           # 1 CPU core
#SBATCH --mem=30G       # memory required, up to 250G on standard nodes.
#SBATCH --gres=tmp:50G  # temporary disk space required on the compute node ($TMPDIR), up to 400G
#SBATCH --time=0:5:0   # time limit for job (format:  days-hours:minutes:seconds)
#SBATCH --job-name="cotjob"
#SBATCH -o cpujobtest.out
#SBATCH -e cpujobtest.err
#SBATCH -p test
source /etc/profile.d/modules.sh

module load gcc

g++ -O0  -o cputest cputest_number_crunching.cpp
echo -O0: 
./cputest 100000



g++ -O1  -o cputest cputest_number_crunching.cpp
echo -O1: 
./cputest 100000

