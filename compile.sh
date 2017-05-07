#!/bin/bash

# module laods
module load compiler/cuda/7.5/compilervars
module load compiler/gcc/4.9.3/compilervars
module load mpi/mpich/3.1.4/gcc/mpivars
module load apps/lammps/gpu
module load mpi/openmpi/1.10.0/gcc/mpivars

# compile
nvcc -I/home/soft/mpich-3.1.4/include/ -L/home/soft/mpich-3.1.4/lib/ -lmpi -O3 a4_mpi.cu -o out
echo "done compiling"
