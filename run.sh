#!/bin/bash

time mpirun -np 1 ./out $1 $2
#cat $2 | head
#echo "middle"
#cat $2 | tail 
