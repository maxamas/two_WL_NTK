#!/bin/bash
#
#SBATCH -p ALL
#SBATCH -c 1 # number of cores
for i in {1..100000}; do
echo $RANDOM >> SomeRandomNumbers.txt
donesort SomeRandomNumbers.txt