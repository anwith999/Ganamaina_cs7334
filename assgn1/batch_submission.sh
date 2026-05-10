#!/bin/bash
#SBATCH -A ASC23013
#SBATCH -J assgn1_pi
#SBATCH -o assgn1_pi.%j.out
#SBATCH -e assgn1_pi.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=56
#SBATCH -p development
#SBATCH -t 00:10:00

module reset
module load gcc

gcc -O3 -fopenmp -o compute_pi compute_pi.c

# Task 1: hardware info
lstopo --no-io > hwloc.txt

N=10000000
MAXT=$SLURM_CPUS_PER_TASK

echo "places,bind,threads,schedule,time" > results.csv

# 2(a) placement tests at MAX threads
for places in core socket; do
  for bind in close spread; do
    export OMP_PLACES=$places
    export OMP_PROC_BIND=$bind
    export OMP_NUM_THREADS=$MAXT
    export OMP_SCHEDULE="static,1"
    t=$(./compute_pi $N | awk '/Time/ {print $3}')
    echo "$places,$bind,$MAXT,NA,$t" >> results.csv
  done
done

# TEMP default for scaling/scheduling; later set to best combo from above
export OMP_PLACES=core
export OMP_PROC_BIND=close

# 2(c)(i) scaling
for t in 1 2 4 8 16 32 56; do
  if [ $t -le $MAXT ]; then
    export OMP_NUM_THREADS=$t
    export OMP_SCHEDULE="static,1"
    tt=$(./compute_pi $N | awk '/Time/ {print $3}')
    echo "$OMP_PLACES,$OMP_PROC_BIND,$t,static-1,$tt" >> results.csv
  fi
done

# 2(c)(ii) scheduling at MAX threads
export OMP_NUM_THREADS=$MAXT
for policy in static dynamic; do
  for chunk in 10 100 1000; do
    export OMP_SCHEDULE="${policy},${chunk}"
    tt=$(./compute_pi $N | awk '/Time/ {print $3}')
    echo "$OMP_PLACES,$OMP_PROC_BIND,$MAXT,${policy}-${chunk},$tt" >> results.csv
  done
done

echo "DONE. Created hwloc.txt and results.csv"
