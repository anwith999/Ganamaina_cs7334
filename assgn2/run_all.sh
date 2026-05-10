#!/bin/bash

mkdir -p results

for n in 4 8 12 16
do
  for r in 1 2 3 4 5
  do
    echo "Running producer_consumer with $n processes, run $r"
    ibrun -n $n ./producer_consumer 120 > results/pc_${n}_${r}.txt
  done
done

for n in 4 8 12 16
do
  for r in 1 2 3 4 5
  do
    echo "Running workpool with $n processes, run $r"
    ibrun -n $n ./workpool 120 > results/wp_${n}_${r}.txt
  done
done
