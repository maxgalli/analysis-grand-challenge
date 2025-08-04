#!/bin/bash

# Define parameter values
workers=(1 2 4 5)
files_per_sample=(1 2 5)
chunksizes=(50000 200000 500000 1500000)

# Iterate over all combinations
for w in "${workers[@]}"; do
    for n in "${files_per_sample[@]}"; do
        for c in "${chunksizes[@]}"; do
            echo "Running: python for_testing.py -w $w -n $n -c $c --force --local"
            python for_testing.py -w "$w" -n "$n" -c "$c" --force --local
        done
    done
done
