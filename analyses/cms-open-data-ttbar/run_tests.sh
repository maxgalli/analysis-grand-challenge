#!/bin/bash

# Define parameter values
workers=(1 2 5)
files_per_sample=(1 2 5)
chunksizes=(50000 200000)

# Iterate over all combinations
for w in "${workers[@]}"; do
    for n in "${files_per_sample[@]}"; do
        for c in "${chunksizes[@]}"; do
            echo "Running: python for_testing.py -w $w -n $n -c $c --force"
            python for_testing.py -w "$w" -n "$n" -c "$c" --force
        done
    done
done
