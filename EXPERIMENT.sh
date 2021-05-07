#!/usr/bin/env bash




# run baseline results
echo "=== Running Baseline ==="
python3 train_baseline.py

echo "=== Train & Validation==="
# specify task: the options include task1a, task1b, task2a. The default choice is task1a
python3 main.py -task task1b

