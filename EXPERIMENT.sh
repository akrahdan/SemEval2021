#!/usr/bin/env bash




# run baseline results
echo "=== Running Baseline ==="
python3 train_baseline.py

echo "=== Train & Validation==="
python3 humor_model.py

