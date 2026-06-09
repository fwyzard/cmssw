#!/bin/bash

# Check if the name of the results directory has been provided:
if [ $# -lt 1 ]; then
    echo "ERROR: no directory name provided."
    echo "Usage: $0 <directory_name>"
    exit 1
fi

RES_DIR="$1"
mkdir -p "$RES_DIR"

# ===== Pull Comparison =====
echo "===== Running Pull Comparison ====="
./comparePull.py
mv -f ComparePull "$RES_DIR"

# ===== Chi2 Comparison =====
echo "===== Running Chi2 Comparison ====="
./compareChi2.py
mv -f CompareChi2 "$RES_DIR"

# ===== 2D Maps Comparison =====
echo "===== Running 2D maps Comparison ====="
./compare2Dmaps.py
mv -f Compare2Dmaps "$RES_DIR"

# ADD HERE SOMETHING ELSE 