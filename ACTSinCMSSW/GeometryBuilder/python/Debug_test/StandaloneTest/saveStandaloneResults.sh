#!/bin/bash

# --- List of parameters to extract ---
params=("phi_res" "phi_pull" "qOverP_res" "qOverP_pull" "theta_res" "theta_pull")

# --- Input file ---
input_file="results_Bfixed.txt"

# --- Loop over parameters ---
for param in "${params[@]}"; do
    # Output file name
    output_file="${param}.txt"
    
    # Use grep to filter lines and awk to extract the number
    grep "$param" "$input_file" | awk -F"$param: " '{print $2}' > "$output_file"
    
    echo "Extracted $param -> $output_file"
done

