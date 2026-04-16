#!/bin/bash

# Read folders from the text file
FOLDERS=()
while IFS= read -r line; do
    FOLDERS+=("$line")
done < "./result_folders/result_RealData.txt"

# Loop through each folder and voxel size
for folder in "${FOLDERS[@]}"; do
    echo "Running with folder: $folder"
    python3 create_metric_graphs.py \
    --input "$folder" \
    --output_path graphs \
    # --long_data True # FOR FOLDERS WITH MANY METRIC FILES
done
