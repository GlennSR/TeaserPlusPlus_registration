#!/bin/bash

# Read folders from the text file
FOLDERS=()
while IFS= read -r line; do
    FOLDERS+=("$line")
done < "/home/gro5293/pcl_registration/nio-sdk/Dataset/Week_22_06/2/Horizontal/2wayTrip_1/metrics_self/partial_map/step10/TeaserPP/result_folders_paths.txt"

# Loop through each folder and voxel size
for folder in "${FOLDERS[@]}"; do
    echo "Running with folder: $folder"
    python3 ../graphical/plot_metric_graphs.py \
    --input "$folder" \
    --output graphs \
    --criteria 1 \
    -tf 0.78 \
    -tr 1.5 \
    -tt 100 \
    --long_data # FOR FOLDERS WITH MANY METRIC FILES 
done
