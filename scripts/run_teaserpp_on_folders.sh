#!/bin/bash

# Define the voxel sizes
VOXEL_SIZES=(15 30 50)
NOISE_STD=(0.01 0.025 0.05)

# Read folders from the text file
FOLDERS=()
while IFS= read -r line; do
    FOLDERS+=("$line")
done < "folders.txt"

# Loop through each folder and voxel size
for folder in "${FOLDERS[@]}"; do
    for voxel in "${VOXEL_SIZES[@]}"; do
        for noise in "${NOISE_STD[@]}"; do
            echo "Running with folder: $folder, voxel size: $voxel and noise std : $noise"
            python3 teaserpp_fpfh.py \
                --source "../test_data/Clean/GT dataset/sameref/Source/ROS/$folder" \
                --target "../test_data/Clean/GT dataset/sameref/Target/maquette300k.ply" \
                --voxel-size "$voxel" \
                --noise-std "$noise"

            # Extract degrees from the folder path (e.g., ry_0degrees)
            if [[ $folder == *"ry_"* ]]; then
                degrees=$(echo "$folder" | grep -oP 'ry_\K[0-9]+deg')
            else
                degrees="unknown"
            fi

            # Rename the /metric folder
            mv "../test_data/Clean/GT dataset/sameref/Source/ROS/$folder/metrics" \
            "../test_data/Clean/GT dataset/sameref/Source/ROS/$folder/ry_${degrees}_metrics_v${voxel}_noise${noise}"
        done
    done
done
