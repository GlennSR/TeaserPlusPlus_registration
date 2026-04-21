#!/bin/bash

# Define the voxel sizes
VOXEL_SIZES=(30 40 50 60 70 80 90 100 110 120 130)
NOISE_STD=(0.0)

# Read folders from the text file
FOLDERS=()
while IFS= read -r line; do
    FOLDERS+=("$line")
done < "./data_folders/folders_WithoutMidWall.txt"

# Loop through each folder and voxel size
for folder in "${FOLDERS[@]}"; do
    for voxel in "${VOXEL_SIZES[@]}"; do
        for noise in "${NOISE_STD[@]}"; do
            echo "Running with folder: $folder, voxel size: $voxel and noise std : $noise"
            python3 teaserpp_fpfh.py \
                --source "../test_data/Simulated/GT_dataset/sameref/Source/ROS/$folder" \
                --target "../test_data/Simulated/GT_dataset/sameref/Target/maquette27k.ply" \
                --voxel-size "$voxel" \
                --noise-std "$noise" 

            # Extract degrees from the folder path (e.g., rz_0deg, rz_-45deg, rz_0degrees, rz_-45_degres)
            if [[ $folder == *"rz_"* ]]; then
                # Capture optional sign and digits after 'rz_'. Accept common suffixes like deg/degrees/degres.
                # Examples matched: rz_0deg -> 0, rz_-45deg -> -45, rz_30_degrees -> 30
                degrees_num=$(echo "$folder" | grep -oP 'rz_\K-?[0-9]+' || true)
                if [[ -n "$degrees_num" ]]; then
                    degrees="${degrees_num}deg"
                else
                    degrees="unknown"
                fi
            else
                degrees="unknown"
            fi

            # Rename the /metric folder
            mv "../test_data/Simulated/GT_dataset/sameref/Source/ROS/$folder/metrics" \
            "../test_data/Simulated/GT_dataset/sameref/Source/ROS/$folder/rz_${degrees}_metrics_v${voxel}_noise${noise}"
        done
    done
done
