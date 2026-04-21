#!/bin/bash

# Define the voxel sizes
VOXEL_SIZES=(100 125 150 175 200 225 250 300 450)
NOISE_STD=(0)

# Read folders from the text file
FOLDERS=()
while IFS= read -r line; do
    FOLDERS+=("$line")
done < "../dataset_folders/folders_TestRealData.txt"

# Read targets from the text file
TARGET_FILES=()
while IFS= read -r line; do
    TARGET_FILES+=("$line")
done < "../dataset_folders/Real_targets.txt"

# Loop through each folder and voxel size
for target in "${TARGET_FILES[@]}"; do
    for folder in "${FOLDERS[@]}"; do
        for voxel in "${VOXEL_SIZES[@]}"; do
            echo "Running with folder: $folder, voxel size: $voxel and noise std : $noise"
            python3 ../teaserpp_fpfh.py \
                --source "$folder" \
                --target "$target" \
                --voxel-size "$voxel" \
                --noise-std "$noise" \
                # --viz True

            # Rename the /metric folder
            mv "$folder/metrics" \
            "$folder/metrics_v${voxel}_$(basename "$target" .ply)"

            # Rename the /teaser_metrics folder
            mv "$folder/teaser_metrics" \
            "$folder/teaser_metrics_v${voxel}_$(basename "$target" .ply)"

            # Rename the /teaser_estimated_poses folder
            mv "$folder/teaser_estimated_poses" \
            "$folder/teaser_estimated_poses_v${voxel}_$(basename "$target" .ply)"
        done
    done
done
