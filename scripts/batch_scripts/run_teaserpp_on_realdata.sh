#!/bin/bash

# Define the voxel sizes
VOXEL_SIZES=(0.3 0.35 0.45 0.5 0.55 0.6)
NOISE_STD=(0)

# Read folders from the text file
FOLDERS=()
while IFS= read -r line; do
    FOLDERS+=("$line")
done < "../dataset_folders/Nio_Dataset.txt"

# Read targets from the text file
TARGET_FILES=()
while IFS= read -r line; do
    TARGET_FILES+=("$line")
done < "../dataset_folders/Real_Nio_target.txt"

# Loop through each folder and voxel size
for target in "${TARGET_FILES[@]}"; do
    for folder in "${FOLDERS[@]}"; do
        for voxel in "${VOXEL_SIZES[@]}"; do
            echo "Running with folder: $folder, voxel size: $voxel and noise std : $noise"
            python3 ../teaserpp_fpfh.py \
                --source "$folder" \
                --target "$target" \
                --voxel-size "$voxel" \
                --refine-registration \
                --refinement-voxel-size 0.25 \
                --use-gicp \
                # --viz True

            # Compute voxel*1000 as integer (handles float voxel values)
            VS_K=$(awk -v v="$voxel" 'BEGIN{printf("%.0f", v*1000)}')
            # Rename the /metric folder
            mv "$folder/metrics_${VS_K}_250" \
            "$folder/metrics_v${voxel}_$(basename "$target" .ply)"

            # Rename the /teaser_metrics folder
            mv "$folder/teaser_metrics_${VS_K}" \
            "$folder/teaser_metrics_v${VS_K}_$(basename "$target" .ply)"

            # Rename the /teaser_estimated_poses folder
            mv "$folder/teaser_estimated_poses" \
            "$folder/teaser_estimated_poses_v${VS_K}_$(basename "$target" .ply)"
        done
    done
done
