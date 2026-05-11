#!/bin/bash

# Define the voxel sizes
VOXEL_SIZES=(75 100 175 200 250 300)

# Read folders from the text file
FOLDERS=()
while IFS= read -r line; do
    FOLDERS+=("$line")
done < "../dataset_folders/folders_TestRealData.txt"

# Read targets from the text file
TARGET_FILES=()
while IFS= read -r line; do
    TARGET_FILES+=("$line")
done < "../dataset_folders/Real_Testtargets.txt"

# Read estimated poses from the text file
ESTIMATED_POSES=()
while IFS= read -r line; do
    ESTIMATED_POSES+=("$line")
done < "../Estimated_poses_folder/teaser_estimated_poses.txt"

# Loop through each folder and voxel size
for target in "${TARGET_FILES[@]}"; do
    for folder in "${FOLDERS[@]}"; do
        for estimated_pose in "${ESTIMATED_POSES[@]}"; do
            for voxel in "${VOXEL_SIZES[@]}"; do
                    echo "Running with folder: $folder, voxel size: $voxel and noise std : $noise"
                    python3 ../gicp_or_icp.py \
                        --source "$folder" \
                        --target "$target" \
                        --estimated-pose "$estimated_pose" \
                        --refinement-voxel-size "$voxel" \
                        # --use-gicp \
                        # --viz True

                    # Rename the /metric folder
                    mv "$folder/metrics" \
                    "$folder/ICP/Teaser$(basename "$estimated_pose")/Voxel${voxel}"
            done
        done
    done
done
