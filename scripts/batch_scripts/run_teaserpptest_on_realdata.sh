#!/bin/bash

# Define the voxel sizes
VOXEL_SIZES=(100 125 150 175 225 250 275 300 450)
NOISE_STD=(0)

# Read folders from the text file
FOLDERS=()
while IFS= read -r line; do
    FOLDERS+=("$line")
done < "../data_folders/folders_TestRealData.txt"

# Read targets from the text file
TARGET_FILES=()
while IFS= read -r line; do
    TARGET_FILES+=("$line")
done < "../data_folders/Real_Testtargets.txt"

# Loop through each folder and voxel size
for target in "${TARGET_FILES[@]}"; do
    for folder in "${FOLDERS[@]}"; do
        for voxel in "${VOXEL_SIZES[@]}"; do
            for noise in "${NOISE_STD[@]}"; do
                echo "Running with folder: $folder, voxel size: $voxel and noise std : $noise"
                python3 teaserpp_fpfh_test.py \
                    --source "$folder" \
                    --target "$target" \
                    --voxel-size "$voxel" \
                    --noise-std "$noise" \
                    --use-gicp \
                    # --viz True

                # Rename the /metric folder
                mv "$folder/metrics" \
                "$folder/GICP/Voxel${voxel}"

                # Rename the /metric folder
                mv "$folder/teaser_metrics" \
                "$folder/Teaser/Voxel${voxel}"
            done
        done
    done
done
