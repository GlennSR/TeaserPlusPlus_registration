#!/bin/bash

# Define the voxel sizes
VOXEL_SIZES=(100 150 200 250 275 300 325 350 400)
REF_VOXEL_SIZES=(100 150 200 250 300)

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
            for ref_voxel_size in "${REF_VOXEL_SIZES[@]}"; do
                echo "Running with folder: $folder, voxel size: $voxel and noise std : $noise"
                python3 ../teaserpp_fpfh_test.py \
                    --source "$folder" \
                    --target "$target" \
                    --voxel-size "$voxel" \
                    --refine-registration \
                    --refinement-voxel-size "$ref_voxel_size" \
                    --use-gicp \
                    -o "$folder"/results_target1/TeaserPP/ \
                    # --viz True
            done
        done
    done
done
