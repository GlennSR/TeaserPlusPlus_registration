#!/bin/bash

# Define the voxel sizes
VOXEL_SIZES=(100 125 150 175 225 250 275 300 450)
NOISE_STD=(0)

# Loop through each folder and voxel size
for voxel in "${VOXEL_SIZES[@]}"; do
    for noise in "${NOISE_STD[@]}"; do
        echo "Running with folder: ../test_data/Real_Lidar/GT_dataset/Source/scans, voxel size: $voxel and noise std : $noise"
        python3 teaserpp_fpfh_test.py \
            --source "../test_data/Real_Lidar/GT_dataset/Source/scans/" \
            --target "../test_data/target/complete_map.ply" \
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
