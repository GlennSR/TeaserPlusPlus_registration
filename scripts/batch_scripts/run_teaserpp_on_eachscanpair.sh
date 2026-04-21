#!/bin/bash

# Define the voxel sizes
VOXEL_SIZES=(50 150 450)
NOISE_STD=(0)

# Read targets from the text file
TARGETS=(9 19 29 39 49 59 69 79 89 99 109 119 129 139 149 159 169 179 189 199 209 219 229 239 249 259 269 279 289 299 309 319 329 339 349 359 369 379)
STEP=9


# Loop through each folder and voxel size
for target in "${TARGETS[@]}"; do
        source=$((target - STEP))
        for voxel in "${VOXEL_SIZES[@]}"; do
            for noise in "${NOISE_STD[@]}"; do
                echo "Running with folder: $folder, voxel size: $voxel and noise std : $noise"
                python3 ../teaserpp_fpfh.py \
                    --source "../../test_data/Real_Lidar/GT_dataset/Source/scans/$source.ply" \
                    --target "../../test_data/Real_Lidar/GT_dataset/Source/scans/$target.ply" \
                    --voxel-size "$voxel" \
                    --noise-std "$noise" \
                    # --viz True

                # Rename the /metric folder
                cp -rl "/home/gro5293/pcl_registration/teaserpp/test_data/Real_Lidar/GT_dataset/Source/metrics" \
                "/home/gro5293/pcl_registration/teaserpp/test_data/Real_Lidar/GT_dataset/Source/Teaser_to_pair_of_scans/Step10/metrics_v${voxel}"
                rm -r "/home/gro5293/pcl_registration/teaserpp/test_data/Real_Lidar/GT_dataset/Source/metrics"
            done
        done
done
