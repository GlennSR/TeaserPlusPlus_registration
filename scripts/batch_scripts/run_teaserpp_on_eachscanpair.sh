#!/bin/bash

# Define the voxel sizes
VOXEL_SIZES=(50 150 300 450)
refinement_voxel_size=150
NOISE_STD=(0)

# Read targets from the text file
TARGETS=(19 29 39 49 59 69 79 89 99 109 119 129 139 149 159 169 179 189 199 209 219 229 239 249 259 269 279 289 299 309 319 329 339 349 359 369 379)
STEP=9


# Loop through each folder and voxel size
for target in "${TARGETS[@]}"; do
    source=$((target - STEP))
    for voxel in "${VOXEL_SIZES[@]}"; do
        echo "Running with folder: $folder, voxel size: $voxel and noise std : $noise"
        python3 ../teaserpp_fpfh_test.py \
            --source "../../test_data/Real_Lidar/GT_dataset/Source/scans/$source.ply" \
            --target "../../test_data/Real_Lidar/GT_dataset/Source/scans/$target.ply" \
            --voxel-size "$voxel" \
            --refine-registration \
            --refinement-voxel-size "$refinement_voxel_size" \
            --use-gicp \
            --viz True

        # Rename the /metric folder
        cp -rl "/home/gro5293/pcl_registration/teaserpp/test_data/Real_Lidar/GT_dataset/Source/scans/metrics" \
        "/home/gro5293/pcl_registration/teaserpp/test_data/Real_Lidar/GT_dataset/Source/Teaser_to_pair_of_scans/Step20/metrics_v${voxel}"
        rm -r "/home/gro5293/pcl_registration/teaserpp/test_data/Real_Lidar/GT_dataset/Source/scans/metrics"
    done
done
