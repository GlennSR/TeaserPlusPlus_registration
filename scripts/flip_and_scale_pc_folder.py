#! /usr/bin/env python3

import argparse
import logging
import os

import open3d as o3d
import numpy as np

from registration.utils.logging import setup_logging
from registration.visualization.viewer import print_point_cloud_info

from registration.utils.transforms import get_flip_transform

logger = logging.getLogger(__name__)

'''
This script applies a flip transformation and scaling to PLY point clouds 
inside a folder and saves them.
Usage:
    python flip_and_scale_pc_folder.py --input <input_folder> --output <output_folder> [--flip <axis>] [--scale <factor>] [--verbose <level>]

Arguments:
    --input: Path to the input folder containing PLY point clouds.
    --output: Path to the output folder to save transformed point clouds.
    --flip: Axis to flip around ('x', 'y', 'z', 'nx', 'ny', 'nz').
    --scale: Scaling factor (float).
    --verbose: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

For example to flip a point cloud coming from the sampling of the maquette (stp file)
    uv run ./scripts/flip_and_scale_pc.py --input data/maquette12k.ply --output data/sameref/maquette12k.ply --flip z

To flip a scan coming from the ROS simulator (flip + scale back to mm)
    uv run ./scripts/flip_and_scale_pc.py --input data/y_-0.75m/pcl_out_time104-116000000.ply --output data/sameref/y_-0.75m.ply --scale 1000 --flip nx
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply a scale wrt the origin and a flip transformation to all the ply point clouds in a folder"
    )
    parser.add_argument("--input", type=str, help="Input folder path", required=True)
    parser.add_argument("--output", type=str, help="Output folder path", required=True)
    parser.add_argument(
        "--flip",
        type=str,
        choices=["x", "y", "z", "nx", "ny", "nz"],
        help="Flip axis (x, y, or z) by 90 degrees (prefix n for negative)",
        required=False,
    )
    parser.add_argument(
        "--scale", type=float, help="Scale factor (default: 1.0)", required=False
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: WARNING)",
    )
    args = parser.parse_args()

    # Set logging level based on user selection
    setup_logging(getattr(logging, args.verbose))

    for pcd_file in os.listdir(args.input):
        if not pcd_file.endswith(".ply"):
            continue

        input_path = os.path.join(args.input, pcd_file)
        output_path = os.path.join(args.output, pcd_file)

        logger.info(f"Processing file: {input_path}")
        pcd = o3d.io.read_point_cloud(input_path)
        # o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")  # type: ignore
        print_point_cloud_info(pcd, input_path)

        if args.scale:
            logger.info(f"Scaling point cloud by factor {args.scale} wrt the origin")
            pcd.scale(args.scale, np.zeros(3))

        if args.flip:
            logger.info(f"Flipping point cloud along {args.flip} axis by 90 degrees")
            pcd.transform(get_flip_transform(args.flip))

        logger.info(f"Saving transformed point cloud to {output_path}")
        o3d.io.write_point_cloud(output_path, pcd)
