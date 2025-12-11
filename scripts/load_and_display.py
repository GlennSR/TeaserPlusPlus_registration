#! /usr/bin/env python3

import argparse
import logging

import open3d as o3d

from registration.utils.logging import setup_logging
from registration.visualization.viewer import print_point_cloud_info
from registration.utils.transforms import (
    get_flip_transform,
)
from global_registration import rough_scale_point_cloud

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # add input file argument
    parser = argparse.ArgumentParser(
        description="Load and display a point cloud with its bounding boxes"
    )
    parser.add_argument("--input", type=str, help="Input file path", required=True)
    parser.add_argument(
        "--flip",
        type=str,
        choices=["x", "y", "z", "nx", "ny", "nz"],
        help="Flip axis (x, y, or z) by 90 degrees (prefix n for negative)",
        required=False,
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

    pcd = o3d.io.read_point_cloud(args.input)
    print_point_cloud_info(pcd, args.input)

    frame_scale = rough_scale_point_cloud(pcd)

    if args.flip:
        pcd.transform(get_flip_transform(args.flip))

    # apply random transformation
    # pcd.transform(
    #     rototranslation_from_rotation_translation(
    #         generate_random_rotation_matrix(), np.zeros(3)
    #     )
    # )

    # compute and display bounding boxes
    axis_aligned_bounding_box = pcd.get_axis_aligned_bounding_box()
    axis_aligned_bounding_box.color = (1, 0, 0)
    oriented_bounding_box = pcd.get_minimal_oriented_bounding_box()
    oriented_bounding_box.color = (0, 1, 0)

    logger.info(
        "Displaying axis_aligned_bounding_box in red and oriented bounding box in green ..."
    )
    mesh_frame_target = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=frame_scale, origin=[0, 0, 0]
    )
    pcd.paint_uniform_color([1, 0.706, 0])  # yellow
    o3d.visualization.draw_geometries(  # type: ignore
        [pcd, axis_aligned_bounding_box, oriented_bounding_box, mesh_frame_target]
    )
