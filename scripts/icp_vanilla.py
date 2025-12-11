#! /usr/bin/env python3

import argparse
import logging

import numpy as np
import open3d as o3d

from registration.utils.logging import setup_logging
from registration.visualization.viewer import draw_registration_result

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # tutorial from here https: // www.open3d.org / docs / latest / tutorial / pipelines / icp_registration.html  # Point-to-plane-ICP

    # add input file argument
    argparse = argparse.ArgumentParser(description="Simple ICP example")
    argparse.add_argument("--source", type=str, help="source file path", required=True)
    argparse.add_argument("--target", type=str, help="target file path", required=True)
    argparse.add_argument(
        "--threshold", type=float, help="threshold for ICP", default=0.02
    )
    argparse.add_argument(
        "--max_iter_icp",
        type=int,
        help="max number of iterations for ICP",
        default=2000,
    )
    argparse.add_argument(
        "-v",
        "--verbose",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: WARNING)",
    )

    args = argparse.parse_args()

    # Set logging level based on user selection
    setup_logging(getattr(logging, args.verbose))

    threshold = args.threshold
    source = o3d.io.read_point_cloud(args.source)
    target = o3d.io.read_point_cloud(args.target)

    # compute the bounding of the source point cloud
    source_bb = source.get_axis_aligned_bounding_box()
    logger.info(f"Bounding box of source point cloud: {source}")
    logger.info(f"{source_bb}")
    target_bb = target.get_axis_aligned_bounding_box()
    logger.info(f"Bounding box of target point cloud: {target}")
    logger.info(f"{target_bb}")

    trans_init = np.asarray(
        [
            [0.862, 0.011, -0.507, 0.05],
            [-0.139, 0.967, -0.215, 0.07],
            [0.487, 0.255, 0.835, -0.0004],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    # initialize the matrix to the 4x4 identity matrix
    # trans_init = np.identity(4)
    draw_registration_result(source, target, trans_init, "Initial settings")

    logger.info("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init
    )
    logger.info(f"{evaluation}")

    logger.info("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=args.max_iter_icp
        ),
    )
    logger.info(f"{reg_p2p}")
    logger.info("Transformation is:")
    logger.info(f"{reg_p2p.transformation}")
    draw_registration_result(
        source, target, reg_p2p.transformation, "ICP point-to-point"
    )

    # check if the target point cloud has normals
    if not target.has_normals():
        logger.info("Target point cloud does not have normals, estimating them...")
        # estimate the normals of the target point cloud
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

    logger.info("Apply point-to-plane ICP")
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    logger.info(f"{reg_p2l}")
    logger.info("Transformation is:")
    print(f"{reg_p2l.transformation}")
    draw_registration_result(
        source, target, reg_p2l.transformation, "ICP point-to-plane"
    )
