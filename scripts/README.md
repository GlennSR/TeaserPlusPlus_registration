# Scripts

## flip_and_scale_pc.py

This script is used to flip and scale point clouds.
It can be used to preprocess point cloud to bring it in the reference frame of the visualizer.

### Usage

For example to flip a point cloud coming from the sampling of the maquette (stp file)

```bash
uv run ./scripts/flip_and_scale_pc.py --input data/maquette12k.ply --output data/sameref/maquette12k.ply --flip z
```

To flip a scan coming from the ROS simulator (flip + scale back to mm)

```bash
uv run ./scripts/flip_and_scale_pc.py --input data/y_-0.75m/pcl_out_time104-116000000.ply --output data/sameref/y_-0.75m.ply --scale 1000 --flip nx
```

## load_and_display.py

This script loads a point cloud from file and displays it using the Open3D visualizer.

## global_registration.py

This script performs global registration between two point clouds using RANSAC followed by ICP refinement.

- it generates a random initial transformation to be applied to the source point cloud
- it then tries to correct the rotation around the gravity axis (y-axis) using a simple heuristic
- aligns the centers of the two point clouds to improve convergence
- performs RANSAC global registration checking that
  - the fitness (ratio of inliers) is above a threshold `--min-fitness`
  - the solution is physically plausible (the gravity axis is not flipped)
- refines the result with ICP

Example usage:

```bash
# align scan to maquette E Shape
uv run ./scripts/global_registration.py --source data/sameref/ry_0_degres.ply --target  data/sameref/E_shape_maq15k.ply --max_iter_icp 100000 --voxel-size 40 --min-fitness 0.35

# align scan to maquette full
uv run ./scripts/global_registration.py --source data/sameref/ry_0_degres.ply --target  data/sameref/maquette27k.ply --max_iter_icp 100000 --voxel-size 30 --min-fitness 0.4
uv run ./scripts/global_registration.py --source data/sameref/ry_0_degres.ply --target  data/sameref/maquette27k.ply --max_iter_icp 100000 --voxel-size 30 --min-fitness 0.53
```

# Teaser++

1. Summary - What TEASER++ Optimizes:
- First (Translation): Maximize number of inliers using adaptive voting/GNC-TLS
- Second (Rotation): Find Maximum Clique of geometrically consistent correspondences
- Third (Refinement): Minimize truncated least squares cost using GNC