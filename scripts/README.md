# Scripts

## flip_and_scale_pc_folder.py

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

## teaserpp_fpfh.py

This script performs robust point cloud registration using the TEASER++ and then apply ICP refinement.

1. **Feature-based correspondence matching** using FPFH features
2. **Mutual filtering** to remove ambiguous matches
3. **TEASER++ global registration** with built-in outlier rejection (MCIS + GNC-TLS)
4. **ICP refinement** for fine alignment
5. **Error computation** against ground truth (if available)

### Usage

First you need to activate the conda environment

```bash
conda activate teaserpp
```

#### Basic Usage (Single File)

```bash
# Register a single point cloud pair
python3 teaserpp_fpfh.py --source <source.ply> --target <target.ply> --voxel-size 30 --viz True
```

#### Batch Processing (Directory)

```bash
# Register all point clouds in a directory against a single target
python3 teaserpp_fpfh.py --source <source_directory/> --target <target.ply> --voxel-size 30 --viz True
```

#### Full Example with Parameters

```bash
python teaserpp_fpfh.py \
    --source ../test_data/GT\ dataset/sameref/ROS/without_objects/Ouster\ 32\ Samples/ry_0degrees/ \
    --target ../test_data/GT\ dataset/sameref/Target/maquette300k.ply \
    --voxel-size 50 \
    --max_iter_icp 2000 \
    --viz True \
    -v INFO
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--source` | str | **required** | Path to source point cloud file (.ply or .pcd) or directory |
| `--target` | str | **required** | Path to target point cloud file (.ply or .pcd) |
| `--voxel-size` | float | `0.05` | Voxel size for downsampling (smaller = more accurate but slower) |
| `--max_iter_icp` | int | `2000` | Maximum iterations for ICP refinement |
| `--viz` | bool | `False` | Enable Open3D visualization windows |
| `-v, --verbose` | str | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL |

1. Summary - What TEASER++ Optimizes:
- First (Translation): Maximize number of inliers using adaptive voting/GNC-TLS
- Second (Rotation): Find Maximum Clique of geometrically consistent correspondences
- Third (Refinement): Minimize truncated least squares cost using GNC