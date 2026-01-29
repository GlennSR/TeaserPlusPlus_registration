# TEASER++

About
-----
This repository is based on https://github.com/MIT-SPARK/TEASER-plusplus to apply Teaser++ on a folder containing source point clouds in relation to a specified target point cloud.
It provides an example source and target dataset inside /test_data, where:
- source is produced by my ROS Simulation: https://github.com/GlennSR/ROS2-Lidar-Simulation
- target is produced using a CAD model and the following repository: https://github.com/simogasp/STEPToPoints

You'll find the main script `teaserpp_fpfh.py` alongside with others utility and visualization scripts

Dependencies
------------

First of all you must install Teaser++ following the `Installation` section on the Teaser++ repository. 
Don't forget to create a conda environment and install the dependencies.

- Python 3.12+
- numpy
- matplotlib (optional, for histograms)
- open3d (recommended >= 0.10)
- teaserpp-python 1.0.0+ — install following the TEASER++ docs for your platform.

You can install the Python dependencies with pip:

```bash
python3 -m pip install numpy matplotlib open3d
# For TEASER++ python bindings, follow the official instructions: https://github.com/MIT-SPARK/TEASER-plusplus
```

How to use `teaserpp_fpfh.py`
---------------------------
`teaserpp_fpfh.py` performs the following steps:

- Load source and target point clouds (PLY/PCD)
- Preprocess (voxel downsample + normals + FPFH)
- Find putative FPFH correspondences (K-NN + mutual filter)
- Run TEASER++ on the correspondences to get a robust global pose
- Run ICP point-to-plane refinement
- Optionally visualize intermediate and final results

Optional:
- Compute metrics and save them to `<source>_metrics.json` in a `metrics/` folder next to the source file

```bash
python3 teaserpp/scripts/teaserpp_fpfh.py \
  --source "<SOURCE_PATH>" \
  --target "<TARGET_PATH>" \
  --voxel-size 30 \
  --max_iter_icp 2000 \
  --noise-std 0.01 \
  --viz True \
  -v INFO
```

Options:

- `--source` Path to source point cloud (.ply or .pcd) (required)
- `--target` Path to target point cloud (required)
- `--voxel-size` Voxel size used for downsampling (default: 30)
- `--max_iter_icp` Max iterations for ICP (default: 2000)
- `--noise-std` Standard deviation of Gaussian noise added to source for testing (default: 0.0)
- `--viz` Enable Open3D visualization (True/False)
- `-v/--verbose` Logging level (DEBUG|INFO|WARNING...)

Minimal example
---------------
Run the script on a single pair (this is the example used in testing):

```bash
python3 teaserpp/scripts/teaserpp_fpfh.py \
  --source "../test_data/Clean/GT dataset/sameref/Source/ROS/without_objects/Ouster 32 Samples/ry_0degrees/z0_6m_128.ply" \
  --target "../test_data/Clean/GT dataset/sameref/Target/maquette300k.ply" \
  --voxel-size 30 \
  --viz True \
  -v INFO
```

Notes:
- If you enable `--viz True` the script will open Open3D visualizers and show images for each step of the registration
- Metrics are written into a JSON file next to your source file under a `metrics/` directory. The JSON includes the estimated transformation, fitness, inlier RMSE and other diagnostics.

![Initial State](https://github.com/GlennSR/TeaserPlusPlus_registration/tree/main/docs/gifs/initial_state.gif)

![Registration Result](https://github.com/GlennSR/TeaserPlusPlus_registration/tree/main/docs/gifs/registration_result.gif)

Troubleshooting & tips
----------------------
- Check Source and Target references: The script loads a JSON with the key `H` used as a homogeneous 4×4 matrix. Make sure the Source point cloud is aligned with the same axis reference than the Target;
- Using the ROS simulation to produce more datasets: If you decide to build your own dataset using my ROS Simulation https://github.com/GlennSR/ROS2-Lidar-Simulation so you'll have use `flip_and_scale_pc_folder.py` script to align the axis and scale the point clouds before using them in the teaser script. 
- TEASER++ bindings: The TEASER++ Python bindings may require compilation — if `import teaserpp_python` fails, follow TEASER++ build instructions for your platform.
