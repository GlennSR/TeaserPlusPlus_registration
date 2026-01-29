# TEASER++

About
-----
This repository is based on https://github.com/MIT-SPARK/TEASER-plusplus to apply Teaser++ on a folder containing source point clouds in relation to a specified target point cloud.
It provides an example source and target dataset inside `/test_data`, where:
- source is produced by the ROS Simulation: https://github.com/GlennSR/ROS2-Lidar-Simulation [1]
- target is produced using a CAD model and the following repository: https://github.com/simogasp/STEPToPoints

You'll find the main script `teaserpp_fpfh.py` alongside with others utility and visualization scripts

Notes:
- The source and the target point clouds can be any .ply file coming from a simulation, the real LiDAR, a converted CAD Model or any other

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
Inside the `scripts/` folder run the script below in a terminal with the conda environment activated:

```bash
python3 teaserpp_fpfh.py \
  --source ../test_data/Clean/GT\ dataset/sameref/Source/ROS/without_objects/Ouster\ 32\ Samples/ry_0degrees/z0_6m_128.ply \
  --target ../test_data/Clean/GT\ dataset/sameref/Target/maquette300k.ply \
  --voxel-size 30 \
  --viz True \
```

Yous should see a terminal output like this:
```bash
[2026-01-29 15:01:12][INFO] Preprocessing source point cloud
[2026-01-29 15:01:12][INFO] Point Cloud 'Downsampled source':
[2026-01-29 15:01:12][INFO] 	Number of points: 12920
[2026-01-29 15:01:12][INFO] 	Has normals: True
[2026-01-29 15:01:12][INFO] 	Point cloud size: [5503.0048  1776.68007 4649.8234 ]
[2026-01-29 15:01:12][INFO] 	Axis-Aligned Bounding Box: min [-2655.1063   -619.46377 -2324.8818 ], max [2847.8985 1157.2163 2324.9416]
[2026-01-29 15:01:12][INFO] 	Oriented Bounding Box: center [ 702.09106278  267.69939967 -249.96480846], extent [4447.38621495 5348.9238097  1781.94086072]
[2026-01-29 15:01:12][INFO] Feature of SOURCE: [[7.94221329e+01 9.80976827e+00 1.30086254e+01 ... 3.99805538e+00
  1.02165916e+01 4.68525141e+00]
 [3.35545564e+01 0.00000000e+00 3.04694011e-02 ... 8.84128438e-02
  8.31030725e-02 1.05119220e-01]
 [1.56727198e-01 2.02696040e-01 8.62852744e-02 ... 2.80253964e+00
  3.17989718e+00 3.35612068e-01]
 ...
 [4.83304946e+01 9.93038830e+00 1.03113611e+01 ... 1.25987255e+01
  3.29618518e+01 4.09985849e+01]
 [3.43914055e+01 8.37976299e+00 7.88855384e+00 ... 1.14326480e+01
  2.64105920e+01 4.45137520e+01]
 [3.13273085e+01 1.49751695e+01 7.26848187e+00 ... 1.61624630e+01
  1.92644769e+01 4.61000755e+01]]
[2026-01-29 15:01:12][INFO] Preprocessing target point cloud
[2026-01-29 15:01:12][INFO] Point Cloud 'Downsampled target':
[2026-01-29 15:01:12][INFO] 	Number of points: 51076
[2026-01-29 15:01:12][INFO] 	Has normals: True
[2026-01-29 15:01:12][INFO] 	Point cloud size: [1403.03479714 1858.96645551 4431.33333333]
[2026-01-29 15:01:12][INFO] 	Axis-Aligned Bounding Box: min [ -708.91578641  -368.02837295 -2212.0024283 ], max [ 694.11901073 1490.93808255 2219.33090503]
[2026-01-29 15:01:12][INFO] 	Oriented Bounding Box: center [-25.09615518 545.20344737   3.86842764], extent [4431.7349524  1411.28516852 1855.14674514]
[2026-01-29 15:01:12][INFO] Feature of TARGET: [[57.53741564  5.48341859 41.54119536 ...  8.07797614 26.80975093
   5.28299749]
 [47.18687486  1.72157838 22.23569786 ...  5.37162686 22.75916392
   4.93196066]
 [48.29441951  1.63089068 22.17505389 ...  7.35573666 22.10358487
   5.24293186]
 ...
 [85.1062642   0.20820489  9.24304987 ...  5.74099859  4.32589742
   0.19501922]
 [31.45912472  0.36431868 43.964676   ... 15.44968948 26.06820043
  10.86561083]
 [82.11030681  0.20820489  9.24304987 ...  5.74099859  4.32589742
   0.19501922]]
[2026-01-29 15:01:16][INFO] FPFH generates 1012 putative correspondences.
Starting scale solver (only selecting inliers if scale estimation has been disabled).
Scale estimation complete.
Max core number: 63
Num vertices: 1013
Max Clique of scale estimation inliers: 
Using chain graph for GNC rotation.
Starting rotation solver.
GNC rotation estimation noise bound:120
GNC rotation estimation noise bound squared:14400
GNC-TLS solver terminated due to cost convergence.
Cost diff: 0
Iterations: 13
Rotation estimation complete.
Starting translation solver.
Translation estimation complete.
[2026-01-29 15:01:17][INFO] Point-to-plane ICP registration is applied on original point clouds
[2026-01-29 15:01:17][INFO] to refine the alignment. This time we use a strict distance threshold 60.000
[2026-01-29 15:01:17][INFO] Elapsed time for TEASER++ Registration: 5.3668 seconds
[2026-01-29 15:01:18][INFO] Mean Open3D distance for the registration result (full cloud): 523.207821
[2026-01-29 15:01:18][INFO] Standard deviation of distances after registration (full cloud): 561.324080
[2026-01-29 15:01:20][INFO] TEASER++ Internal Metrics:
[2026-01-29 15:01:20][INFO]   Translation inliers: 2 / 1012 (0.2%)
[2026-01-29 15:01:20][INFO]   Rotation inliers (max clique): 17 / 1012 (1.7%)
[2026-01-29 15:01:20][INFO]   Solution valid: N/A
[2026-01-29 15:01:20][INFO] Open3D Evaluation Metrics:
[2026-01-29 15:01:20][INFO]   Fitness: 0.8890 (fraction of inlier points)
[2026-01-29 15:01:20][INFO]   Inlier RMSE: 6.3583 mm (lower is better)
[2026-01-29 15:01:20][INFO]   Correspondence set size: 84239
[2026-01-29 15:01:20][INFO] Inlier distances mean=3.6029 mm
[2026-01-29 15:01:20][INFO] ICP refinement result: RegistrationResult with fitness=8.890190e-01, inlier_rmse=6.358326e+00, and correspondence_set size of 84239
Access transformation to get result.
[2026-01-29 15:01:20][INFO] Estimated matrix:
[[ 9.99999888e-01  1.92741369e-04  4.32497927e-04  5.00296414e+02]
 [-1.92638504e-04  9.99999953e-01 -2.37869275e-04  2.29732250e+02]
 [-4.32543754e-04  2.37785933e-04  9.99999878e-01 -4.99985765e+02]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
[2026-01-29 15:01:20][INFO] Result fitness: 0.8890190491266952, inlier RMSE: 6.358326106797943 mm
[2026-01-29 15:01:20][INFO] Target point cloud diagonal length: 5026.320 mm
[2026-01-29 15:01:20][INFO] ICP inlier RMSE as percentage of target diagonal length: 0.1265 %
[2026-01-29 15:01:20][INFO] Source Ground Truth transform: 
[[   1.    0.   -0.  500.]
 [   0.    1.    0.  230.]
 [   0.    0.    1. -500.]
 [   0.    0.    0.    1.]]
[2026-01-29 15:01:20][INFO] Rotation error (radians): 0.0005 (degrees: 0.0304), Translation error: 0.5784
[2026-01-29 15:01:20][INFO] Registration RMSE: 0.6026211583867066
[2026-01-29 15:01:20][INFO] Saved metrics to ../test_data/Clean/GT dataset/sameref/Source/ROS/without_objects/Ouster 32 Samples/ry_0degrees/metrics/z0_6m_128_metrics.json
```

Notes:
- If you enable `--viz True` the script will open Open3D visualizers and show images for each step of the registration
- Metrics are written into a JSON file next to your source file under a `metrics/` directory. The JSON includes the estimated transformation, fitness, inlier RMSE and other diagnostics.

#### Initial State
<img src="/docs/gifs/initial_state.gif" width="750" height="750"/>

#### TEASER++ + ICP Result
<img src="/docs/gifs/registration_result.gif" width="750" height="750"/>

Troubleshooting & tips
----------------------
- Check Source and Target references: The script loads a JSON with the key `H` used as a homogeneous 4×4 matrix. Make sure the Source point cloud is aligned with the same axis reference than the Target;
- Using the ROS simulation to produce more datasets: If you decide to build your own dataset using the ROS Simulation [1] so you'll have use `flip_and_scale_pc_folder.py` script to align the axis and scale the point clouds before using them in the teaser script. 
- If you want to use another source point cloud and check if the calculated metrics are correct, just make sure to correct the axis reference t match the target reference
- TEASER++ bindings: The TEASER++ Python bindings may require compilation — if `import teaserpp_python` fails, follow TEASER++ build instructions for your platform.
