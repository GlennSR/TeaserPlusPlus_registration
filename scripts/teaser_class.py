# A classe vai receber inicialmente como parametros do constructor:
# - target_filepath: str -> caminho do point cloud alvo
# - voxel_size: float -> tamanho do voxel para downsampling, DEFAULT=30
# - max_iter_icp: int -> número máximo de iterações do ICP, DEFAULT=2000
# - noise_std: float -> desvio padrão do ruído a ser adicionado, DEFAULT=0.0
# - verbose: str -> nível de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL), DEFAULT='INFO'
# - viz: bool -> se deve visualizar os resultados com open3d, DEFAULT=False
# 
# A classe terá os seguintes métodos:
# - load_sources(self, source_path: str) -> List[o3d.geometry.PointCloud]
#     -> carrega uma pasta de arquivos point cloud .ply e seus .json respectivos com a matriz homogenea H de transformação inicial
# 
# - load_source(self, source_filepath: str) -> o3d.geometry.PointCloud
#     -> carrega um único arquivo point cloud .ply
#
# - register_one(self) -> np.ndarray(np.ndarray(4,4)))
#     -> chama teaserpp_registration() com o source unico e retorna a matriz de transformação 4x4 calculada
#
# - register_all(self) -> dict(np.ndarray(np.ndarray(4,4))))
#    -> chama register_one() para cada source carregado e retorna um dicionário com os resultados
# 
# - calculate_metrics(self, metric_filepath: str) -> None
#    -> Recebe o caminho de uma pasta ou arquivo .json com as métricas de registro groud truth
#    -> calcula as métricas de registro comparando a transformação calculada com a ground truth do arquivo .json

from teaserpp_fpfh import teaserpp_registration
from registration.utils.transforms import transformation_error
from registration.utils.metrics import compute_rmse_transformations
import open3d as o3d
import numpy as np
import logging
import os
import json

logger = logging.getLogger(__name__)

class TeaserPP:
    """
    TEASER++ REGISTRATION CLASS
    This class launches Teaser++ registration + an ICP refinnement to align a source to a target point cloud, both as .ply files.
    It also has an option to output the registration metrics, such as fitness, inlier RMSE and others, calculated with Open3D and TEASER++ internal metrics,
    to a .json file and a function to calculate the rotation and translation error of the registration result compared to a ground truth transformation, 
    passed as a .json file by the user.
    
        The class has the following methods:
        - load_source_files(self, source_path: str) -> None
            -> Loads multiple point clouds from a directory, looking for .ply files and their respective .json files with the initial transformation guess.

        - load_source_file(self, source_filepath: str, initial_transform_file: str = None) -> None
            -> Loads a single point cloud from a file, and its respective initial transformation guess from a .json file. If no initial transformation file is provided, it uses the identity matrix as the initial guess.

        - load_source(self, source: o3d.geometry.PointCloud, initial_transform: np.ndarray) -> None
            -> Loads a single point cloud and its initial transformation guess.

        - clear_sources(self) -> None
            -> Clears all loaded source point clouds and their transformations.

        - get_source_pcds(self) -> list
            -> Returns the list of loaded source point clouds.
        
        - get_initial_guess_transforms(self) -> list
            -> Returns the list of initial transformation guesses for the source point clouds.
        
        - get_target_pcd(self) -> o3d.geometry.PointCloud
            -> Returns the target point cloud.

        - get_transformations(self) -> dict
            -> Returns the dictionary of estimated transformations for each source point cloud.

        - register_one(self, index: int = 0) -> tuple
            -> Registers the source point cloud at the given index to the target point cloud.

        - register_all(self) -> dict[str, tuple]
            -> Registers all loaded source point clouds to the target point cloud and returns a dictionary with the results.
        
        - calculate_errors(self, source_gt_path: str, source_index: int) -> None
            -> Calculates the rotation and translation error of the registration result compared to a ground truth transformation.

    """
    def __init__(self, target_filepath: str, voxel_size: float = 30.0, max_iter_icp: int = 2000, verbose: str = 'INFO', viz: bool = False, calculate_metrics: bool = False, metric_savepath: str = None):
        '''
        Class constructor
        
        :param target_filepath: The path to the target point cloud file (.ply)
        :param voxel_size: The voxel size to be used for downsampling the point clouds before registration (in mm)
        :param max_iter_icp: The maximum number of iterations for the ICP refinement
        :param verbose: The verbosity level for logging
        :param viz: Enable visualization of the registration steps using open3D
        :param calculate_metrics: Enable calculation of the registration metrics and save them to a .json file
        :param metric_savepath: The path to save the registration metrics .json files (if calculate_metrics is True)
        
        :param _target_pcd: The target point cloud loaded from the target_filepath
        :param _source_pcds: A list to store the loaded source point clouds
        :param _initial_guess_transforms: A list to store the initial transformation guesses for each source point cloud
        :param _transformations: A dictionary to store the resulting transformations for each source point cloud
        '''
        self._target_filepath = target_filepath
        self._voxel_size = voxel_size
        self._max_iter_icp = max_iter_icp
        self._verbose = verbose
        self._viz = viz
        self._calculate_metrics = calculate_metrics
        self._metric_savepath = metric_savepath

        self._target_pcd = o3d.io.read_point_cloud(target_filepath)
        self._source_pcds = []
        self._initial_guess_transforms = []
        self._transformations = {}

    def load_source_files(self, source_path: str) -> None:
        '''
        Loads multiple point clouds from a directory and appends them to the class lists.
        
        :param source_path: The path to the directory containing the source point cloud files (.ply) and their respective initial transformation files (.json)
        '''
        # Create a list with only the supported point cloud files for registration
        source_files = [f for f in os.listdir(source_path) if f.endswith('.ply') or f.endswith('.pcd')] 
        number_of_files = len(source_files)
        logger.info(f"Loading {number_of_files} files.")
        # Create a list of the corresponding initial transformation files 
        initial_transform_files = [f.replace('.ply', '.json').replace('.pcd', '.json') for f in source_files] 
        logger.info(f"Looking for initial transformation files: {initial_transform_files}") # Test
        # For each source file, loads the point cloud and its respective initial guess
        for filename in source_files:
            file_path = os.path.join(source_path, filename)
            initial_transform_file = os.path.join(source_path, initial_transform_files[source_files.index(filename)])
            self.load_source_file(file_path, initial_transform_file)


    def load_source_file(self, source_filepath: str, initial_transform_file: str = None) -> None:
        '''
        Loads a single point cloud from a file, and its respective initial transformation guess from a .json file and appends them to the class lists. 
        If no initial transformation file is provided, it uses the identity matrix as the initial guess.
        
        :param source_filepath: The path to the source point cloud file (.ply)
        :param initial_transform_file: The path to the initial transformation file (.json)
        ''' 
        logger.info(f"Loading source file: {source_filepath} with initial transform file: {initial_transform_file}")
        pcd = o3d.io.read_point_cloud(source_filepath)
        # Stores the point cloud to the class source list
        self._source_pcds.append(pcd)

        # load initial transformation guess
        if initial_transform_file is None:
            logger.warning("No initial transformation file provided, using identity matrix as initial guess.")
            initial_guess_transform = np.eye(4)
        else:
            try:
                with open(initial_transform_file, 'r') as file:
                    initial_guess_transform = np.array(json.load(file)["H"])
                    logger.info(f"Initial transformation guess: \n{initial_guess_transform}")
            except FileNotFoundError:
                logger.error(f"The file '{initial_transform_file}' was not found.")
            # Stores the initial guess to the class initial guesses list
            self._initial_guess_transforms.append(initial_guess_transform)

    def load_source(self, source: o3d.geometry.PointCloud, initial_transform: np.ndarray) -> None:
        '''
        Loads a single point cloud directly in open3D Point Cloud format
        and its initial transformation guess as a 4x4 numpy array and appends them to the class lists.
        
        :param source: The source point cloud in open3D Point Cloud format
        :param initial_transform: The initial transformation guess as a 4x4 numpy array
        '''

        self._source_pcds.append(source)
        self._initial_guess_transforms.append(initial_transform)

    
    def clear_sources(self) -> None:
        '''
        Clear the class stored point clouds and transformations lists.
        '''
        self._source_pcds = []
        self._initial_guess_transforms = []
        self._transformations = {}
        logger.info("Cleared all source point clouds and transformations.")

    
    def get_source_pcds(self) -> list:
        return self._source_pcds


    def get_initial_guess_transforms(self) -> list:
        return self._initial_guess_transforms
    

    def get_target_pcd(self) -> o3d.geometry.PointCloud:
        return self._target_pcd
    

    def get_transformations(self) -> dict:
        return self._transformations


    def register_one(self, index: int = 0) -> tuple:
        '''
        Launchs the registration to a single source point cloud specified by its index in the loaded sources of the class.
        Calls teaserpp_registration() method from the teaserpp_fpfh module.
        
        :param index: Index of the source point cloud to be registered, default is 0 (the first loaded source)
        
        :return: A tuple containing the estimated transformation as a 4x4 numpy array, the ICP registration solver, 
        and the calculated metrics as a dictionary (if calculate_metrics is True) or None (if calculate_metrics is False or if no metric save path is provided).

        '''
        logger.info(f"Registering source point cloud with {len(self._source_pcds[index].points)} points.")
        estimated_transform, icp_sol, teaser_solver, nb_points_target_down, nb_points_source_down, num_corrs, total_time, trans_init = teaserpp_registration(self._source_pcds[index], 
                                                self._initial_guess_transforms[index],
                                                self._target_pcd, 
                                                self._voxel_size, 
                                                self._max_iter_icp,  
                                                self._verbose, 
                                                self._viz)
        # Saving the resulting transformation to the class transformations dictionary
        self._transformations[index] = estimated_transform

        # If enabled, calculate the registration metrics and save them to a .json file
        if self._calculate_metrics and self._metric_savepath is not None:
            output_metrics = calculate_metrics(self._target_pcd, self._source_pcds[index],
                            nb_points_target_down, nb_points_source_down,
                            teaser_solver, icp_sol, trans_init, 
                            num_corrs, self._voxel_size * 2, total_time, self._metric_savepath, index)
            return estimated_transform, icp_sol, output_metrics
        if self._calculate_metrics and self._metric_savepath is None:
            logger.warning("Error! No save path provided for metrics, skipping metrics calculation.")
        return estimated_transform, icp_sol, None

    def register_all(self) -> dict[str, tuple]:
        '''
        Launches the registration for all loaded source point clouds.

        :return: A dictionary containing a tuple for each source point cloud with the estimated transformation, ICP registration solver, and calculated metrics.
        '''
        results = {}
        for index in range(len(self._source_pcds)):
            estimated_transform, icp_sol, metrics = self.register_one(index)
            self._transformations[index] = estimated_transform
            results[f"source_{index}"] = (estimated_transform, icp_sol, metrics)
        return results
    

    def calculate_errors(self, source_gt_path: str, source_index: int) -> None:
        ''' 
        Calculates the rotation and translation error of the registration result compared to a ground truth transformation, passed as a .json file by the user.
        
        :param source_gt_path: The path to the .json file containing the ground truth transformation matrix under the key "H"
        :param source_index: The index of the source point cloud for which to calculate the errors
        
        :return: A tuple containing the rotation error in radians and degrees, and the translation error
        '''
        # Load Ground Thruth transformation from .json file
        estimated_Transform = self._transformations[source_index]
        try:
            with open(source_gt_path, 'r') as file:
                source_gt_transform = np.array(json.load(file)["H"])
                logger.info(f"Source Ground Truth transform: \n{source_gt_transform}")
        except FileNotFoundError:
            logger.error(f"The file '{source_gt_path}' was not found.")
        # NB this only make sense if you are aligning the same model
        # difference between initial and final transformation
        rot_err, trans_err = transformation_error(
            estimated_Transform, source_gt_transform
        )
        matrix = estimated_Transform @ source_gt_transform
        logger.debug(f"Product of the transformations:\n{matrix}")
        logger.info(
            f"Rotation error (radians): {rot_err:.4f} (degrees: {np.degrees(rot_err):.4f}), Translation error: {trans_err:.4f}"
        )

        # compute the rms error between initial and final translation (assuming that the points are corresponding)
        registration_rmse = compute_rmse_transformations(
            estimated_Transform, source_gt_transform, self.source_pcds[source_index]
        )
        logger.info(f"Registration RMSE: {registration_rmse}")
        return rot_err, trans_err


import teaserpp_python
import copy

def calculate_metrics(target_raw: o3d.geometry.PointCloud,
                         source_raw: o3d.geometry.PointCloud,
                         target_down_nb_points: int,
                         source_down_nb_points: int,
                         teaser_solver: teaserpp_python.teaserpp_python.RobustRegistrationSolver,
                         icp_sol: o3d.pipelines.registration.RegistrationResult,
                         trans_init: np.ndarray,
                         num_corrs: int,
                         NOISE_BOUND: float,
                         registration_total_time: float,
                         output_path: str = None,
                         index: int = 0,
                         ):
    """
    Calculate and log various metrics to evaluate the registration result.
    
    Args:
        :param target_raw: Target point cloud
        :param source_raw: Source point cloud
        :param teaser_solver: The TEASER++ solver calculated
        :param icp_sol: The ICP refinement result (Final transformation)
        :param num_corrs: Number of correspondences calculated with FPFH
        :param NOISE_BOUND: Noise bound used in TEASER++
        :param registration_total_time: Total time taken for registration
        :param args: Command line arguments
    """

    # Calculate the metric of the result transformation using Open3D compute_point_cloud_distance() method
    # Full-cloud distances
    source_raw_T_icp = copy.deepcopy(source_raw).transform(icp_sol.transformation)
    distances_o3d = target_raw.compute_point_cloud_distance(source_raw_T_icp)
    logger.info(f"Mean Open3D distance for the registration result (full cloud): {np.mean(distances_o3d):.6f}")

    # Calculate the standard deviation of the full-cloud distances
    std_distance = np.std(distances_o3d)
    logger.info(f"Standard deviation of distances after registration (full cloud): {std_distance:.6f}")

    # TEASER++ internal quality metrics
    teaser_solution = teaser_solver.getSolution()
    translation_inliers = teaser_solver.getTranslationInliers()
    rotation_inliers = teaser_solver.getRotationInliers()
    logger.info(f"TEASER++ Internal Metrics:")
    logger.info(f"  Translation inliers: {len(translation_inliers)} / {num_corrs} ({len(translation_inliers)/num_corrs*100:.1f}%)")
    logger.info(f"  Rotation inliers (max clique): {len(rotation_inliers)} / {num_corrs} ({len(rotation_inliers)/num_corrs*100:.1f}%)")
    logger.info(f"  Solution valid: {teaser_solution.valid if hasattr(teaser_solution, 'valid') else 'N/A'}")

    # Evaluate the solution using Open3D
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source_raw, target_raw, NOISE_BOUND, icp_sol.transformation
    )
    logger.info(f"Open3D Evaluation Metrics:")
    logger.info(f"  Fitness: {evaluation.fitness:.4f} (fraction of source inlier points)")
    logger.info(f"  Inlier RMSE: {evaluation.inlier_rmse:.4f} mm (lower is better)")
    logger.info(f"  ICP correspondence set size: {len(evaluation.correspondence_set)} / {len(source_raw.points)} source points ({len(evaluation.correspondence_set)/len(source_raw.points)*100:.1f}%)")
    logger.info(f"  FPFH correspondences: {num_corrs}")


    ## Calculate inliers mean error (distances) between the correspondent points

    # Build point clouds of the correspondent inlier points
    corr = np.asarray(evaluation.correspondence_set)
    src_corr_pts = np.asarray(source_raw.points)[corr[:,0]]
    tgt_corr_pts = np.asarray(target_raw.points)[corr[:,1]]

    percentage_inliers_to_target = len(src_corr_pts) / len(target_raw.points)
    logger.info(f"Percentage of inliers with respect to target point cloud: {percentage_inliers_to_target*100:.2f} %")
    percentage_inliers_to_source = len(src_corr_pts) / len(source_raw.points)
    logger.info(f"Percentage of inliers with respect to source point cloud: {percentage_inliers_to_source*100:.2f} %")

    src_corr_pcd = o3d.geometry.PointCloud()
    tgt_corr_pcd = o3d.geometry.PointCloud()
    src_corr_pcd.points = o3d.utility.Vector3dVector(src_corr_pts)
    tgt_corr_pcd.points = o3d.utility.Vector3dVector(tgt_corr_pts)

    # Compute inliers distances after registration
    src_corr_pcd_T = copy.deepcopy(src_corr_pcd)
    src_corr_pcd_T.transform(icp_sol.transformation)
    distances_inliers = tgt_corr_pcd.compute_point_cloud_distance(src_corr_pcd_T)

    logger.info(f"Mean distance for the registration inliers (only inliers): {np.mean(distances_inliers):.6f}")
    
    # Correct icp transformation by the initial transformation used for gravity alignment
    T_icp_corrected = icp_sol.transformation @ trans_init

    logger.info(f"ICP refinement result: {icp_sol}")
    logger.info(f"Estimated matrix:\n{T_icp_corrected}")
    logger.info(f"Estimated non corrected matrix:\n{icp_sol.transformation}")
    logger.info(
        f"Result fitness: {icp_sol.fitness}, inlier RMSE: {icp_sol.inlier_rmse} mm"
    )

    # Calculate the diagonal length of the target point cloud bounding box and the RMSE as percentage of it
    max_point = np.max(np.asarray(target_raw.points), axis=0)
    min_point = np.min(np.asarray(target_raw.points), axis=0)
    target_diagonal_length = np.linalg.norm(max_point - min_point)
    logger.info(f"Target point cloud diagonal length: {target_diagonal_length:.3f} mm")

    rmse_percentage = icp_sol.inlier_rmse / target_diagonal_length * 100
    logger.info(f"ICP inlier RMSE as percentage of target diagonal length: {rmse_percentage:.4f} %")

    # Save the calculated metrics to a .json file
    output_metrics = {
        "fitness": icp_sol.fitness,
        "inlier_rmse": icp_sol.inlier_rmse,
        "rmse_percentage_to_target_diagonal": rmse_percentage,
        "mean_distance_points_full_cloud": float(np.mean(distances_o3d)),
        "max_distance_points_full_cloud": float(np.max(distances_o3d)),
        "standard_deviation_distance_full_cloud": float(std_distance),
        "inlier_mean_distance": np.mean(distances_inliers),
        "registration_total_time_sec": registration_total_time,
        "nb_of_points_target_down": target_down_nb_points,
        "nb_of_points_source_down": source_down_nb_points,
        "percentage_inliers_to_target": percentage_inliers_to_target,
        "percentage_inliers_to_source": percentage_inliers_to_source,
        "estimated_transformation": T_icp_corrected.tolist(),
    }
    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        file_name = str(index) + '.json'
        metrics_file = os.path.join(output_path, file_name)
        with open(metrics_file, 'w') as file:
            json.dump(output_metrics, file, indent=4)
            logger.info(f"Saved metrics to {metrics_file}")
    except FileNotFoundError:
        logger.error(f"The file 'metrics.json' was not found.")
    
    return output_metrics


if __name__ == "__main__":
    # Example usage
    # target_filepath = "../test_data/Real_Lidar/GT_dataset/Target/map_64k.ply"
    # source_path = "../test_data/Real_Lidar/GT_dataset/Source/Combined_sources_step20/0.ply"
    # initial_transform_file = "../test_data/Real_Lidar/GT_dataset/Source/Combined_sources_step20/0.json"
    # teaser = TeaserPP(target_filepath, voxel_size=150, max_iter_icp=1000, viz=False, calculate_metrics=True, metric_savepath="./metrics_output")
    # teaser.load_source_file(source_path, initial_transform_file)
    # estimated_position, _, _ = teaser.register_one()
    # logger.info(f"Estimated position: {estimated_position}")
    # logger.info(f"Error: {teaser.calculate_errors(source_gt_path=initial_transform_file, source_index=0)}")

    target_filepath = "../test_data/Real_Lidar/GT_dataset/Target/map_64k.ply"
    source_path = "../test_data/Real_Lidar/GT_dataset/Source/Combined_sources_step20/"
    teaser = TeaserPP(target_filepath, voxel_size=150, max_iter_icp=1000, viz=False, calculate_metrics=True, metric_savepath="./metrics_output")
    teaser.load_source_files(source_path)
    logger.info(f"Loaded {len(teaser.get_source_pcds())} source point clouds.")
    logger.info(f"Initial guess transforms: {teaser.get_initial_guess_transforms()}")
    results = teaser.register_all()
    for source_name, (estimated_transform, icp_sol, metrics) in results.items():
        logger.info(f"Estimated position for {source_name}: {estimated_transform}")
    # logger.info(f"Error: {teaser.calculate_errors(source_gt_path=initial_transform_file, source_index=0)}")