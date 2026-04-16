import matplotlib.pyplot as plt
import numpy as np
import json
import os
import argparse
import logging
from registration.utils.logging import setup_logging

logger = logging.getLogger(__name__)

def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            json_data = json.load(file)
            metric_values["rotation_error_deg"].append(json_data["rotation_error_deg"])
            metric_values["translation_error"].append(json_data["translation_error"])
            metric_values["fitness"].append(json_data["fitness"])
            metric_values["nb_of_points_target_down"].append(json_data["nb_of_points_target_down"])
            metric_values["nb_of_points_source_down"].append(json_data["nb_of_points_source_down"])
            if "nb_of_fpfh_correspondences" in json_data:
                metric_values["nb_of_fpfh_correspondences"].append(json_data["nb_of_fpfh_correspondences"])
            else:
                metric_values["nb_of_fpfh_correspondences"].append(0)
            metric_values["difference_source_to_target_downsampled_points"].append(np.abs(json_data["nb_of_points_target_down"] - json_data["nb_of_points_source_down"]))
            metric_values["percentage_source_inliers_to_target"].append(json_data["percentage_inliers_to_target"])
            metric_values["rmse_percentage_to_target_diagonal"].append(100*json_data["rmse_percentage_to_target_diagonal"])
            metric_values["mean_distance_points_full_cloud"].append(json_data["mean_distance_points_full_cloud"])
            metric_values["max_distance_points_full_cloud"].append(json_data["max_distance_points_full_cloud"])
            metric_values["std_deviation_distance_full_cloud"].append(json_data["standard_deviation_distance_full_cloud"])
            metric_values["inlier_mean_distance"].append(json_data["inlier_mean_distance"])
            metric_values["registration_total_time_sec"].append(json_data["registration_total_time_sec"])

    except FileNotFoundError:
        logger.error(f"The file '{file_path}' was not found.")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training metrics from a JSON file or folder.")
    parser.add_argument("--input", type=str, help="Path to the JSON file containing training metrics.", required=True)
    parser.add_argument("--output_path", type=str, help="Path to save the graphics.", required=True)
    parser.add_argument("--long_data", type=bool, default=False, help="Indicates if the dataset is long and requires a different visualization approach.")
    parser.add_argument(
        "-v",
        "--verbose",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )

    input_args = parser.parse_args()
    # Set logging level based on user selection
    setup_logging(getattr(logging, input_args.verbose))

    metric_values = {
                "metric_label": [],
                "rotation_error_deg": [],
                "translation_error": [],
                "fitness": [],
                "nb_of_points_target_down": [],
                "nb_of_points_source_down": [],
                "difference_source_to_target_downsampled_points": [],
                "nb_of_fpfh_correspondences": [],
                "percentage_source_inliers_to_target": [],
                "rmse_percentage_to_target_diagonal": [],
                "mean_distance_points_full_cloud": [],
                "max_distance_points_full_cloud": [],
                "std_deviation_distance_full_cloud": [],
                "inlier_mean_distance": [],
                "registration_total_time_sec": []
            }
    
    # Thresholds for the registration to be considered successful
    thresholds = {
                "rotation_error_deg": 5.0,
                "translation_error": 200.0,
    }
    
    is_data_long = input_args.long_data
    if is_data_long:
        metric_files = []
        for i in range(0,380):
            if os.path.exists(os.path.join(input_args.input, f"{i}_metrics.json")):
                metric_files.append(f"{i}_metrics.json")
        number_of_files = len(metric_files)
        logger.info(f"Source is a directory, creating graphics to all its {number_of_files} files.")
        
    elif is_data_long == False:
        # Create a list with only the supported point cloud files for registration
        metric_files = [f for f in os.listdir(input_args.input) if f.endswith('.json')] 
        metric_files = sorted(metric_files)  # Sort files based on the number in their name
        number_of_files = len(metric_files)
        logger.info(f"Source is a directory, creating graphics to all its {number_of_files} files.")

    count = 0
    for filename in metric_files:
        metric_label = filename.replace('_metrics.json', '')
        logger.info(f"Loading metric: {metric_label}")
        file_path = os.path.join(input_args.input, filename)
        metric_values["metric_label"].append(metric_label)
        load_json(file_path)
        count += 1
    logger.info(f"Loaded {count} metric files")

    mean_rotation_error = np.mean(metric_values["rotation_error_deg"])
    inlier_values = [value for value in metric_values["translation_error"] if value < 99999]
    mean_translation_error = np.mean(inlier_values)
    logger.info(f"Mean rotation error (degrees): {mean_rotation_error:.4f}")
    logger.info(f"Mean translation error (mm): {mean_translation_error:.4f}")

    # successful_registrations = [1 if (metric_values["rotation_error_deg"][index] < thresholds["rotation_error_deg"] 
    #                                 and metric_values["translation_error"][index] < thresholds["translation_error"]) else 0 
    #                                 for index in range(len(metric_values["rotation_error_deg"]))]
    mean_fitness_successful_registrations = 100 * np.mean([metric_values["fitness"][index] for index in range(len(metric_values["fitness"])) if (metric_values["rotation_error_deg"][index] < thresholds["rotation_error_deg"] 
                                    and metric_values["translation_error"][index] < thresholds["translation_error"])])
    logger.info(f"Mean fitness of successful registrations: {mean_fitness_successful_registrations:.4f}")

    mean_percentage_source_inliers_to_target_successful_registrations = 100 * np.mean([metric_values["percentage_source_inliers_to_target"][index] for index in range(len(metric_values["percentage_source_inliers_to_target"])) if (metric_values["rotation_error_deg"][index] < thresholds["rotation_error_deg"] 
                                    and metric_values["translation_error"][index] < thresholds["translation_error"])])
    logger.info(f"Mean percentage of inliers to target of successful registrations: {mean_percentage_source_inliers_to_target_successful_registrations:.4f}%")

    mean_fitness = 100 * np.mean(metric_values["fitness"])
    logger.info(f"Mean fitness of all registrations: {mean_fitness:.4f}")

    mean_percentage_source_inliers_to_target = 100 * np.mean(metric_values["percentage_source_inliers_to_target"])
    logger.info(f"Mean percentage of inliers to target of all registrations: {mean_percentage_source_inliers_to_target:.4f}%")

    is_deleted = False
    # Visualization
    if is_data_long:
        for metric, values in metric_values.items():
            if metric == "metric_label":
                continue

            # Prepare x positions and labels
            x_positions = list(range(len(values)))

            # Dynamic figure width: scale with number of samples (min 12, ~0.3 inch per sample)
            fig_width = max(12, len(values) * 0.3)
            fig, ax = plt.subplots(figsize=(fig_width, 8))

            # If the registration worked (rotation_error_deg < 5.0 or translation_error < 100) then change the color to green.
            colors = ['green' if (metric_values["rotation_error_deg"][value_index] < thresholds["rotation_error_deg"] 
                                    and metric_values["translation_error"][value_index] < thresholds["translation_error"]) else 'red' 
                                    for value_index in range(len(values))]

            ax.scatter(x_positions, values, c=colors, marker='s', s=20)
            ax.set_title('Plot of ' + metric, fontsize=14)
            ax.set_xlabel('Sample Index', fontsize=12)
            ax.set_ylabel(f"{metric}", fontsize=12)
            if metric == "rotation_error_deg":
                rotation_registrations = [1 if (metric_values[metric][value_index] < thresholds["rotation_error_deg"]) else 0 
                                    for value_index in range(len(values))]
                # Set y-axis ticks in steps of 10 for easier reading
                y_min = min(values)
                y_max = max(values)
                ax.set_yticks(np.arange(int(y_min // 10) * 10, y_max + 10, 10))
            # Show metric labels on x-axis (rotated vertically for readability)
            try:
                ax.set_xticks(x_positions)
                ax.set_xticklabels(metric_values['metric_label'], rotation=90, ha='center', fontsize=10)
            except Exception:
                # If labels don't match length or cause an error, skip labeling
                pass
            ax.grid(True)
            fig.tight_layout()
            # plt.show()
            if not os.path.exists(os.path.join(input_args.input, input_args.output_path)):
                os.makedirs(os.path.join(input_args.input, input_args.output_path))
            elif is_deleted == False:
                for filename in os.listdir(os.path.join(input_args.input, input_args.output_path)):
                    file_path = os.path.join(os.path.join(input_args.input, input_args.output_path), filename)
                    # Check if it is a file (not a subdirectory)
                    if os.path.isfile(file_path):
                        os.remove(file_path)  # Remove the file
                is_deleted = True
            
            logger.info(f"Created output directory at {os.path.join(input_args.input, input_args.output_path)}")
            fig.savefig(os.path.join(input_args.input, input_args.output_path, f"{metric}.png"), dpi=150, bbox_inches='tight')
            plt.close()
    elif is_data_long == False:
        for metric, values in metric_values.items():
            if metric == "metric_label":
                continue

            # Prepare x positions and labels
            x_positions = list(range(len(values)))

            # If the registration worked (rotation_error_deg < 5.0 or translation_error < 100) then change the color to green.
            colors = ['green' if (metric_values["rotation_error_deg"][value_index] < thresholds["rotation_error_deg"] 
                                    and metric_values["translation_error"][value_index] < thresholds["translation_error"]) else 'red' 
                                    for value_index in range(len(values))]

            plt.scatter(x_positions, values, c=colors, marker='s')
            plt.title('Plot of ' + metric)
            plt.xlabel('Sample Index')
            plt.ylabel(f"{metric}")
            if metric == "rotation_error_deg":
                rotation_registrations = [1 if (metric_values[metric][value_index] < thresholds["rotation_error_deg"]) else 0 
                                    for value_index in range(len(values))]
                # Set y-axis ticks in steps of 10 for easier reading
                y_min = min(values)
                y_max = max(values)
                plt.yticks(np.arange(int(y_min // 10) * 10, y_max + 10, 10))
            # Show metric labels on x-axis (rotated for readability)
            try:
                plt.xticks(x_positions, metric_values['metric_label'], rotation=45, ha='right')
            except Exception:
                # If labels don't match length or cause an error, skip labeling
                pass
            plt.grid(True)
            plt.tight_layout()
            # plt.show()
            if not os.path.exists(os.path.join(input_args.input, input_args.output_path)):
                os.makedirs(os.path.join(input_args.input, input_args.output_path))
            elif is_deleted == False:
                for filename in os.listdir(os.path.join(input_args.input, input_args.output_path)):
                    file_path = os.path.join(os.path.join(input_args.input, input_args.output_path), filename)
                    # Check if it is a file (not a subdirectory)
                    if os.path.isfile(file_path):
                        os.remove(file_path)  # Remove the file
                is_deleted = True
            
            logger.info(f"Created output directory at {os.path.join(input_args.input, input_args.output_path)}")
            plt.savefig(os.path.join(input_args.input, input_args.output_path, f"{metric}.png"))
            plt.close()
            
    
    logger.info(f"Saved plot for {metric} as {metric}.png")

    # Save the percentage of successful registrations based on Rotatio & Translation errors or on the Rotation error only.
    successful_registrations = colors.count('green')/len(colors) * 100
    logger.info(f"Number of successful registrations: {successful_registrations:.2f}%")
    successful_rotation_registrations = sum(rotation_registrations)/len(rotation_registrations) * 100
    logger.info(f"Number of successful rotation registrations: {successful_rotation_registrations:.2f}%")

    # voxel_size = int(input_args.input.split('/')[-1].replace('Voxel', ''))
    voxel_size = 0

    success_rate = {
        "path": input_args.input,
        "voxel_size": voxel_size,
        "total_registrations": len(colors),
        "rotation_error_threshold_deg": thresholds["rotation_error_deg"],
        "translation_error_threshold_mm": thresholds["translation_error"],
        "successful_registrations_percentage": successful_registrations,
        "successful_rotation_registrations_percentage": successful_rotation_registrations,
        "mean_rotation_error_deg": mean_rotation_error,
        "mean_translation_error_mm": mean_translation_error,
        "mean_nb_of_points_target_downsampled": np.mean(metric_values["nb_of_points_target_down"]),
        "mean_nb_of_points_source_downsampled": np.mean(metric_values["nb_of_points_source_down"]),
        "mean_difference_source_to_target_points_downsampled": np.mean(metric_values["difference_source_to_target_downsampled_points"]),
        "mean_nb_of_fpfh_correspondences": np.mean(metric_values["nb_of_fpfh_correspondences"]) ,
        "mean_fitness": mean_fitness,
        "mean_percentage_source_inliers_to_target": mean_percentage_source_inliers_to_target,
        "mean_fitness_successful_registrations": mean_fitness_successful_registrations,
        "mean_percentage_source_inliers_to_target_successful_registrations": mean_percentage_source_inliers_to_target_successful_registrations
    }

    # voxel_size = input_args.input.split('/')[-1]
    # number_voxel_size = voxel_size
    # logger.info(f"Voxel size: {voxel_size}")
    with open(os.path.join(input_args.input, input_args.output_path, f"success_rate.json"), 'w') as file:
            json.dump(success_rate, file, indent=4)
            logger.info(f"Saved metrics to {os.path.join(input_args.input, input_args.output_path, f'success_rate_{voxel_size}.json')}")
