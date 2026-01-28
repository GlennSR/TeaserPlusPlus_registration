import matplotlib.pyplot as plt
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
            metric_values["rmse_percentage_to_target_diagonal"].append(json_data["rmse_percentage_to_target_diagonal"])
            metric_values["mean_distance"].append(json_data["mean_distance"])
            metric_values["max_distance"].append(json_data["max_distance"])
            metric_values["std_deviation_distance"].append(json_data["standard_deviation_distance"])
            metric_values["inlier_mean_distance"].append(json_data["inlier_mean_distance"])
            metric_values["registration_total_time_sec"].append(json_data["registration_total_time_sec"])

    except FileNotFoundError:
        logger.error(f"The file '{file_path}' was not found.")
        return None

# def visualize_metrics(input_path):


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training metrics from a JSON file or folder.")
    parser.add_argument("--input", type=str, help="Path to the JSON file containing training metrics.", required=True)
    parser.add_argument("--output_path", type=str, help="Path to save the graphics.", required=True)
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
                "rotation_error_deg": [],
                "translation_error": [],
                "fitness": [],
                "rmse_percentage_to_target_diagonal": [],
                "mean_distance": [],
                "max_distance": [],
                "std_deviation_distance": [],
                "inlier_mean_distance": [],
                "registration_total_time_sec": []
            }
    
    
    # Create a list with only the supported point cloud files for registration
    metric_files = [f for f in os.listdir(input_args.input) if f.endswith('.json')] 
    number_of_files = len(metric_files)
    logger.info(f"Source is a directory, applying TEASER++ registration to all its {number_of_files} files.")
    
    count = 0
    for filename in metric_files:
        file_path = os.path.join(input_args.input, filename)
        load_json(file_path)
        count += 1

    logger.info(f"Loaded {count} metric files")

    # Visualization
    for metric, values in metric_values.items():
        plt.hist(values, bins=10, color='blue', rwidth=1.0)
        plt.title('Histogram of ' + metric)
        plt.xlabel(f"{metric}")
        plt.ylabel('Frequency')
        plt.grid(True)
        # plt.show()
        if not os.path.exists(os.path.join(input_args.input, input_args.output_path)):
            os.makedirs(os.path.join(input_args.input, input_args.output_path))
        
        logger.info(f"Created output directory at {os.path.join(input_args.input, input_args.output_path)}")
        plt.savefig(os.path.join(input_args.input, input_args.output_path, f"{metric}_hist.png"))
        plt.close()
        logger.info(f"Saved plot for {metric} as {metric}_hist.png")

