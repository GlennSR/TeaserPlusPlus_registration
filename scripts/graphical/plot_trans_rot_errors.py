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
    except FileNotFoundError:
        logger.error(f"The file '{file_path}' was not found.")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize rotation and translation errors from JSON metric files.")
    parser.add_argument("--input", type=str, help="Path to the directory containing the JSON metric files.", required=True)
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
    setup_logging(getattr(logging, input_args.verbose))

    metric_values = {
        "metric_label": [],
        "rotation_error_deg": [],
        "translation_error": [],
    }

    # Thresholds for the registration to be considered successful
    thresholds = {
        "rotation_error_deg": 2.0,
        "translation_error": 200.0,
    }

    is_data_long = input_args.long_data
    if is_data_long:
        metric_files = []
        for i in range(0, 380):
            if os.path.exists(os.path.join(input_args.input, f"{i}_metrics.json")):
                metric_files.append(f"{i}_metrics.json")
        number_of_files = len(metric_files)
        logger.info(f"Source is a directory, creating graphics to all its {number_of_files} files.")
    else:
        metric_files = [f for f in os.listdir(input_args.input) if f.endswith('.json')]
        metric_files = sorted(metric_files)
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

    # Create output directory
    output_dir = os.path.join(input_args.input, input_args.output_path)
    is_deleted = False
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        is_deleted = True

    # Plot each metric
    for metric, values in metric_values.items():
        if metric == "metric_label":
            continue

        x_positions = list(range(len(values)))

        colors = ['green' if (metric_values["rotation_error_deg"][i] < thresholds["rotation_error_deg"]
                              and metric_values["translation_error"][i] < thresholds["translation_error"]) else 'red'
                  for i in range(len(values))]

        if is_data_long:
            fig_width = max(12, len(values) * 0.3)
            fig, ax = plt.subplots(figsize=(fig_width, 8))
            ax.scatter(x_positions, values, c=colors, marker='s', s=20)
            ax.set_title('Plot of ' + metric, fontsize=14)
            ax.set_xlabel('Sample Index', fontsize=12)
            ax.set_ylabel(f"{metric}", fontsize=12)
            if metric == "rotation_error_deg":
                y_min = min(values)
                y_max = max(values)
                ax.set_yticks(np.arange(int(y_min // 10) * 10, y_max + 10, 10))
            try:
                ax.set_xticks(x_positions)
                ax.set_xticklabels(metric_values['metric_label'], rotation=90, ha='center', fontsize=10)
            except Exception:
                pass
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"{metric}.png"), dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.scatter(x_positions, values, c=colors, marker='s')
            plt.title('Plot of ' + metric)
            plt.xlabel('Sample Index')
            plt.ylabel(f"{metric}")
            if metric == "rotation_error_deg":
                y_min = min(values)
                y_max = max(values)
                plt.yticks(np.arange(int(y_min // 10) * 10, y_max + 10, 10))
            try:
                plt.xticks(x_positions, metric_values['metric_label'], rotation=45, ha='right')
            except Exception:
                pass
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{metric}.png"))
            plt.close()

        logger.info(f"Saved plot for {metric} as {metric}.png")

    # Success rate
    successful_registrations = colors.count('green') / len(colors) * 100
    logger.info(f"Number of successful registrations: {successful_registrations:.2f}%")

    rotation_registrations = [1 if metric_values["rotation_error_deg"][i] < thresholds["rotation_error_deg"] else 0
                              for i in range(len(metric_values["rotation_error_deg"]))]
    successful_rotation_registrations = sum(rotation_registrations) / len(rotation_registrations) * 100
    logger.info(f"Number of successful rotation registrations: {successful_rotation_registrations:.2f}%")

    voxel_size = int(input_args.input.split('/')[-1].replace('Voxel', ''))

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
    }

    voxel_size = input_args.input.split('/')[-1]
    number_voxel_size = voxel_size
    logger.info(f"Voxel size: {voxel_size}")
    with open(os.path.join(input_args.input, input_args.output_path, f"success_rate_{voxel_size}.json"), 'w') as file:
            json.dump(success_rate, file, indent=4)
            logger.info(f"Saved metrics to {os.path.join(input_args.input, input_args.output_path, f'success_rate_{voxel_size}.json')}")
