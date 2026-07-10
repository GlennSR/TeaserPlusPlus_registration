"""
Plot a grouped bar chart comparing the localization success rate of
Teaser-only, Teaser+ICP, and Teaser+GICP across multiple voxel sizes.

Usage
-----
python3 plot_compare_success_rates.py \
    --input  result_folders/success_rates/result_success_rates.txt \
    --output ./graphs

Input .txt format
-----------------
Group headers like ``VoxelXXX`` define which TEASER voxel size the following
JSON files belong to.  Each JSON line is classified by the method in its path:

    - Teaser-only  : contains  'Teaser/'  (but NOT 'Teaser+ICP' / 'Teaser+GICP')
    - Teaser+ICP   : contains  'Teaser+ICP'  (but NOT 'Teaser+GICP')
    - Teaser+GICP  : contains  'Teaser+GICP'

Example::

    Voxel200
    ../../Teaser/success_rate_Voxel200.json
    ../../Teaser+GICP/Teaser_Voxel200/success_rate_Voxel250.json
    ../../Teaser+ICP/Teaser_Voxel200/success_rate_Voxel250.json

    Voxel300
    ...

The ``"voxel_size"`` field inside the JSON is **ignored** for grouping —
the header line is the authoritative group key.
Blank lines, lines starting with '#', and lines containing '---' are ignored.

Each JSON must have at least:
    - "voxel_size"
    - "successful_registrations_percentage"
    - "rotation_error_threshold_deg"
    - "translation_error_threshold_mm"

Teaser-only and Teaser+ICP are optional — if absent the chart is still produced
with however many methods are found.
"""

import argparse
import json
import logging
import os
import re

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    from registration.utils.logging import setup_logging
except ImportError:
    def setup_logging(level=logging.INFO):
        logging.basicConfig(
            level=level,
            format="[%(asctime)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

COLOR_TEASER = "#7EC8E3"   # light blue
COLOR_ICP    = "#F0A830"   # orange / gold
COLOR_GICP   = "#4CAF50"   # green

LABEL_TEASER = "TEASER++ only"
LABEL_ICP    = "TEASER++ + ICP"
LABEL_GICP   = "TEASER++ + GICP"


# ──────────────────────────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────────────────────────

def _classify_line(line: str) -> str | None:
    """Return 'teaser', 'icp', 'gicp', or None."""
    l = line.lower()
    if "teaser+gicp" in l:
        return "gicp"
    if "teaser+icp" in l:
        return "icp"
    # Match 'Teaser/' or 'Teaser\' but NOT 'Teaser+...'
    if re.search(r'teaser[/\\]', l):
        return "teaser"
    return None


def parse_success_rate_txt(
    txt_path: str,
) -> dict[str, dict[int, dict]]:
    """
    Parse the .txt file and return a nested dict:
        { method: { voxel_size: json_data } }
    where method is 'teaser', 'icp', or 'gicp'.

    Lines like "Voxel200", "Voxel300", etc. define the current group voxel size.
    All JSON lines that follow (until the next group header) are assigned to that
    voxel size, regardless of the 'voxel_size' field inside the JSON itself.
    """
    results: dict[str, dict[int, dict]] = {"teaser": {}, "icp": {}, "gicp": {}}
    script_dir = os.path.dirname(os.path.abspath(txt_path))
    current_vox: int | None = None

    with open(txt_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "---" in line:
                continue

            # ── Group header, e.g. "Voxel200" ────────────────────────────────
            header_match = re.fullmatch(r'[Vv]oxel(\d+)', line)
            if header_match:
                current_vox = int(header_match.group(1))
                logger.debug(f"Group: Voxel{current_vox}")
                continue

            # ── JSON path line ────────────────────────────────────────────────
            method = _classify_line(line)
            if method is None:
                logger.debug(f"Skipping line (unknown method): {line}")
                continue

            if current_vox is None:
                logger.warning(
                    f"No group header found before this line, skipping: {line}"
                )
                continue

            json_path = line if os.path.isabs(line) else os.path.normpath(
                os.path.join(script_dir, line)
            )

            if not os.path.isfile(json_path):
                logger.warning(f"File not found, skipping: {json_path}")
                continue

            try:
                with open(json_path, "r") as jf:
                    data = json.load(jf)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error in {json_path}: {e}")
                continue

            results[method][current_vox] = data
            logger.debug(f"  [{method}] Voxel{current_vox} ← {json_path}")

    for method, voxels in results.items():
        if voxels:
            logger.info(f"  {method}: voxel sizes found = {sorted(voxels)}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def _make_bar_chart(
    results: dict[str, dict[int, dict]],
    json_key: str,
    ylabel: str,
    title: str,
    output_path: str,
    value_fmt: str = "{:.2f}",
    ylim_top_pad: float = 10.0,
):
    """
    Generic grouped bar chart helper.

    Parameters
    ----------
    results     : nested dict  { method: { voxel_size: json_data } }
    json_key    : key to read from each JSON dict (e.g. 'successful_registrations_percentage')
    ylabel      : Y-axis label
    title       : chart title
    output_path : where to save the PNG
    value_fmt   : format string for the label printed above each bar
    ylim_top_pad: extra space added above the tallest bar
    """
    all_voxels = sorted(set(v for m in results.values() for v in m))
    if not all_voxels:
        logger.error("No data to plot.")
        return

    active_methods = [m for m in ("teaser", "icp", "gicp") if results[m]]
    n_methods = len(active_methods)
    n_voxels  = len(all_voxels)

    method_meta = {
        "teaser": (COLOR_TEASER, LABEL_TEASER),
        "icp":    (COLOR_ICP,    LABEL_ICP),
        "gicp":   (COLOR_GICP,   LABEL_GICP),
    }

    group_width = 0.7
    bar_width   = group_width / n_methods
    x_centers   = np.arange(n_voxels)
    offsets     = np.linspace(
        -(group_width - bar_width) / 2,
         (group_width - bar_width) / 2,
        n_methods,
    )

    fig, ax = plt.subplots(figsize=(max(10, n_voxels * 3.5), 7))

    all_values = []
    legend_patches = []
    for j, method in enumerate(active_methods):
        color, label = method_meta[method]
        values = [
            results[method].get(v, {}).get(json_key, float("nan"))
            for v in all_voxels
        ]
        all_values.extend(v for v in values if not np.isnan(v))
        x_pos = x_centers + offsets[j]
        bars = ax.bar(
            x_pos,
            values,
            width=bar_width * 0.92,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ylim_top_pad * 0.02,
                    value_fmt.format(val),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )
        legend_patches.append(mpatches.Patch(color=color, label=label))

    ax.set_xticks(x_centers)
    ax.set_xticklabels([str(v) for v in all_voxels], fontsize=12)
    ax.set_xlabel("Voxel Size (mm)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, (max(all_values) if all_values else 100) + ylim_top_pad)
    ax.legend(handles=legend_patches, loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_xlim(x_centers[0] - 0.55, x_centers[-1] + 0.55)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_all_charts(
    results: dict[str, dict[int, dict]],
    output_dir: str,
):
    """Produce the three comparison charts and print a summary."""
    all_voxels     = sorted(set(v for m in results.values() for v in m))
    active_methods = [m for m in ("teaser", "icp", "gicp") if results[m]]
    method_meta    = {
        "teaser": LABEL_TEASER,
        "icp":    LABEL_ICP,
        "gicp":   LABEL_GICP,
    }

    # Retrieve threshold info for the success-rate title
    rot_thr, trans_thr = None, None
    for method in active_methods:
        for data in results[method].values():
            rot_thr   = data.get("rotation_error_threshold_deg")
            trans_thr = data.get("translation_error_threshold_mm")
            break
        if rot_thr is not None:
            break

    thr_label = ""
    if rot_thr is not None and trans_thr is not None:
        thr_label = f" (rot < {rot_thr:.0f}deg, transl < {trans_thr:.0f}mm)"

    # ── 1. Success rate ───────────────────────────────────────────────────────
    _make_bar_chart(
        results,
        json_key   = "successful_registrations_percentage",
        ylabel     = "Success Rate (%)",
        title      = f"Localization Success Rate Across Voxel Sizes and Methods{thr_label}",
        output_path= os.path.join(output_dir, "success_rate_comparison.png"),
        value_fmt  = "{:.1f}%",
        ylim_top_pad = 15.0,
    )

    # ── 2. Mean rotation error ────────────────────────────────────────────────
    _make_bar_chart(
        results,
        json_key   = "mean_rotation_error_deg",
        ylabel     = "Mean Rotation Error (degrees)",
        title      = "Mean Rotation Error Across Voxel Sizes and Methods",
        output_path= os.path.join(output_dir, "mean_rotation_error_comparison.png"),
        value_fmt  = "{:.2f}°",
        ylim_top_pad = 2.0,
    )

    # ── 3. Mean translation error ─────────────────────────────────────────────
    _make_bar_chart(
        results,
        json_key   = "mean_translation_error_mm",
        ylabel     = "Mean Translation Error (mm)",
        title      = "Mean Translation Error Across Voxel Sizes and Methods",
        output_path= os.path.join(output_dir, "mean_translation_error_comparison.png"),
        value_fmt  = "{:.1f} mm",
        ylim_top_pad = 30.0,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n── Summary ──────────────────────────────────────────────")
    for v in all_voxels:
        logger.info(f"  Voxel{v}:")
        for method in active_methods:
            data = results[method].get(v)
            if data:
                rate  = data.get("successful_registrations_percentage", float("nan"))
                r_err = data.get("mean_rotation_error_deg",    float("nan"))
                t_err = data.get("mean_translation_error_mm",  float("nan"))
                logger.info(
                    f"    {method_meta[method]}: "
                    f"success={rate:.1f}%  "
                    f"rot={r_err:.2f}°  "
                    f"transl={t_err:.1f} mm"
                )
            else:
                logger.info(f"    {method_meta[method]}: N/A")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot a grouped bar chart of localization success rate comparing "
            "TEASER++ only, TEASER++ + ICP, and TEASER++ + GICP across voxel sizes."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        metavar="TXT_FILE",
        help=(
            "Path to a .txt file listing success_rate JSON files. "
            "Each line is classified as Teaser-only, Teaser+ICP, or Teaser+GICP "
            "based on the path. Voxel size is read from the JSON 'voxel_size' field."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./graphs",
        help=(
            "Output directory where the PNG files will be saved "
            "(default: ./graphs). Three files are produced: "
            "success_rate_comparison.png, mean_rotation_error_comparison.png, "
            "mean_translation_error_comparison.png."
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO).",
    )
    args = parser.parse_args()

    setup_logging(getattr(logging, args.verbose))

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    results = parse_success_rate_txt(args.input)

    if not any(results.values()):
        logger.error("No valid data found in the input file.")
        return

    plot_all_charts(results, args.output)


if __name__ == "__main__":
    main()
