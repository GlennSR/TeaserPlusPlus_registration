"""
Plot rotation and translation error boxplots comparing Teaser-only vs Teaser+ICP vs Teaser+GICP
across multiple voxel sizes.

Usage
-----
python3 plot_boxplots.py \
    --input  result_folders/result_IndividualScans.txt \
    --output ./graphs

The input .txt file may contain multiple sections separated by blank lines or
comment lines (starting with #, or containing "---").
Each relevant line must contain "Teaser/VoxelXXX", "Teaser+ICP/VoxelXXX" or
"Teaser+GICP/VoxelXXX" in its path, e.g.:

    ../../test_data/.../Teaser/Voxel200
    ../../test_data/.../Teaser+ICP/Voxel200
    ../../test_data/.../Teaser+GICP/Voxel200

Pairs are matched automatically by voxel size.
Teaser+ICP is optional — if absent, only Teaser and Teaser+GICP are plotted.

Each folder must contain JSON files named like "0_metrics.json", "1_metrics.json",
etc., each with at least:
    - "rotation_error_deg"
    - "translation_error"
"""

import argparse
import json
import logging
import os
import re

import matplotlib.pyplot as plt
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
# Parsing the input .txt file
# ──────────────────────────────────────────────────────────────────────────────

def parse_folder_pairs_from_txt(txt_path: str) -> list[tuple[int, str, str | None, str]]:
    """
    Read a .txt file and extract (voxel_size, teaser_dir, icp_dir, gicp_dir) tuples.

    Lines are classified as:
      - Teaser-only  : path contains 'Teaser/Voxel'    (but NOT 'Teaser+ICP' / 'Teaser+GICP')
      - Teaser+ICP   : path contains 'Teaser+ICP/Voxel' (but NOT 'Teaser+GICP')
      - Teaser+GICP  : path contains 'Teaser+GICP/Voxel'

    Comment/separator lines (starting with '#', containing '---', or blank) are ignored.
    Teaser+ICP is optional — voxels that have no ICP folder will have icp_dir=None.
    """
    teaser_dirs: dict[int, str] = {}
    icp_dirs:    dict[int, str] = {}
    gicp_dirs:   dict[int, str] = {}

    script_dir = os.path.dirname(os.path.abspath(txt_path))

    with open(txt_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "---" in line:
                continue

            folder = line if os.path.isabs(line) else os.path.normpath(
                os.path.join(script_dir, line)
            )

            vox_match = re.search(r'[Vv]oxel(\d+)', line)
            if not vox_match:
                logger.debug(f"Skipping line (no voxel size found): {line}")
                continue
            vox = int(vox_match.group(1))

            if "Teaser+GICP" in line or "teaser+gicp" in line.lower():
                gicp_dirs[vox] = folder
            elif "Teaser+ICP" in line or "teaser+icp" in line.lower():
                icp_dirs[vox] = folder
            elif re.search(r'Teaser[/\\]', line):
                teaser_dirs[vox] = folder
            else:
                logger.debug(f"Skipping line (not Teaser, Teaser+ICP or Teaser+GICP): {line}")

    # Require at minimum a Teaser and a GICP match
    common_voxels = sorted(set(teaser_dirs) & set(gicp_dirs))
    missing = (set(teaser_dirs) | set(gicp_dirs)) - set(common_voxels)
    if missing:
        logger.warning(
            f"The following voxel sizes are missing Teaser or GICP data and will be skipped: "
            f"{sorted(missing)}"
        )

    pairs = [
        (v, teaser_dirs[v], icp_dirs.get(v), gicp_dirs[v])
        for v in common_voxels
    ]

    has_icp = any(p[2] is not None for p in pairs)
    logger.info(
        f"Found {len(pairs)} matched voxel-size group(s): {[p[0] for p in pairs]} "
        f"(Teaser+ICP data: {'yes' if has_icp else 'not found'})"
    )
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_voxel_size(folder_path: str) -> int:
    """Extract the numeric voxel size from a folder path ending in 'VoxelXXX'."""
    match = re.search(r'[Vv]oxel(\d+)', os.path.basename(folder_path.rstrip("/\\")))
    if not match:
        raise ValueError(
            f"Cannot extract voxel size from folder name: '{folder_path}'. "
            "Folder must end with 'VoxelXXX'."
        )
    return int(match.group(1))


def whisker_bounds(data: list[float]) -> tuple[float, float]:
    """Return (lower_whisker, upper_whisker) using the standard 1.5×IQR rule."""
    if not data:
        return (0.0, 0.0)
    arr = np.asarray(data)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lower = float(arr[arr >= q1 - 1.5 * iqr].min())
    upper = float(arr[arr <= q3 + 1.5 * iqr].max())
    return lower, upper


def load_errors_from_folder(folder_path: str) -> tuple[list[float], list[float]]:
    """Load rotation_error_deg and translation_error from all JSON metric files in a folder."""
    json_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".json")],
        key=lambda f: int(re.search(r'\d+', f).group()) if re.search(r'\d+', f) else 0,
    )
    if not json_files:
        logger.warning(f"No JSON files found in: {folder_path}")
        return [], []

    rot_errors, trans_errors = [], []
    for fname in json_files:
        fpath = os.path.join(folder_path, fname)
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            if data["translation_error"] < 8000:  # Filter out extreme outliers
                rot_errors.append(data["rotation_error_deg"])
                trans_errors.append(data["translation_error"])
        except (KeyError, json.JSONDecodeError) as e:
            logger.warning(f"Skipping {fpath}: {e}")

    logger.info(
        f"  Loaded {len(rot_errors)} samples from {os.path.basename(folder_path)}"
    )
    return rot_errors, trans_errors


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

# Colours matching the reference images
COLOR_TEASER = "#7EC8E3"   # light blue  (Teaser only)
COLOR_ICP    = "#FFA500"   # orange      (Teaser + ICP)
COLOR_GICP   = "#4CAF50"   # green       (Teaser + GICP)

LABEL_TEASER = "TEASER++ only"
LABEL_ICP    = "TEASER++ + ICP"
LABEL_GICP   = "TEASER++ + GICP"


def make_boxplot(
    ax: plt.Axes,
    voxel_sizes: list[int],
    data_teaser: list[list[float]],
    data_icp: list[list[float] | None],
    data_gicp: list[list[float]],
    ylabel: str,
    title: str,
    y_margin_top: float = 0.0,
    y_margin_bottom: float = 0.0,
):
    """Draw side-by-side boxplots for each voxel size on *ax*.

    *data_icp* may contain ``None`` for voxel sizes that have no ICP results;
    those groups will only show two boxes.

    Y-axis limits are set to [min_whisker - y_margin_bottom, max_whisker + y_margin_top].
    """
    n = len(voxel_sizes)
    has_icp = any(d is not None for d in data_icp)

    if has_icp:
        group_width = 0.75
        box_width   = group_width / 3 * 0.85
        offsets     = [-group_width / 3, 0.0, +group_width / 3]  # T, ICP, GICP
    else:
        group_width = 0.6
        box_width   = group_width / 2 * 0.85
        offsets     = [-group_width / 4, None, +group_width / 4]  # T, (skip), GICP

    x_ticks = np.arange(1, n + 1)

    handles = []
    for i, (vox, t_data, icp_data, g_data) in enumerate(
        zip(voxel_sizes, data_teaser, data_icp, data_gicp)
    ):
        def _boxplot(data, x_pos, color):
            bp = ax.boxplot(
                data,
                positions=[x_pos],
                widths=box_width,
                patch_artist=True,
                showfliers=True,
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=dict(color="#444"),
                capprops=dict(color="#444"),
                flierprops=dict(marker="", markersize=3, alpha=0.5, markerfacecolor=color),
            )
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.85)
            return bp

        x_t = x_ticks[i] + offsets[0]
        x_g = x_ticks[i] + offsets[2]
        bp_t = _boxplot(t_data, x_t, COLOR_TEASER)
        bp_g = _boxplot(g_data, x_g, COLOR_GICP)

        if i == 0:
            handles.append(bp_t["boxes"][0])

        if icp_data is not None:
            x_icp = x_ticks[i] + offsets[1]
            bp_icp = _boxplot(icp_data, x_icp, COLOR_ICP)
            if i == 0:
                handles.append(bp_icp["boxes"][0])

        if i == 0:
            handles.append(bp_g["boxes"][0])

    legend_labels = [LABEL_TEASER]
    if has_icp:
        legend_labels.append(LABEL_ICP)
    legend_labels.append(LABEL_GICP)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(v) for v in voxel_sizes], fontsize=11)
    ax.set_xlabel("Voxel Size (mm)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(handles, legend_labels, loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_xlim(x_ticks[0] - 0.6, x_ticks[-1] + 0.6)

    # Compute Y limits from whisker extents across all datasets
    all_lowers, all_uppers = [], []
    for datasets in [data_teaser, data_icp, data_gicp]:
        for d in datasets:
            if d:
                lo, hi = whisker_bounds(d)
                all_lowers.append(lo)
                all_uppers.append(hi)
    if all_lowers and all_uppers:
        ax.set_ylim(
            min(all_lowers) - y_margin_bottom,
            max(all_uppers) + y_margin_top,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot rotation and translation error boxplots comparing "
            "TEASER++ only, TEASER++ + ICP (optional), and TEASER++ + GICP "
            "across voxel sizes."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        metavar="TXT_FILE",
        help=(
            "Path to a .txt file listing the metric folders. "
            "Lines containing 'Teaser/VoxelXXX' are treated as TEASER-only results; "
            "lines containing 'Teaser+ICP/VoxelXXX' as TEASER+ICP results (optional); "
            "lines containing 'Teaser+GICP/VoxelXXX' as TEASER+GICP results. "
            "Entries are matched automatically by voxel size. "
            "Blank lines, lines with '---', and lines starting with '#' are ignored."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./graphs",
        help="Directory where the output PNG files will be saved (default: ./graphs).",
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

    groups = parse_folder_pairs_from_txt(args.input)

    if not groups:
        logger.error("No matched Teaser / Teaser+GICP folder entries found in the input file.")
        return

    # Load data
    voxel_sizes  = []
    rot_teaser,  rot_icp,  rot_gicp   = [], [], []
    trans_teaser, trans_icp, trans_gicp = [], [], []

    for vox, t_dir, icp_dir, g_dir in groups:
        logger.info(f"Loading Voxel{vox} — TEASER:      {t_dir}")
        r_t, tr_t = load_errors_from_folder(t_dir)

        if icp_dir is not None:
            logger.info(f"Loading Voxel{vox} — TEASER+ICP:  {icp_dir}")
            r_icp, tr_icp = load_errors_from_folder(icp_dir)
        else:
            r_icp, tr_icp = None, None

        logger.info(f"Loading Voxel{vox} — TEASER+GICP: {g_dir}")
        r_g, tr_g = load_errors_from_folder(g_dir)

        if not r_t or not r_g:
            logger.warning(f"Skipping Voxel{vox}: empty data in Teaser or Teaser+GICP folder.")
            continue

        voxel_sizes.append(vox)
        rot_teaser.append(r_t);   rot_icp.append(r_icp);   rot_gicp.append(r_g)
        trans_teaser.append(tr_t); trans_icp.append(tr_icp); trans_gicp.append(tr_g)

    if not voxel_sizes:
        logger.error("No data loaded. Check your input folders.")
        return

    os.makedirs(args.output, exist_ok=True)

    # ── Rotation error plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, len(voxel_sizes) * 3), 7))
    make_boxplot(
        ax, voxel_sizes, rot_teaser, rot_icp, rot_gicp,
        ylabel="Rotation Error (degrees)",
        title="Localization Rotation Error Comparison Across Voxel Sizes",
        y_margin_top=5.0,
        y_margin_bottom=1,
    )
    fig.tight_layout()
    rot_out = os.path.join(args.output, "rotation_error_boxplot.png")
    fig.savefig(rot_out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {rot_out}")

    # ── Translation error plot ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, len(voxel_sizes) * 3), 7))
    make_boxplot(
        ax, voxel_sizes, trans_teaser, trans_icp, trans_gicp,
        ylabel="Translation Error (mm)",
        title="Localization Translation Error Comparison Across Voxel Sizes",
        y_margin_top=50.0,
        y_margin_bottom=20.0,
    )
    fig.tight_layout()
    trans_out = os.path.join(args.output, "translation_error_boxplot.png")
    fig.savefig(trans_out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {trans_out}")

    # ── Summary statistics ───────────────────────────────────────────────────
    logger.info("\n── Summary ──────────────────────────────────────────────")
    for i, vox in enumerate(voxel_sizes):
        icp_rot_str   = f", TEASER+ICP: median={np.median(rot_icp[i]):.2f}°"   if rot_icp[i]   is not None else ""
        icp_trans_str = f", TEASER+ICP: median={np.median(trans_icp[i]):.1f} mm" if trans_icp[i] is not None else ""
        logger.info(f"  Voxel{vox}:")
        logger.info(f"    Rotation    — TEASER: median={np.median(rot_teaser[i]):.2f}°"
                    f"{icp_rot_str}, TEASER+GICP: median={np.median(rot_gicp[i]):.2f}°")
        logger.info(f"    Translation — TEASER: median={np.median(trans_teaser[i]):.1f} mm"
                    f"{icp_trans_str}, TEASER+GICP: median={np.median(trans_gicp[i]):.1f} mm")


if __name__ == "__main__":
    main()
