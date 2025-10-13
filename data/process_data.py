"""
================================================================================
Image Dataset Preprocessing and Sorting Script
================================================================================

Overview:
---------
This script is designed to automate the process of preparing image datasets for
machine learning, particularly for anomaly detection tasks. It reads metadata
from CSV files, applies a series of optional image preprocessing steps, and then
organizes the images into a structured 'train' and 'test' directory format
compatible with frameworks like Anomalib.

Key Features:
-------------
- Reads multiple CSV files to determine how to categorize images.
- Copies and sorts images into 'train/good' and 'test/good' folders.
- Automatically creates subdirectories for different rejection reasons based on
  comments in the CSV, e.g., 'test/blurry', 'test/scratched'.
- Applies optional image transformations such as grayscale conversion, Canny
  edge detection, and histogram equalization.
- Splits the 'good' images into training and testing sets based on a specified ratio.
- Logs all operations to a file ('image_processing.log') and the console.

Prerequisites:
--------------
1. A root **Input Directory** containing all necessary files.
2. Inside the input directory, you need:
   - One or more **CSV files**, each detailing a set of images.
   - The corresponding **Image Folders**, named as specified in the map file.
3. A **JSON Map File** that links each CSV file to its respective image folder.

Expected Directory Structure (Example):
---------------------------------------
/path/to/your/data/ (This is your --input-dir)
|
|-- round-1.csv
|-- round-2.csv
|-- map.json
|
|-- round-1-images/
|   |-- 001.jpg
|   |-- 002.jpg
|   `-- ...
|
`-- round-2-images/
    |-- 001.jpg
    |-- 002.jpg
    `-- ...

JSON Map File (`map.json`) Format:
----------------------------------
The map file is a simple JSON object where keys are the CSV filenames and
values are the names of the folders containing the corresponding images.

Example `map.json`:
{
  "round-1.csv": "round-1-images",
  "round-2.csv": "round-2-images"
}

Usage:
------
Run the script from your terminal. You must provide the input directory,
the output directory, and the path to the JSON map file.

Basic Command:
--------------
python your_script_name.py -i /path/to/data -o /path/to/output -m /path/to/data/map.json

Command with Preprocessing:
---------------------------
python your_script_name.py -i /path/to/data -o /path/to/output_processed -m /path/to/data/map.json --grayscale --equalize-local

Example of usage:
------------------------------
python process_data.py -i data/BowTie/round-2/ -o data/BowTie-New/original -m data/map.json

python process_data.py -i data/BowTie/round-2/ -o data/BowTie-New/grayscale -m data/map.json --grayscale

python process_data.py --combine 116 117 118 119 -i data/BowTie/round-2/ -o data/BowTie-New/combined -m data/map.json

Command-Line Arguments:
-----------------------
-i, --input-dir (required)    : Path to the root directory containing your
                                source images, CSVs, and map file.
-o, --output-dir (required)   : Path to the base directory where the structured
                                output folders will be created.
-m, --map-file (required)     : Path to the JSON file that maps CSVs to image folders.
--split-ratio (optional)      : Fraction of 'good' images to move to the test set.
                                (Default: 0, meaning 0% go to test).

--- Preprocessing Flags (optional) ---
--grayscale                   : Convert all images to grayscale.
--canny                       : Apply Canny edge detection.
--canny-sigma                 : Set the sigma value for the Canny filter. (Default: 1.0).
--equalize-global             : Apply global histogram equalization.
--equalize-local              : Apply local (adaptive) histogram equalization.
--equalize-disk-size          : Set the disk size for local equalization. (Default: 30).

"""

import pandas as pd
import shutil
import logging
import random
import json
import argparse
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# --- Image Processing Imports ---
import numpy as np
import skimage as ski
from skimage import io, exposure, img_as_ubyte
from skimage.morphology import disk

# --- Logging Setup ---
# (Unchanged)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("image_processing.log"), logging.StreamHandler()],
)


# =============================================================================
# == PREPROCESSING & NORMALIZATION FUNCTIONS
# =============================================================================
# (These functions are unchanged)
def apply_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts an RGB image to grayscale."""
    if image.ndim == 3:
        return ski.color.rgb2gray(image)
    return image


def apply_canny(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Applies the Canny edge detection filter."""
    image_gray = apply_grayscale(image)
    return ski.feature.canny(image_gray, sigma=sigma)


def apply_equalize_global(image: np.ndarray) -> np.ndarray:
    """Applies global histogram equalization."""
    image_gray = apply_grayscale(image)
    image_ubyte = img_as_ubyte(image_gray)
    return exposure.equalize_hist(image_ubyte)


def apply_equalize_local(image: np.ndarray, disk_size: int = 30) -> np.ndarray:
    """Applies local (adaptive) histogram equalization."""
    image_gray = apply_grayscale(image)
    image_ubyte = img_as_ubyte(image_gray)
    footprint = disk(disk_size)
    return ski.filters.rank.equalize(image_ubyte, footprint=footprint)


def normalize_color(image: np.ndarray, reference_image: np.ndarray) -> np.ndarray:
    """Adjusts the color distribution of an image to match a reference image."""
    is_multichannel = image.ndim == 3
    return exposure.match_histograms(
        image, reference_image, channel_axis=-1 if is_multichannel else None
    )


# =============================================================================
# == CORE SCRIPT LOGIC (MODE 1: SORTING)
# =============================================================================
def process_image_dataset(args: argparse.Namespace):
    """Main function to process, preprocess, and sort image datasets."""
    # (This function is unchanged)
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    image_map = args.image_map
    processing_steps = [
        k
        for k, v in vars(args).items()
        if v is True
        and k in ["grayscale", "canny", "equalize_global", "equalize_local"]
    ]
    logging.info(f"Starting image processing from source: {input_path}")
    logging.info(f"Output will be saved to: {output_path}")
    if not input_path.is_dir():
        logging.error(f"Input path '{input_path}' does not exist. Aborting.")
        return
    for csv_filename, source_folder in image_map.items():
        logging.info(f"--- Processing dataset for '{csv_filename}' ---")
        csv_path = input_path / csv_filename
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logging.error(f"Failed to read CSV {csv_path}: {e}")
            continue
        output_dir = output_path / Path(csv_filename).stem
        train_dir = output_dir / "train" / "good"
        test_accept_dir = output_dir / "test" / "good"
        default_reject_dir = output_dir / "test" / "reject"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_accept_dir.mkdir(parents=True, exist_ok=True)
        default_reject_dir.mkdir(parents=True, exist_ok=True)
        source_image_path_base = input_path / source_folder
        for _, row in df.iterrows():
            img_num = row.get("img")
            img_type = str(row.get("type", "")).strip()
            source_img_path = source_image_path_base / f"{int(img_num):03d}.jpg"
            if not source_img_path.is_file():
                logging.warning(f"Source image not found: {source_img_path}. Skipping.")
                continue
            dest_path = None
            if img_type == "A":
                dest_path = train_dir / source_img_path.name
            elif img_type == "R":
                comment = row.get("comments")
                if pd.notna(comment) and str(comment).strip():
                    folder_name = (
                        str(comment).strip().lower().replace(" ", "_").replace("/", "-")
                    )
                    specific_reject_dir = output_dir / "test" / folder_name
                    specific_reject_dir.mkdir(exist_ok=True)
                    dest_path = specific_reject_dir / source_img_path.name
                else:
                    dest_path = default_reject_dir / source_img_path.name
            if not dest_path:
                continue
            if not processing_steps:
                shutil.copyfile(source_img_path, dest_path)
            else:
                try:
                    image = io.imread(source_img_path)
                    if args.grayscale:
                        image = apply_grayscale(image)
                    if args.equalize_global:
                        image = apply_equalize_global(image)
                    if args.equalize_local:
                        image = apply_equalize_local(
                            image, disk_size=args.equalize_disk_size
                        )
                    if args.canny:
                        image = apply_canny(image, sigma=args.canny_sigma)
                    io.imsave(dest_path, img_as_ubyte(image))
                except Exception as e:
                    logging.error(f"Failed to process and save {source_img_path}: {e}")
        accepted_images = list(train_dir.glob("*.jpg"))
        random.shuffle(accepted_images)
        num_to_move = int(len(accepted_images) * args.split_ratio)
        if num_to_move > 0:
            images_to_move = random.sample(accepted_images, num_to_move)
            for image_path in images_to_move:
                shutil.move(str(image_path), test_accept_dir / image_path.name)
        for dir_path in [test_accept_dir, default_reject_dir]:
            try:
                if dir_path.exists() and not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    logging.info(f"Removed empty directory: {dir_path}")
            except OSError as e:
                logging.warning(f"Could not remove directory {dir_path}: {e}")


# =============================================================================
# == CORE SCRIPT LOGIC (MODE 2: COMBINING)
# =============================================================================


def combine_folders(args: argparse.Namespace):
    """Copies and combines all 'good' and defect images, renaming with a prefix."""
    folders_to_combine = args.combine
    input_base_path = Path(args.input_dir)
    output_base_path = Path(args.output_dir)

    logging.info("--- Starting Combine Mode ---")
    logging.info(f"Folders to combine: {', '.join(folders_to_combine)}")

    # 1. Define destination paths
    dest_dir = output_base_path
    dest_train_dir = dest_dir / "train" / "good"
    dest_test_base_dir = dest_dir / "test"
    dest_train_dir.mkdir(parents=True, exist_ok=True)
    dest_test_base_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output will be saved to: {dest_dir}")

    # 2. Collect all image paths, storing the folder name as a prefix
    train_good_paths = []
    anomaly_paths = defaultdict(list)

    for folder_name in folders_to_combine:
        # Try exact name first; if not found try prefix matching (e.g. "116" -> "116_zoomed_in")
        source_folder = input_base_path / folder_name
        if not source_folder.is_dir():
            matches = [
                p
                for p in input_base_path.iterdir()
                if p.is_dir() and p.name.startswith(f"{folder_name}")
            ]
            if matches:
                source_folder = matches[0]
                logging.info(f"Resolved '{folder_name}' -> '{source_folder.name}'")
            else:
                logging.warning(
                    f"Source folder '{folder_name}' not found (and no prefix match). Skipping."
                )
                continue

        # Collect ALL good images with their prefix
        train_good_paths.extend(
            [(folder_name, p) for p in source_folder.glob("train/good/*.jpg")]
        )
        train_good_paths.extend(
            [(folder_name, p) for p in source_folder.glob("test/good/*.jpg")]
        )

        # Collect anomaly images with their prefix
        source_test_dir = source_folder / "test"
        if source_test_dir.is_dir():
            for item in source_test_dir.iterdir():
                if item.is_dir() and item.name != "good":
                    defect_name = item.name
                    anomaly_paths[defect_name].extend(
                        [(folder_name, p) for p in item.glob("*.jpg")]
                    )

    total_good = len(train_good_paths)
    total_anomaly = sum(len(paths) for paths in anomaly_paths.values())

    if total_good + total_anomaly == 0:
        logging.error("No images found in the specified folders. Aborting.")
        return
    logging.info(f"Found {total_good} good images and {total_anomaly} anomaly images.")

    # 3. Set up normalization reference image if requested
    reference_image = None
    if args.normalize:
        logging.info("Normalization is enabled.")
        all_paths = [p for _, p in train_good_paths] + [
            p for paths in anomaly_paths.values() for _, p in paths
        ]
        if not all_paths:
            logging.warning("No images found to select a normalization reference from.")
        else:
            try:
                if args.norm_reference:
                    reference_path = Path(args.norm_reference)
                    if not reference_path.is_file():
                        raise FileNotFoundError
                    logging.info(f"Using provided reference image: {reference_path}")
                else:
                    reference_path = all_paths[0]
                    logging.info(
                        f"Using first image as reference: {reference_path.name}"
                    )
                reference_image = io.imread(reference_path)
            except Exception as e:
                logging.error(f"Could not load reference image: {e}. Aborting.")
                return

    # 4. Helper function to process and COPY images with new prefixed name
    def process_and_copy(image_data, destination_folder, pbar_desc):
        destination_folder.mkdir(exist_ok=True)
        for prefix, img_path in tqdm(image_data, desc=pbar_desc):
            try:
                # Create new filename with prefix
                new_filename = f"{prefix}_{img_path.name}"
                dest_path = destination_folder / new_filename

                if args.normalize and reference_image is not None:
                    current_image = io.imread(img_path)
                    normalized_image = normalize_color(current_image, reference_image)
                    io.imsave(dest_path, img_as_ubyte(normalized_image))
                else:
                    shutil.copy(str(img_path), dest_path)
            except Exception as e:
                logging.error(f"Failed to process image {img_path}: {e}")

    # 5. Process all categories of images
    process_and_copy(train_good_paths, dest_train_dir, "Processing All Good")

    for defect_name, paths in anomaly_paths.items():
        dest_defect_dir = dest_test_base_dir / defect_name
        process_and_copy(paths, dest_defect_dir, f"Processing Test/{defect_name}")

    logging.info("Successfully combined and copied all images.")


# =============================================================================
# == MAIN EXECUTION
# =============================================================================


def main():
    # (This function is unchanged)
    parser = argparse.ArgumentParser(
        description="Process, sort, and combine image datasets for deep learning.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--combine",
        nargs="+",
        help="Activate Combine Mode. List folder names to merge (e.g., 116 117).",
    )
    parser.add_argument(
        "-i", "--input-dir", required=True, help="Root directory for input data."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Base directory where output will be saved.",
    )
    sort_group = parser.add_argument_group("Sorting Mode Options")
    sort_group.add_argument(
        "-m",
        "--map-file",
        help="Path to JSON file mapping CSVs to image folders (required for sorting).",
    )
    sort_group.add_argument(
        "--split-ratio",
        type=float,
        default=0.0,
        help="Fraction of 'good' images for the test set (default: 0.0).",
    )
    combine_group = parser.add_argument_group("Combine Mode Options")
    combine_group.add_argument(
        "--normalize",
        action="store_true",
        help="Enable color normalization during the combine step.",
    )
    combine_group.add_argument(
        "--norm-reference", help="Path to an image for color normalization reference."
    )
    proc_group = parser.add_argument_group("Preprocessing Options (for Sorting Mode)")
    proc_group.add_argument(
        "--grayscale", action="store_true", help="Convert images to grayscale."
    )
    proc_group.add_argument(
        "--canny", action="store_true", help="Apply Canny edge detection."
    )
    proc_group.add_argument(
        "--canny-sigma",
        type=float,
        default=1.0,
        help="Sigma for Canny filter (default: 1.0).",
    )
    proc_group.add_argument(
        "--equalize-global",
        action="store_true",
        help="Apply global histogram equalization.",
    )
    proc_group.add_argument(
        "--equalize-local",
        action="store_true",
        help="Apply local (adaptive) histogram equalization.",
    )
    proc_group.add_argument(
        "--equalize-disk-size",
        type=int,
        default=30,
        help="Disk size for local equalization (default: 30).",
    )
    args = parser.parse_args()
    if args.combine:
        combine_folders(args)
    else:
        if not args.map_file:
            parser.error("--map-file is required when not in --combine mode.")
        try:
            with open(args.map_file, "r") as f:
                args.image_map = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            parser.error(f"Could not read or parse map file '{args.map_file}': {e}")
        process_image_dataset(args)
    logging.info("--- Script finished. ---")


if __name__ == "__main__":
    main()
