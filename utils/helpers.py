"""Helper utilities.

This file consolidates a small set of general-purpose utilities that were
previously split across `utils.misc` and `utils.helpers`. Callers should
import directly from `utils.helpers`.
"""

import os
import shutil
import logging
import numpy as np


def denormalize_image_for_display(tensor_image):
    """Reverses ImageNet normalization for display.

    Expects a CHW numpy array or tensor-like that can be transposed.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    denormalized_img = (
        ((tensor_image.transpose(1, 2, 0) * std) + mean) * 255.0
    ).astype(np.uint8)
    return denormalized_img


def extract_phase_and_defect(filepath: str, class_name: str):
    """Infer phase ('train'|'test') and defect/category name from a filepath.

    This is defensive: it tries to find the `class_name` component in the path
    and otherwise falls back to locating 'train' or 'test' segments.
    Returns (phase, defect) where defect is e.g. 'good' or a defect-type name.
    """
    parts = filepath.split(os.sep)
    phase = None
    defect = None
    # Try to find class_name in path parts
    if class_name in parts:
        idx = parts.index(class_name)
        # Expect layout: .../<class_name>/<phase>/<defect>/<file>
        if idx + 1 < len(parts):
            phase = parts[idx + 1]
        if idx + 2 < len(parts):
            defect = parts[idx + 2]
    # Fallback: locate 'train' or 'test' in path
    if phase is None:
        for p in ("train", "test"):
            if p in parts:
                phase = p
                pi = parts.index(p)
                if pi + 1 < len(parts):
                    defect = parts[pi + 1]
                break
    # Last resort: use parent directory name as defect and 'train' as phase for training files
    if phase is None:
        phase = "train"
    if defect is None:
        # parent folder name
        defect = os.path.basename(os.path.dirname(filepath)) or "unknown"
    return phase, defect


def save_and_compress_image_splits(class_save_dir: str, class_name: str, train_dataset, test_dataset):
    """Copy original image files used for train/test into a structured folder, compress it, and remove the folder.

    The structure created under `class_save_dir` is:
      saved_image_splits/train/<defect_type>/*
      saved_image_splits/test/<defect_type>/*

    After copying, a tar.gz archive `<class_name>_image_splits.tar.gz` is created in `class_save_dir`
    and the temporary `saved_image_splits` folder is deleted to save space.
    """
    base_out = os.path.join(class_save_dir, "saved_image_splits")
    try:
        if os.path.exists(base_out):
            shutil.rmtree(base_out)
        for phase_name, dataset in (("train", train_dataset), ("test", test_dataset)):
            for idx, fp in enumerate(getattr(dataset, "image_filepaths", [])):
                _, defect = extract_phase_and_defect(fp, class_name)
                dest_phase = phase_name
                dest_dir = os.path.join(base_out, dest_phase, defect)
                os.makedirs(dest_dir, exist_ok=True)
                try:
                    shutil.copy2(fp, dest_dir)
                except Exception as e:
                    logging.warning(f"Failed to copy '{fp}' to '{dest_dir}': {e}")

        # Create compressed archive (tar.gz)
        archive_base = os.path.join(class_save_dir, f"{class_name}_image_splits")
        archive_path = shutil.make_archive(archive_base, "gztar", root_dir=base_out)
        logging.info(f"Saved and compressed image splits to: {archive_path}")
    except Exception as e:
        logging.exception(f"Failed to save/compress image splits: {e}")
    finally:
        # Remove temporary folder if exists
        try:
            if os.path.exists(base_out):
                shutil.rmtree(base_out)
        except Exception:
            logging.warning("Could not remove temporary saved_image_splits directory.")