# run_experiment.py (Version 3)

# --- Core Libraries ---
import os
import random
import pickle
import logging
import argparse
from collections import OrderedDict

# --- Numerical and Scientific Libraries ---
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy import ndimage

# --- Machine Learning and Deep Learning ---
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import (
    wide_resnet50_2,
    resnet18,
    efficientnet_b5,
    EfficientNet_B5_Weights,
)
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch.amp import autocast
from sklearn.metrics import average_precision_score
import scipy.stats as sps

# --- Visualization ---
import matplotlib.pyplot as plt

# --- Custom Dataloader ---
# Assumes bowtie.py is in a 'datasets' subfolder
import datasets.bowtie as bowtie
import copy

# Bring commonly used helpers from the utils package to keep this script slim.
from utils.helpers import (
    get_batch_embeddings,
    denormalize_image_for_display,
    concatenate_embeddings,
    plot_summary_visuals,
    plot_mean_anomaly_maps,
    plot_individual_visualizations,
    plot_patch_score_distributions,
)

INTERMEDIATE_FEATURE_MAPS = []

def setup_logging(log_path, log_name):
    """Configures a master logger for the entire experiment run."""
    log_file = os.path.join(log_path, f"{log_name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    logging.info(f"Logging initialized. Log file at: {log_file}")

def hook_function(module, input, output):
    """A simple hook that appends the output of a layer to a global list."""
    INTERMEDIATE_FEATURE_MAPS.append(output)

def run_class_processing(args, class_name, model, device, random_feature_indices):
    """Main function to process a single class: load data, train, test, and save results."""
    global INTERMEDIATE_FEATURE_MAPS

    compute_device = torch.device("cpu")
    if args.stats_on_gpu:
        if device.type == "cuda":
            compute_device = device
            logging.info("Mean/Covariance calculations will be performed on the GPU.")
        else:
            logging.info(
                "`--stats_on_gpu` was set, but no CUDA device found. Using CPU."
            )
    else:
        logging.info(
            "Mean/Covariance calculations will be performed on the CPU (default)."
        )

    logging.info(f"--- Starting processing for CLASS: {class_name} ---")

    # --- 1. Setup & Dataloading ---
    class_save_dir = os.path.join(args.master_save_dir, class_name)
    os.makedirs(class_save_dir, exist_ok=True)
    data_manager = bowtie.BowtieDataManager(
        dataset_path=args.data_path,
        class_name=class_name,
        resize=args.resize,
        cropsize=args.cropsize,
        seed=args.seed,
        augmentations_enabled=args.augmentations_enabled,
        horizontal_flip=args.horizontal_flip,
        vertical_flip=args.vertical_flip,
        augmentation_prob=args.augmentation_prob,
    )

    # --- Verbose Logging Block ---
    logging.info("------------------- EXPERIMENT CONFIGURATION -------------------")
    logging.info(f"[DATASET] Class Name: {class_name}")
    logging.info(f"[DATASET] Data Path: {args.data_path}")
    logging.info(f"[DATASET] Train Set Size: {len(data_manager.train)} images")
    logging.info(f"[DATASET] Test Set Size: {len(data_manager.test)} images")
    normal_test_count = sum(1 for label in data_manager.test.labels if label == 0)
    abnormal_test_count = len(data_manager.test.labels) - normal_test_count
    logging.info(
        f"[DATASET] Test Set Composition: {normal_test_count} Normal, {abnormal_test_count} Abnormal"
    )
    logging.info(f"[PREPROC] Image Resize Target: {args.resize}")
    logging.info(f"[PREPROC] Center Crop Size: {args.cropsize}")
    if args.augmentations_enabled:
        logging.info("[AUGMENT] Status: ENABLED")
        logging.info(f"[AUGMENT] Horizontal Flip: {args.horizontal_flip}")
        logging.info(f"[AUGMENT] Vertical Flip: {args.vertical_flip}")
        logging.info(f"[AUGMENT] Application Probability: {args.augmentation_prob}")
    else:
        logging.info("[AUGMENT] Status: DISABLED")
    logging.info(f"[MODEL] Architecture: {args.model_architecture}")
    dummy_input = torch.randn(1, 3, args.cropsize, args.cropsize).to(device)
    with torch.no_grad():
        _ = model(dummy_input)
    map_shapes = [f.shape for f in INTERMEDIATE_FEATURE_MAPS]
    final_h, final_w = map_shapes[-1][-2], map_shapes[-1][-1]
    total_patches = final_h * final_w
    logging.info(f"[MODEL] Intermediate Feature Map Shapes (B,C,H,W): {map_shapes}")
    logging.info(f"[MODEL] Final Anomaly Map Grid Size (H x W): {final_h} x {final_w}")
    logging.info(f"[MODEL] Total Patches per Image: {total_patches}")
    INTERMEDIATE_FEATURE_MAPS = []
    total_dim = sum(shape[1] for shape in map_shapes)
    logging.info(
        f"[EMBED] Total Feature Dimension (Concatenated Channels): {total_dim}"
    )
    logging.info(
        f"[EMBED] Reduced Feature Dimension (Randomly Selected): {len(random_feature_indices)}"
    )
    logging.info(f"[PADIM] Covariance Regularization (epsilon): 0.01")
    logging.info(f"[SYSTEM] Random Seed: {args.seed}")
    logging.info(f"[SYSTEM] Device: {device}")
    logging.info("------------------------------------------------------------------")

    train_dataloader = DataLoader(
        data_manager.train, batch_size=args.batch_size, pin_memory=True, shuffle=True
    )
    test_dataloader = DataLoader(
        data_manager.test, batch_size=args.batch_size, pin_memory=True
    )
    logging.info(f"Results for this class will be saved in: {class_save_dir}")

    # --- 2. Learn Distribution ---
    dist_path = os.path.join(class_save_dir, "learned_distribution.pkl")
    if not os.path.exists(dist_path):
        logging.info("No cached distribution found. Learning from scratch...")

        # --- Pass 1: Calculate Mean ---
        logging.info(f"Starting Pass 1: Calculating mean on {compute_device}...")
        model.eval()
        total_samples, sum_of_features = 0, None
        b, c, h, w = 0, 0, 0, 0

        for image_batch, _, _ in train_dataloader:
            batch_embeddings = get_batch_embeddings(
                image_batch, INTERMEDIATE_FEATURE_MAPS, random_feature_indices, model, device, compute_device
            )

            b, c, h, w = batch_embeddings.shape
            if sum_of_features is None:
                # --- MODIFIED: Sum tensor created on compute_device ---
                sum_of_features = torch.zeros(
                    c, h * w, dtype=torch.float32, device=compute_device
                )

            batch_embeddings = batch_embeddings.view(b, c, h * w)
            sum_of_features += torch.sum(batch_embeddings, dim=0)
            total_samples += b

        mean_vectors = sum_of_features / total_samples
        logging.info(f"Pass 1 Complete. Mean calculated over {total_samples} samples.")

        # --- Pass 2: Calculate Covariance ---
        logging.info(f"Starting Pass 2: Calculating covariance on {compute_device}...")
        sum_of_outer_products = torch.zeros(
            c, c, h * w, dtype=torch.float32, device=compute_device
        )

        for image_batch, _, _ in train_dataloader:
            batch_embeddings = get_batch_embeddings(
                image_batch, INTERMEDIATE_FEATURE_MAPS, random_feature_indices, model, device, compute_device
            )

            b, c, h, w = batch_embeddings.shape
            batch_embeddings = batch_embeddings.view(b, c, h * w)

            centered_batch = batch_embeddings - mean_vectors

            # --- ROBUST & OPTIMIZED COVARIANCE CALCULATION ---
            # Permute from (batch, channels, patches) to (batch, patches, channels)
            centered_batch_permuted = centered_batch.permute(0, 2, 1)

            # Use a robust einsum string that sums the outer products over the batch 'b'.
            # 'bpi,bpj->pij' = for each patch 'p', multiply channel vectors 'i' and 'j'.
            # Result has shape (patches, channels, channels)
            outer_products_sum = torch.einsum(
                "bpi,bpj->pij", centered_batch_permuted, centered_batch_permuted
            )

            # Permute result back to (channels, channels, patches) to match the accumulator
            sum_of_outer_products += outer_products_sum.permute(1, 2, 0)

        cov_matrices = sum_of_outer_products / (total_samples - 1)

        identity = (
            torch.eye(c, device=compute_device).unsqueeze(2).expand(-1, -1, h * w)
        )
        cov_matrices += 0.01 * identity

        learned_distribution = [mean_vectors.cpu().numpy(), cov_matrices.cpu().numpy()]
        logging.info("Pass 2 Complete. Covariance calculated.")

        if args.save_distribution:
            with open(dist_path, "wb") as f:
                pickle.dump(learned_distribution, f)
            logging.info(f"Saved learned distribution to: {dist_path}")
    else:
        logging.info(f"Loading cached distribution from: {dist_path}")
        with open(dist_path, "rb") as f:
            learned_distribution = pickle.load(f)

    # --- 3. Evaluation & Anomaly Scoring ---
    logging.info("Starting evaluation...")

    # Determine the device for Mahalanobis distance calculation
    eval_device = torch.device("cpu")
    if args.mahalanobis_on_gpu:
        if device.type == "cuda":
            eval_device = device
            logging.info("Mahalanobis distance will be computed on the GPU.")
        else:
            logging.info(
                "`--mahalanobis_on_gpu` was set, but no CUDA device found. Using CPU for Mahalanobis."
            )
    else:
        logging.info("Mahalanobis distance will be computed on the CPU (default).")

    # Move learned statistics to the chosen evaluation device
    mean_t = torch.tensor(
        learned_distribution[0], device=eval_device, dtype=torch.float32
    )
    cov_inv_t = torch.linalg.inv(
        torch.tensor(
            learned_distribution[1], device=eval_device, dtype=torch.float32
        ).permute(2, 0, 1)
    )
    mean_t = mean_t.permute(1, 0)

    test_images_list, ground_truth_labels, all_distances = [], [], []
    model.eval()
    for image_batch, labels, _ in test_dataloader:
        test_images_list.extend(image_batch.cpu().numpy())
        ground_truth_labels.extend(labels.cpu().numpy())

        # The helper `get_batch_embeddings` runs the model on the main `device` and
        # moves the final embedding to the specified `target_device`. We now pass `eval_device`.
        with torch.no_grad(), autocast(device_type=device.type):
            batch_embeddings = get_batch_embeddings(
                image_batch, INTERMEDIATE_FEATURE_MAPS, random_feature_indices, model, device, eval_device
            )

        b, c, h, w = batch_embeddings.shape
        batch_embeddings = batch_embeddings.view(b, c, h * w).permute(0, 2, 1)

        # All tensors are now guaranteed to be on `eval_device`, resolving the error.
        diff = batch_embeddings - mean_t
        dist_squared = torch.sum(
            torch.einsum("bpc,pcd->bpd", diff, cov_inv_t) * diff, dim=2
        )
        distances_batch = torch.sqrt(dist_squared)
        all_distances.append(distances_batch.cpu().numpy())

    distances = np.concatenate(all_distances, axis=0)

    anomaly_maps_raw = distances.reshape(len(test_images_list), h, w)

    # --- (The rest of the function remains the same) ---
    raw_min = float(anomaly_maps_raw.min())
    raw_max = float(anomaly_maps_raw.max())
    if raw_max > raw_min:
        anomaly_maps_raw_norm = (anomaly_maps_raw - raw_min) / (raw_max - raw_min)
    else:
        anomaly_maps_raw_norm = np.zeros_like(anomaly_maps_raw)

    score_maps = (
        F.interpolate(
            torch.tensor(anomaly_maps_raw).unsqueeze(1),
            size=args.cropsize,
            mode="bilinear",
            align_corners=False,
        )
        .squeeze()
        .numpy()
    )

    for i in range(score_maps.shape[0]):
        score_maps[i] = gaussian_filter(score_maps[i], sigma=4)

    # --- 4. Calculate Metrics ---
    # ... (This entire section is correct and does not need changes)
    # ... (Code from your provided function from "logging.info("Calculating metrics...")" to the final "return {...}" dictionary)
    logging.info("Calculating metrics and saving results...")
    max_score, min_score = score_maps.max(), score_maps.min()
    normalized_scores = (score_maps - min_score) / (max_score - min_score)
    image_level_scores = normalized_scores.reshape(normalized_scores.shape[0], -1).max(
        axis=1
    )
    ground_truth_labels = np.asarray(ground_truth_labels)

    image_roc_auc = roc_auc_score(ground_truth_labels, image_level_scores)
    try:
        image_pr_auc = average_precision_score(ground_truth_labels, image_level_scores)
    except Exception:
        image_pr_auc = None

    fpr, tpr, thresholds = roc_curve(ground_truth_labels, image_level_scores)
    logging.info(f"Image-level ROC AUC for class '{class_name}': {image_roc_auc:.4f}")

    gmeans = np.sqrt(tpr * (1 - fpr))
    best_idx = np.argmax(gmeans)
    thresholds_from_roc = roc_curve(ground_truth_labels, image_level_scores)[2]
    optimal_threshold = thresholds_from_roc[best_idx]

    with open(os.path.join(class_save_dir, "results.txt"), "w") as f:
        f.write(f"Image-level ROC AUC: {image_roc_auc:.4f}\n")
        if image_pr_auc is not None:
            f.write(f"Image-level PR AUC: {image_pr_auc:.4f}\n")

    # --- 5. Generate and Save All Visualizations ---
    plot_summary_visuals(
        class_save_dir,
        class_name,
        image_roc_auc,
        fpr,
        tpr,
        ground_truth_labels,
        image_level_scores,
        pr_auc=image_pr_auc,
    )
    plot_mean_anomaly_maps(
        class_save_dir, ground_truth_labels, anomaly_maps_raw_norm, score_maps
    )

    normal_indices = np.where(ground_truth_labels == 0)[0]
    if len(normal_indices) > 0:
        threshold_norm = np.percentile(normalized_scores[normal_indices].ravel(), 99)
    else:
        threshold_norm = np.percentile(normalized_scores.ravel(), 99)

    per_image_stats = []
    # ... (rest of per_image_stats calculation)
    n_pixels = normalized_scores.shape[1] * normalized_scores.shape[2]
    top1pct_n = max(1, int(np.ceil(0.01 * n_pixels)))
    img_h, img_w = normalized_scores.shape[1], normalized_scores.shape[2]
    center_coord = np.array([img_h / 2.0, img_w / 2.0])
    for i in range(normalized_scores.shape[0]):
        heat = normalized_scores[i]
        flat = heat.ravel()
        # ... (rest of the loop)
        sorted_flat = np.sort(flat)
        maxv = float(np.max(flat))
        mean_top1pct = float(np.mean(sorted_flat[-top1pct_n:]))
        p95 = float(np.percentile(flat, 95))
        meanv = float(np.mean(flat))
        medianv = float(np.median(flat))
        stdv = float(np.std(flat))
        skewv = float(sps.skew(flat))
        kurtv = float(sps.kurtosis(flat))
        frac_above = float(np.mean(flat >= threshold_norm))
        mask = heat >= threshold_norm
        num_components = 0
        largest_cc_area = 0.0
        mean_cc_area = 0.0
        bbox_area = 0
        centroid = (np.nan, np.nan)
        centroid_dist = np.nan
        if np.any(mask):
            labeled, num = ndimage.label(mask)
            num_components = int(num)
            areas = ndimage.sum(mask.astype(float), labeled, range(1, num + 1))
            if len(areas) > 0:
                areas = np.array(areas)
                largest_cc_area = float(np.max(areas))
                mean_cc_area = float(np.mean(areas))
            ys, xs = np.where(mask)
            ymin, ymax = int(np.min(ys)), int(np.max(ys))
            xmin, xmax = int(np.min(xs)), int(np.max(xs))
            bbox_area = int((ymax - ymin + 1) * (xmax - xmin + 1))
            try:
                cy, cx = ndimage.center_of_mass(heat)
                centroid = (float(cx), float(cy))
                centroid_dist = float(np.linalg.norm(np.array([cy, cx]) - center_coord))
            except Exception:
                centroid = (np.nan, np.nan)
                centroid_dist = np.nan
        total_pixels = float(img_h * img_w)
        per_image_stats.append(
            {
                "max": maxv,
                "mean_top1pct": mean_top1pct,
                "p95": p95,
                "mean": meanv,
                "median": medianv,
                "std": stdv,
                "skewness": skewv,
                "kurtosis": kurtv,
                "frac_above": frac_above,
                "threshold": threshold_norm,
                "num_components": num_components,
                "largest_cc_area": largest_cc_area,
                "largest_cc_frac": (
                    largest_cc_area / total_pixels if total_pixels > 0 else 0.0
                ),
                "mean_cc_area": mean_cc_area,
                "bbox_area": bbox_area,
                "centroid_x": centroid[0],
                "centroid_y": centroid[1],
                "centroid_dist": centroid_dist,
            }
        )

    per_image_csv_path = os.path.join(class_save_dir, "per_image_metrics.csv")
    import csv

    with open(per_image_csv_path, "w", newline="") as csvfile:
        # ... (rest of CSV writing)
        fieldnames = [
            "filepath",
            "gt_label",
            "pred_label",
            "score",
            "max",
            "mean_top1pct",
            "p95",
            "mean",
            "median",
            "std",
            "skewness",
            "kurtosis",
            "frac_above",
            "threshold",
            "num_components",
            "largest_cc_area",
            "largest_cc_frac",
            "mean_cc_area",
            "bbox_area",
            "centroid_x",
            "centroid_y",
            "centroid_dist",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        predictions = (image_level_scores >= optimal_threshold).astype(int)
        for i, stats in enumerate(per_image_stats):
            filepath = data_manager.test.image_filepaths[i]
            gt = int(ground_truth_labels[i])
            pred = int(predictions[i])
            row = {
                "filepath": filepath,
                "gt_label": gt,
                "pred_label": pred,
                "score": float(image_level_scores[i]),
            }
            row.update(stats)
            writer.writerow(row)

    plot_individual_visualizations(
        test_images=test_images_list,
        raw_maps=anomaly_maps_raw,
        norm_scores=normalized_scores,
        img_scores=image_level_scores,
        save_dir=os.path.join(class_save_dir, "visualizations"),
        test_filepaths=data_manager.test.image_filepaths,
        per_image_stats=per_image_stats,
        optimal_threshold=optimal_threshold,
    )
    plot_patch_score_distributions(
        class_save_dir, ground_truth_labels, anomaly_maps_raw
    )
    logging.info(f"All individual results for class '{class_name}' saved.")

    # --- 6. Calculate Final Metrics & Prepare Data for Master CSV ---
    # ... (rest of the function)
    tn, fp, fn, tp = confusion_matrix(ground_truth_labels, predictions).ravel()
    epsilon = 1e-6
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    logging.info(
        f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1_score:.4f}"
    )
    normal_scores = image_level_scores[ground_truth_labels == 0]
    anomalous_scores = image_level_scores[ground_truth_labels == 1]

    cohen_d, wass, mw_p, auc_ci_low, auc_ci_high = (
        None,
        None,
        None,
        None,
        None,
    )  # Initialize

    # ... (rest of the calculations for cohen_d, wass, etc.)
    def cohens_d(a, b):
        nx, ny = len(a), len(b)
        dof = nx + ny - 2
        pooled_std = np.sqrt(
            ((nx - 1) * np.var(a, ddof=1) + (ny - 1) * np.var(b, ddof=1)) / dof
        )
        return (np.mean(a) - np.mean(b)) / (pooled_std + 1e-12)

    try:
        if len(normal_scores) > 1 and len(anomalous_scores) > 1:
            cohen_d = float(cohens_d(anomalous_scores, normal_scores))
    except Exception:
        pass
    try:
        wass = float(sps.wasserstein_distance(normal_scores, anomalous_scores))
    except Exception:
        pass
    try:
        _, mw_p = sps.mannwhitneyu(
            normal_scores, anomalous_scores, alternative="two-sided"
        )
    except Exception:
        pass

    def bootstrap_auc(y_true, y_scores, n_bootstrap=1000, seed=0):
        rng = np.random.RandomState(seed)
        bootstrapped_scores = []
        for _ in range(n_bootstrap):
            indices = rng.randint(0, len(y_scores), len(y_scores))
            if len(np.unique(y_true[indices])) < 2:
                continue
            score = roc_auc_score(y_true[indices], y_scores[indices])
            bootstrapped_scores.append(score)
        if not bootstrapped_scores:
            return None, None
        return np.percentile(bootstrapped_scores, 2.5), np.percentile(
            bootstrapped_scores, 97.5
        )

    try:
        auc_ci_low, auc_ci_high = bootstrap_auc(
            ground_truth_labels, image_level_scores, n_bootstrap=200
        )
    except Exception:
        pass

    with open(os.path.join(class_save_dir, "results.txt"), "a") as f:
        # ... (rest of writing to results.txt)
        f.write(f"Optimal Threshold: {optimal_threshold:.4f}\n")
        f.write(f"True Negatives: {tn}\n")
        f.write(f"False Positives: {fp}\n")
        f.write(f"False Negatives: {fn}\n")
        f.write(f"True Positives: {tp}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1_score:.4f}\n")
        if cohen_d is not None:
            f.write(f"Cohen_d: {cohen_d:.4f}\n")
        if wass is not None:
            f.write(f"Wasserstein: {wass:.4f}\n")
        if mw_p is not None:
            f.write(f"Mann-Whitney p: {mw_p:.4e}\n")
        if auc_ci_low is not None:
            f.write(f"ROC AUC CI: [{auc_ci_low:.4f}, {auc_ci_high:.4f}]\n")

    try:
        class_report = {
            # ... (class_report creation)
            "class_name": class_name,
            "roc_auc": float(image_roc_auc),
            "pr_auc": float(image_pr_auc) if image_pr_auc is not None else None,
            "roc_auc_ci": (
                [auc_ci_low, auc_ci_high]
                if (auc_ci_low is not None and auc_ci_high is not None)
                else None
            ),
            "cohen_d": cohen_d,
            "wasserstein": wass,
            "mannwhitney_p": mw_p,
            "num_images": int(len(image_level_scores)),
        }

        def agg_by_label(values, labels):
            vals = np.array(values)
            return {
                "mean_normal": (
                    float(np.mean(vals[labels == 0])) if np.any(labels == 0) else None
                ),
                "mean_anomalous": (
                    float(np.mean(vals[labels == 1])) if np.any(labels == 1) else None
                ),
                "median_normal": (
                    float(np.median(vals[labels == 0])) if np.any(labels == 0) else None
                ),
                "median_anomalous": (
                    float(np.median(vals[labels == 1])) if np.any(labels == 1) else None
                ),
            }

        class_report["image_level_score_stats"] = agg_by_label(
            image_level_scores, ground_truth_labels
        )
        class_report_path = os.path.join(class_save_dir, "class_report.json")
        import json

        with open(class_report_path, "w") as jf:
            json.dump(class_report, jf, indent=2)
    except Exception:
        logging.exception("Failed to write class_report.json")

    return {
        "class_name": class_name,
        "roc_auc": round(image_roc_auc, 4),
        "optimal_threshold": round(optimal_threshold, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "cohen_d": round(cohen_d, 4) if cohen_d is not None else None,
        "wasserstein": round(wass, 4) if wass is not None else None,
        "mannwhitney_p": round(mw_p, 6) if mw_p is not None else None,
        "roc_auc_ci_low": round(auc_ci_low, 4) if auc_ci_low is not None else None,
        "roc_auc_ci_high": round(auc_ci_high, 4) if auc_ci_high is not None else None,
    }


# ==========================================================================================
# SCRIPT ENTRYPOINT
# ==========================================================================================


def main():
    parser = argparse.ArgumentParser(
        description="PaDiM Anomaly Detection Experiment Runner"
    )
    parser.add_argument(
        "--model_architecture",
        type=str,
        default="wide_resnet50_2",
        choices=["wide_resnet50_2", "resnet18", "efficientnet_b5"],
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../anomaly_detection/data/BowTie-New/original",
        help="Root path to the dataset.",
    )
    parser.add_argument(
        "--base_results_dir",
        type=str,
        default="./results",
        help="Directory to save all experiment results.",
    )
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--cropsize", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument(
        "--save_distribution",
        action="store_true",
        help="Save the learned distribution model to a .pkl file.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Set the batch size for training and testing.",
    )
    # Data augmentation options (forwarded to dataset manager)
    parser.add_argument(
        "--augmentations_enabled",
        action="store_true",
        help="Enable training-time data augmentations (applied only to train set).",
    )
    parser.add_argument(
        "--horizontal_flip",
        action="store_true",
        help="Allow horizontal flip augmentation when augmentations are enabled.",
    )
    parser.add_argument(
        "--vertical_flip",
        action="store_true",
        help="Allow vertical flip augmentation when augmentations are enabled.",
    )
    parser.add_argument(
        "--augmentation_prob",
        type=float,
        default=0.5,
        help="Probability that the augmentation block is applied to a sample (when using probabilistic mode).",
    )
    parser.add_argument(
        "--results_subdir",
        type=str,
        default="",
        help="Optional subfolder inside the base results directory to save results (e.g. 'custom_folder'). If empty, saves directly in base_results_dir.)",
    )
    parser.add_argument(
        "--mahalanobis_on_gpu",
        action="store_true",
        help="If set, compute Mahalanobis distances on the GPU. If omitted, compute on CPU (slower but avoids GPU memory issues).",
    )
    parser.add_argument(
        "--stats_on_gpu",
        action="store_true",
        help="If set, run mean/covariance calculations on the GPU. Default is CPU for memory safety.",
    )
    args = parser.parse_args()

    # --- 1. Initial Setup ---
    # (Implementation is the same as before, code omitted for brevity)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_name = (
        f"{args.model_architecture}_resize-{args.resize}_crop-{args.cropsize}"
    )
    # If a results_subdir is provided, place the experiment folder under base_results_dir/results_subdir/<experiment_name>
    if args.results_subdir:
        args.master_save_dir = os.path.join(args.base_results_dir, args.results_subdir)
    else:
        args.master_save_dir = os.path.join(args.base_results_dir, experiment_name)
    os.makedirs(args.master_save_dir, exist_ok=True)

    setup_logging(args.master_save_dir, "experiment_log")
    logging.info(f"--- Starting Experiment: {experiment_name} ---")

    # --- 2. Model & Feature Selection ---
    if args.model_architecture == "wide_resnet50_2":
        model = wide_resnet50_2(weights="DEFAULT")
        total_dim, reduced_dim = 1792, 550
    elif args.model_architecture == "resnet18":
        model = resnet18(weights="DEFAULT")
        total_dim, reduced_dim = 448, 100
    elif args.model_architecture == "efficientnet_b5":
        model = efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
        # For EfficientNet-B5, the output channels of blocks 2, 4, and 6 are:
        # Block 2: 40 channels
        # Block 4: 112 channels
        # Block 6: 320 channels
        # Total = 40 + 112 + 320 = 472
        total_dim, reduced_dim = 472, 200

    model.to(device)
    model.eval()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    random_feature_indices = torch.tensor(
        random.sample(range(0, total_dim), reduced_dim)
    )

    if "resnet" in args.model_architecture:
        model.layer1[-1].register_forward_hook(hook_function)
        model.layer2[-1].register_forward_hook(hook_function)
        model.layer3[-1].register_forward_hook(hook_function)
    elif "efficientnet" in args.model_architecture:
        model.features[2].register_forward_hook(hook_function)
        model.features[4].register_forward_hook(hook_function)
        model.features[6].register_forward_hook(hook_function)

    # --- 3. Loop Through All Classes ---
    all_class_results = []
    discovered_classes = bowtie.get_class_names(args.data_path)

    # If the provided data_path itself contains train/ and test/, treat it as a
    # single class folder and run for that class name (the basename of data_path).
    is_direct_class_folder = os.path.isdir(
        os.path.join(args.data_path, "train")
    ) and os.path.isdir(os.path.join(args.data_path, "test"))

    if is_direct_class_folder:
        class_name = os.path.basename(os.path.abspath(args.data_path))
        logging.info(
            f"Data path appears to be a single class folder. Running for class: {class_name}"
        )
        args_copy = copy.copy(args)
        # BowtieDataManager expects dataset_path to be the parent folder that
        # contains class subfolders; when data_path already points to class
        # folder, set data_path to its parent and pass the class folder name.
        args_copy.data_path = os.path.dirname(os.path.abspath(args.data_path))
        try:
            class_summary = run_class_processing(
                args_copy, class_name, model, device, random_feature_indices
            )
            all_class_results.append(class_summary)
        except Exception as e:
            logging.exception(f"Error processing class {class_name}: {e}")
    else:
        if not discovered_classes:
            logging.error(
                f"No class subfolders found in data path: {args.data_path}. "
                "Make sure the dataset root contains one folder per class (each with train/ and test/)."
            )
            return

        for class_name in discovered_classes:
            try:
                class_summary = run_class_processing(
                    args, class_name, model, device, random_feature_indices
                )
                all_class_results.append(class_summary)
            except Exception as e:
                logging.error(
                    f"!!! FAILED to process class {class_name}. Error: {e}",
                    exc_info=True,
                )

    # --- 4. Save Master Results ---
    # (Implementation is the same as before, code omitted for brevity)
    if not all_class_results:
        logging.warning(
            "No classes were processed successfully. Master results CSV will not be generated."
        )
        return

    master_df = pd.DataFrame(all_class_results)
    csv_path = os.path.join(args.master_save_dir, "master_results.csv")
    master_df.to_csv(csv_path, index=False)

    logging.info(f"--- Experiment Complete ---")
    logging.info(f"Master results saved to: {csv_path}")
    logging.info("\n" + master_df.to_string())


if __name__ == "__main__":
    main()
