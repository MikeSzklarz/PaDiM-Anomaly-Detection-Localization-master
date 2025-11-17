import os
import pickle
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from scipy.ndimage import gaussian_filter
from scipy import ndimage
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import average_precision_score
import scipy.stats as sps

import datasets.bowtie as bowtie
import datasets.mvtec as mvtec
from utils.embeddings import get_batch_embeddings
from utils.plotting import (
    plot_summary_visuals,
    plot_mean_anomaly_maps,
    plot_individual_visualizations,
    plot_patch_score_distributions,
)

import pandas as pd
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from utils.misc import denormalize_image_for_display

import cv2

def run_class_processing(
    args, class_name, model, device, random_feature_indices, hooks
):
    """Main function to process a single class. Extracted from `run_experiment.py`.

    This function expects a mutable `hooks` list. The experiment script should
    register a forward hook that appends outputs into the same list and then pass
    that list here so embeddings can be built from the collected feature maps.
    """

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
    train_class_name = class_name

    if getattr(args, "test_class_path", None):
        test_abs = os.path.abspath(args.test_class_path)
        if not (
            os.path.isdir(os.path.join(test_abs, "train"))
            and os.path.isdir(os.path.join(test_abs, "test"))
        ):
            logging.error(
                f"Provided --test_class_path does not look like a class folder (missing train/ or test/): {test_abs}"
            )
            raise ValueError(
                f"Invalid test_class_path: {test_abs}. Must contain train/ and test/ subfolders."
            )
        test_class_name = os.path.basename(test_abs)
        test_dataset_path = os.path.dirname(test_abs)
    elif getattr(args, "test_class_name", None) is not None:
        test_class_name = args.test_class_name
        test_dataset_path = args.data_path
    else:
        test_class_name = train_class_name
        test_dataset_path = args.data_path

    logging.info(
        f"--- Starting processing for TRAIN: {train_class_name} | TEST: {test_class_name} ---"
    )

    if (
        getattr(args, "test_class_path", None)
        or getattr(args, "test_class_name", None) is not None
    ):
        class_save_dir = os.path.join(
            args.master_save_dir, f"train_{train_class_name}_test_{test_class_name}"
        )
    else:
        class_save_dir = os.path.join(args.master_save_dir, train_class_name)
    os.makedirs(class_save_dir, exist_ok=True)

    if args.dataset == "mvtec":
        train_data_loader = mvtec.MVTecDataLoader(
            dataset_path=args.data_path,
            class_name=train_class_name,
            resize=args.resize,
            cropsize=args.cropsize,
        )
        test_data_loader = mvtec.MVTecDataLoader(
            dataset_path=test_dataset_path,
            class_name=test_class_name,
            resize=args.resize,
            cropsize=args.cropsize,
        )

    else:  # default: Bowtie
        train_data_loader = bowtie.BowtieDataLoader(
            dataset_path=args.data_path,
            class_name=train_class_name,
            resize=args.resize,
            cropsize=args.cropsize,
            seed=args.seed,
            augmentations_enabled=args.augmentations_enabled,
            horizontal_flip=args.horizontal_flip,
            vertical_flip=args.vertical_flip,
            augmentation_prob=args.augmentation_prob,
        )

        test_data_loader = bowtie.BowtieDataLoader(
            dataset_path=test_dataset_path,
            class_name=test_class_name,
            resize=args.resize,
            cropsize=args.cropsize,
            seed=args.seed,
            augmentations_enabled=False,
            horizontal_flip=False,
            vertical_flip=False,
            augmentation_prob=0.0,
            normal_test_sample_ratio=args.test_sample_ratio,
        )
        
    logging.info("------------------- EXPERIMENT CONFIGURATION -------------------")
    logging.info(f"[DATASET] Class Name: {class_name}")
    logging.info(f"[DATASET] Data Path: {args.data_path}")
    logging.info(f"[DATASET] Train Set Size: {len(train_data_loader.train)} images")
    logging.info(f"[DATASET] Test Set Size: {len(test_data_loader.test)} images")
    normal_test_count = sum(1 for label in test_data_loader.test.labels if label == 0)
    abnormal_test_count = len(test_data_loader.test.labels) - normal_test_count
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
    map_shapes = [f.shape for f in hooks]
    final_h, final_w = map_shapes[-1][-2], map_shapes[-1][-1]
    total_patches = final_h * final_w
    logging.info(f"[MODEL] Intermediate Feature Map Shapes (B,C,H,W): {map_shapes}")
    logging.info(f"[MODEL] Final Anomaly Map Grid Size (H x W): {final_h} x {final_w}")
    logging.info(f"[MODEL] Total Patches per Image: {total_patches}")
    hooks.clear()
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
        train_data_loader.train,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_data_loader.test, batch_size=args.batch_size, pin_memory=True
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
                image_batch,
                hooks,
                random_feature_indices,
                model,
                device,
                compute_device,
            )

            b, c, h, w = batch_embeddings.shape
            if sum_of_features is None:
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
                image_batch,
                hooks,
                random_feature_indices,
                model,
                device,
                compute_device,
            )

            b, c, h, w = batch_embeddings.shape
            batch_embeddings = batch_embeddings.view(b, c, h * w)

            centered_batch = batch_embeddings - mean_vectors

            centered_batch_permuted = centered_batch.permute(0, 2, 1)

            outer_products_sum = torch.einsum(
                "bpi,bpj->pij", centered_batch_permuted, centered_batch_permuted
            )

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

    mean_t = torch.tensor(
        learned_distribution[0], device=eval_device, dtype=torch.float32
    )
    cov_inv_t = torch.linalg.inv(
        torch.tensor(
            learned_distribution[1], device=eval_device, dtype=torch.float32
        ).permute(2, 0, 1)
    )
    mean_t = mean_t.permute(1, 0)

    test_images_list, ground_truth_labels, ground_truth_masks_list, all_distances = [], [], [], []
    model.eval()
    for image_batch, labels, masks in test_dataloader:
        test_images_list.extend(image_batch.cpu().numpy())
        ground_truth_labels.extend(labels.cpu().numpy())
        ground_truth_masks_list.append(masks.cpu().numpy())

        with torch.no_grad(), autocast(device_type=device.type):
            batch_embeddings = get_batch_embeddings(
                image_batch,
                hooks,
                random_feature_indices,
                model,
                device,
                eval_device,
            )

        b, c, h, w = batch_embeddings.shape
        batch_embeddings = batch_embeddings.view(b, c, h * w).permute(0, 2, 1)

        diff = batch_embeddings - mean_t
        dist_squared = torch.sum(
            torch.einsum("bpc,pcd->bpd", diff, cov_inv_t) * diff, dim=2
        )
        distances_batch = torch.sqrt(dist_squared)
        all_distances.append(distances_batch.cpu().numpy())

    distances = np.concatenate(all_distances, axis=0)

    ground_truth_masks = np.array([])
    if ground_truth_masks_list:
        ground_truth_masks = np.concatenate(ground_truth_masks_list, axis=0)
        # Ensure masks are [B, H, W] and binary (0 or 1)
        ground_truth_masks = (ground_truth_masks.squeeze() > 0.5).astype(np.uint8)

    anomaly_maps_raw = distances.reshape(len(test_images_list), h, w)

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

    logging.info("Calculating metrics and saving results...")
    max_score, min_score = score_maps.max(), score_maps.min()
    normalized_scores = (score_maps - min_score) / (max_score - min_score)
    image_level_scores = normalized_scores.reshape(normalized_scores.shape[0], -1).max(
        axis=1
    )
    ground_truth_labels = np.asarray(ground_truth_labels)

    if args.dataset == "mvtec" and ground_truth_masks.any():
            logging.info("Running segmentation thresholding sweep for MVTec...")
            try:
                run_segmentation_sweep(
                    normalized_scores=normalized_scores,
                    ground_truth_masks=ground_truth_masks,
                    ground_truth_labels=ground_truth_labels,
                    test_filepaths=test_data_loader.test.image_filepaths,
                    class_save_dir=class_save_dir,
                    test_images_list=test_images_list
                )
                logging.info("Segmentation sweep complete. Results saved.")
            except Exception as e:
                logging.error(f"Failed to run segmentation sweep: {e}", exc_info=True)
    elif args.dataset == "mvtec":
        logging.warning("MVTec dataset selected, but no ground truth masks were loaded. Skipping sweep.")

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

    # Writing to results.txt has been removed per request; log the key metrics instead.
    logging.info(f"Image-level ROC AUC: {image_roc_auc:.4f}")
    if image_pr_auc is not None:
        logging.info(f"Image-level PR AUC: {image_pr_auc:.4f}")

    # Compute image-level predictions from the optimal threshold (keep this
    # variable available for later metrics/plots even though we no longer write per-image CSVs).
    predictions = (image_level_scores >= optimal_threshold).astype(int)

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
        class_save_dir,
        ground_truth_labels,
        anomaly_maps_raw_norm,
        score_maps,
        img_scores=image_level_scores,
        optimal_threshold=optimal_threshold,
    )

    normal_indices = np.where(ground_truth_labels == 0)[0]
    if len(normal_indices) > 0:
        threshold_norm = np.percentile(normalized_scores[normal_indices].ravel(), 99)
    else:
        threshold_norm = np.percentile(normalized_scores.ravel(), 99)

    per_image_stats = []
    n_pixels = normalized_scores.shape[1] * normalized_scores.shape[2]
    top1pct_n = max(1, int(np.ceil(0.01 * n_pixels)))
    img_h, img_w = normalized_scores.shape[1], normalized_scores.shape[2]
    center_coord = np.array([img_h / 2.0, img_w / 2.0])
    for i in range(normalized_scores.shape[0]):
        heat = normalized_scores[i]
        flat = heat.ravel()
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

    plot_individual_visualizations(
        test_images=test_images_list,
        raw_maps=anomaly_maps_raw,
        norm_scores=normalized_scores,
        img_scores=image_level_scores,
        save_dir=os.path.join(class_save_dir, "visualizations"),
        test_filepaths=test_data_loader.test.image_filepaths,
        per_image_stats=per_image_stats,
        optimal_threshold=optimal_threshold,
    )
    plot_patch_score_distributions(
        class_save_dir, ground_truth_labels, anomaly_maps_raw
    )
    logging.info(f"All individual results for class '{class_name}' saved.")

    tn, fp, fn, tp = confusion_matrix(ground_truth_labels, predictions).ravel()
    epsilon = 1e-6
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    
    normal_scores = image_level_scores[ground_truth_labels == 0]
    anomalous_scores = image_level_scores[ground_truth_labels == 1]

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

    # Writing to results.txt has been removed per request; log the summary instead.
    logging.info(f"Optimal Threshold: {optimal_threshold:.4f}")
    logging.info(
        f"True Negatives: {tn} | False Positives: {fp} | False Negatives: {fn} | True Positives: {tp}"
    )
    logging.info(
        f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1_score:.4f}"
    )
    if auc_ci_low is not None:
        logging.info(f"ROC AUC CI: [{auc_ci_low:.4f}, {auc_ci_high:.4f}]")

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
        "roc_auc_ci_low": round(auc_ci_low, 4) if auc_ci_low is not None else None,
        "roc_auc_ci_high": round(auc_ci_high, 4) if auc_ci_high is not None else None,
    }












def calculate_iou_dice(pred_mask, gt_mask):
    """Calculates IoU and Dice score for binary masks."""
    # Handle "good" images (no ground truth anomaly)
    if np.sum(gt_mask) == 0:
        if np.sum(pred_mask) == 0:
            return 1.0, 1.0  # Correctly predicted no anomaly
        else:
            # Falsely predicted anomaly. This is a perfect "false positive" segmentation.
            # IoU is 0, Dice is 0.
            return 0.0, 0.0
    
    # Handle "bad" images (with ground truth anomaly)
    intersection = np.sum((pred_mask == 1) & (gt_mask == 1))
    union = np.sum((pred_mask == 1) | (gt_mask == 1))
    
    iou = intersection / (union + 1e-6)
    
    dice_denom = np.sum(pred_mask) + np.sum(gt_mask)
    dice = (2 * intersection) / (dice_denom + 1e-6)
    
    return float(iou), float(dice)

def get_f1_optimal_threshold(all_scores_flat, all_masks_flat):
    """Finds the threshold that maximizes F1 score on pixel-level PR curve."""
    try:
        precision, recall, thresholds = precision_recall_curve(all_masks_flat, all_scores_flat)
        
        # Calculate F1 score for each threshold
        # We need to add a small epsilon to avoid division by zero
        f1 = (2 * precision * recall) / (precision + recall + 1e-6)
        
        # The 'thresholds' array is one element shorter than 'f1'
        # (f1[0] corresponds to no threshold, f1[-1] to threshold=1)
        # We should ignore the last f1 value (default P=1, R=0)
        
        best_f1_idx = np.argmax(f1[:-1])
        best_threshold = thresholds[best_f1_idx]
        
        return float(best_threshold)
    except Exception as e:
        logging.warning(f"Could not calculate F1-optimal threshold: {e}. Defaulting to 0.5.")
        return 0.5

def run_segmentation_sweep(normalized_scores, ground_truth_masks, ground_truth_labels, test_filepaths, class_save_dir, test_images_list):
    """
    Runs a parameter sweep for segmentation thresholding strategies and saves results.
    """
    results = []
    num_images = len(normalized_scores)
    image_filenames = [os.path.basename(f) for f in test_filepaths]
    
    logging.info(f"Sweep: Found {num_images} score maps and {len(ground_truth_masks)} masks.")

    # --- Strategy A: Class-wide F1-Optimal ---
    logging.info("Sweep Strategy: A - Class-wide F1 Optimal")
    all_scores_flat = normalized_scores.ravel()
    all_masks_flat = ground_truth_masks.ravel()
    f1_thresh = get_f1_optimal_threshold(all_scores_flat, all_masks_flat)
    
    for i in range(num_images):
        pred_mask = (normalized_scores[i] >= f1_thresh).astype(np.uint8)
        iou, dice = calculate_iou_dice(pred_mask, ground_truth_masks[i])
        results.append({
            "image": image_filenames[i],
            "gt_label": ground_truth_labels[i],
            "strategy": "ClassF1",
            "param": "N/A",
            "threshold": f1_thresh,
            "iou": iou,
            "dice": dice
        })

    # --- Strategy B: Image-level Percentile ---
    logging.info("Sweep Strategy: B - Image-level Percentile")
    for p in [95, 98, 99, 99.5, 99.9]:
        for i in range(num_images):
            thresh = np.percentile(normalized_scores[i].ravel(), p)
            pred_mask = (normalized_scores[i] >= thresh).astype(np.uint8)
            iou, dice = calculate_iou_dice(pred_mask, ground_truth_masks[i])
            results.append({
                "image": image_filenames[i],
                "gt_label": ground_truth_labels[i],
                "strategy": "ImagePercentile",
                "param": p,
                "threshold": thresh,
                "iou": iou,
                "dice": dice
            })

    # --- Strategy C: Class-wide "Normal" Percentile ---
    logging.info("Sweep Strategy: C - Class-wide Normal Percentile")
    normal_indices = np.where(ground_truth_labels == 0)[0]
    if len(normal_indices) > 0:
        normal_scores_flat = normalized_scores[normal_indices].ravel()
        for p in np.arange(90.0, 99.9, 0.1):
            thresh = np.percentile(normal_scores_flat, p)
            for i in range(num_images):
                pred_mask = (normalized_scores[i] >= thresh).astype(np.uint8)
                iou, dice = calculate_iou_dice(pred_mask, ground_truth_masks[i])
                results.append({
                    "image": image_filenames[i],
                    "gt_label": ground_truth_labels[i],
                    "strategy": "NormalPercentile",
                    "param": p,
                    "threshold": thresh,
                    "iou": iou,
                    "dice": dice
                })
    else:
        logging.warning("Sweep Strategy C: No 'normal' images found in test set. Skipping.")

    # --- Strategy D: Fixed Value Threshold ---
    logging.info("Sweep Strategy: D - Fixed Value")
    for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for i in range(num_images):
            pred_mask = (normalized_scores[i] >= t).astype(np.uint8)
            iou, dice = calculate_iou_dice(pred_mask, ground_truth_masks[i])
            results.append({
                "image": image_filenames[i],
                "gt_label": ground_truth_labels[i],
                "strategy": "FixedValue",
                "param": t,
                "threshold": t,
                "iou": iou,
                "dice": dice
            })

    # --- Save results ---
    df = pd.DataFrame(results)
    save_path = os.path.join(class_save_dir, "segmentation_sweep_results.csv")
    df.to_csv(save_path, index=False)
    logging.info(f"Sweep results saved to {save_path}")

    # --- Print summary ---
    # We only care about the IOU/Dice for anomalous images
    anomalous_df = df[df['gt_label'] == 1].copy()
    if not anomalous_df.empty:
        summary = anomalous_df.groupby(['strategy', 'param'])[['iou', 'dice']].mean()
        logging.info("--- Segmentation Sweep Summary (Mean Scores on ANOMALOUS Images) ---")
        logging.info("\n" + summary.to_string())
        
        logging.info("Generating comparison segmentation visualizations...")
        visuals_save_dir = os.path.join(class_save_dir, "segmentation_sweep_visuals")
        os.makedirs(visuals_save_dir, exist_ok=True)

        performers_list = []

        # 1. Get the "ClassF1 (Original)" performer
        if 'ClassF1' in summary.index:
            f1_iou = float(summary.loc['ClassF1']['iou'].iloc[0])
            # f1_thresh is already calculated from earlier in the function
            performers_list.append({
                "name": "ClassF1 (Original)",
                "param": "ClassF1",
                "threshold": f1_thresh,
                "mean_iou": f1_iou
            })
        else:
            logging.warning("Could not find 'ClassF1' results. Using default for plot.")
            performers_list.append({"name": "ClassF1 (Original)", "param": "N/A", "threshold": 0.5, "mean_iou": 0.0})

        # 2. Get the "Top 2 NormalPercentile" performers
        if not anomalous_df.empty and 'NormalPercentile' in summary.index:
            top_2_series = summary.loc['NormalPercentile']['iou'].nlargest(2)
            plot_names = ["NP Top 1", "NP Top 2"]
            
            for i, (param, iou_score) in enumerate(top_2_series.items()):
                thresh = df[
                    (df['strategy'] == 'NormalPercentile') & (df['param'] == param)
                ]['threshold'].values[0]
                
                performers_list.append({
                    "name": plot_names[i],
                    "param": param,
                    "threshold": thresh,
                    "mean_iou": iou_score
                })
        else:
            logging.warning("Could not find 'NormalPercentile' results. Using defaults for plot.")
            performers_list.append({"name": "NP Top 1", "param": 99.9, "threshold": 0.5, "mean_iou": 0.0})
            performers_list.append({"name": "NP Top 2", "param": 99.0, "threshold": 0.4, "mean_iou": 0.0})

        # Ensure we always have 3 performers for plotting, even if some failed
        while len(performers_list) < 3:
            performers_list.append({"name": "Error", "param": 0, "threshold": 0.5, "mean_iou": 0.0})

        logging.info(f"Visualization performers: {[(p['name'], p['param'], round(p['mean_iou'], 4)) for p in performers_list]}")

        for i in range(num_images):
            # Create subfolders for 'good' and 'anomalous'
            subfolder = "anomalous" if ground_truth_labels[i] == 1 else "good"
            final_save_dir = os.path.join(visuals_save_dir, subfolder)
            os.makedirs(final_save_dir, exist_ok=True)
            
            plot_segmentation_sweep_visuals(
                img_original=test_images_list[i],
                gt_mask=ground_truth_masks[i],
                heatmap=normalized_scores[i],
                save_dir=final_save_dir,
                filename=image_filenames[i],
                top_3_performers=performers_list # Pass our new list
            )
        
        logging.info(f"Saved {num_images} visualizations to {visuals_save_dir}")
    else:
        logging.info("--- Segmentation Sweep Summary (No anomalous images found) ---")
        
def plot_segmentation_sweep_visuals(
    img_original,
    gt_mask,
    heatmap,
    save_dir,
    filename,
    top_3_performers
):
    """
    Saves a 1x4 visualization for the segmentation sweep.
    [Ground Truth] | [Top 1 Perfomer] | [Top 2 Performer] | [Top 3 Performer]
    
    Colors:
    - Black: True Negative
    - White: False Negative (GT only)
    - Red:   False Positive (Pred only)
    - Green: True Positive (Overlap)
    """
    try:
        img_denormalized = denormalize_image_for_display(img_original)
        h, w = img_denormalized.shape[:2]
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # --- Panel 1: Ground Truth Mask (as-is) ---
        axes[0].imshow(gt_mask, cmap="gray")
        axes[0].set_title("Ground Truth Mask")
        axes[0].axis("off")
        
        # Colors for 3-color mask
        # We use BGR for openCV/numpy, but matplotlib expects RGB, so we'll define as RGB
        color_white_fn = [255, 255, 255] # False Negative (GT only)
        color_red_fp = [255, 0, 0]     # False Positive (Pred only)
        color_green_tp = [0, 255, 0]    # True Positive (Overlap)

        plot_titles = ["Top 1", "Top 2", "Top 3"]

        for i, performer in enumerate(top_3_performers):
            ax = axes[i+1]
            thresh = performer['threshold']
            param = performer['param']
            mean_iou = performer['mean_iou']
            plot_title = performer['name']
            
            # Generate the predicted mask for this threshold
            pred_mask = (heatmap >= thresh).astype(np.uint8)
            
            # Calculate the IoU for THIS image
            iou, dice = calculate_iou_dice(pred_mask, gt_mask)

            # --- Create the 3-color comparison mask ---
            # Start with the denormalized image as the base
            vis_image = img_denormalized.copy()
            
            # Create a color overlay
            overlay = np.zeros_like(vis_image, dtype=np.uint8)
            
            # 1. False Negative (GT only)
            overlay[(gt_mask == 1) & (pred_mask == 0)] = color_white_fn
            
            # 2. False Positive (Pred only)
            overlay[(gt_mask == 0) & (pred_mask == 1)] = color_red_fp
            
            # 3. True Positive (Overlap)
            overlay[(gt_mask == 1) & (pred_mask == 1)] = color_green_tp
            
            # Blend the overlay with the original image
            # Find where the overlay is not black
            mask_indices = np.any(overlay > 0, axis=-1)

            if np.any(mask_indices):
                vis_image[mask_indices] = cv2.addWeighted(
                    vis_image[mask_indices], 0.4, overlay[mask_indices], 0.6, 0
                )

            ax.imshow(vis_image)    
            
            # Create a dynamic title
            title_text = f"{plot_title}\n"
            if performer['param'] == "ClassF1":
                title_text += f"Image IoU: {iou:.4f} (Mean IoU: {mean_iou:.4f})"
            else:
                title_text += f"Param: {param:.1f}% | Image IoU: {iou:.4f}\n(Mean IoU: {mean_iou:.4f})"
            
            ax.set_title(title_text)
            ax.axis("off")

        # Add overall stats to the title
        fig.suptitle(f"File: {filename}", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.9]) # Adjust for suptitle
        
        save_path = os.path.join(save_dir, os.path.splitext(filename)[0] + ".png")
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        
    except Exception as e:
        logging.warning(f"Failed to generate visualization for {filename}: {e}", exc_info=True)