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
import csv
import json

import datasets.bowtie as bowtie
from utils.embeddings import get_batch_embeddings
from utils.plotting import (
    plot_summary_visuals,
    plot_mean_anomaly_maps,
    plot_individual_visualizations,
    plot_patch_score_distributions,
)


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

    test_images_list, ground_truth_labels, all_distances = [], [], []
    model.eval()
    for image_batch, labels, _ in test_dataloader:
        test_images_list.extend(image_batch.cpu().numpy())
        ground_truth_labels.extend(labels.cpu().numpy())

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

    # Per-image metrics CSV writing removed per request. `per_image_stats` is still
    # computed and passed to plotting functions, but we will not persist a
    # `per_image_metrics.csv` file to avoid creating large per-image artifacts.
    logging.info("Per-image metrics CSV export disabled by configuration.")

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
    )

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

    # Writing to results.txt has been removed per request; log the summary instead.
    logging.info(f"Optimal Threshold: {optimal_threshold:.4f}")
    logging.info(
        f"True Negatives: {tn} | False Positives: {fp} | False Negatives: {fn} | True Positives: {tp}"
    )
    logging.info(
        f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1_score:.4f}"
    )
    if cohen_d is not None:
        logging.info(f"Cohen_d: {cohen_d:.4f}")
    if wass is not None:
        logging.info(f"Wasserstein: {wass:.4f}")
    if mw_p is not None:
        logging.info(f"Mann-Whitney p: {mw_p:.4e}")
    if auc_ci_low is not None:
        logging.info(f"ROC AUC CI: [{auc_ci_low:.4f}, {auc_ci_high:.4f}]")

    try:
        class_report = {
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
