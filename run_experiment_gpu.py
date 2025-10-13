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
from torchvision.models import wide_resnet50_2, resnet18, efficientnet_b5, EfficientNet_B5_Weights
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.metrics import average_precision_score
import scipy.stats as sps

# --- Visualization ---
import matplotlib.pyplot as plt

# --- Custom Dataloader ---
# Assumes bowtie.py is in a 'datasets' subfolder
import datasets.bowtie as bowtie
import copy

# --- Global Variables ---
INTERMEDIATE_FEATURE_MAPS = []

# ==========================================================================================
# HELPER FUNCTIONS
# ==========================================================================================


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


def denormalize_image_for_display(tensor_image):
    """Reverses normalization for viewing."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    denormalized_img = (
        ((tensor_image.transpose(1, 2, 0) * std) + mean) * 255.0
    ).astype(np.uint8)
    return denormalized_img


def concatenate_embeddings(larger_map, smaller_map):
    """Aligns and concatenates two feature maps of different spatial resolutions."""
    b, c1, h1, w1 = larger_map.size()
    _, c2, h2, w2 = smaller_map.size()
    stride = int(h1 / h2)
    unfolded = F.unfold(larger_map, kernel_size=stride, dilation=1, stride=stride)
    unfolded = unfolded.view(b, c1, -1, h2, w2)
    output_tensor = torch.zeros(
        b, c1 + c2, unfolded.size(2), h2, w2, device=larger_map.device
    )
    for i in range(unfolded.size(2)):
        patch = unfolded[:, :, i, :, :]
        output_tensor[:, :, i, :, :] = torch.cat((patch, smaller_map), 1)
    output_tensor = output_tensor.view(b, -1, h2 * w2)
    final_embedding = F.fold(
        output_tensor, kernel_size=stride, output_size=(h1, w1), stride=stride
    )
    return final_embedding


# ==========================================================================================
# PLOTTING FUNCTIONS
# ==========================================================================================


def plot_summary_visuals(
    class_save_dir,
    class_name,
    image_roc_auc,
    fpr,
    tpr,
    gt_labels,
    img_scores,
    pr_auc=None,
):
    """Handles the creation and saving of all summary visual plots for a single class."""
    # --- ROC Curve, Score Distribution, Confusion Matrix ---
    # (Implementation is the same as before, code omitted for brevity)
    # --- ROC Curve ---
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label=f"{class_name} ROC AUC: {image_roc_auc:.3f}")
    if pr_auc is not None:
        plt.annotate(
            f"PR-AUC: {pr_auc:.3f}", xy=(0.65, 0.05), xycoords="figure fraction"
        )
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for Class {class_name}")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(class_save_dir, "roc_curve.png"))
    plt.close()

    # --- Score Distribution ---
    plt.figure(figsize=(10, 6))
    normal_scores = img_scores[gt_labels == 0]
    abnormal_scores = img_scores[gt_labels == 1]
    if len(normal_scores) > 0:
        plt.hist(normal_scores, bins=50, label="Normal Scores", color="blue", alpha=0.7)
    if len(abnormal_scores) > 0:
        plt.hist(
            abnormal_scores, bins=50, label="Abnormal Scores", color="red", alpha=0.7
        )
    plt.xlabel("Image-Level Anomaly Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Scores for Normal vs. Abnormal Images")
    plt.legend()
    plt.xlim(0, 1)
    plt.savefig(os.path.join(class_save_dir, "score_distribution.png"))
    plt.close()

    # --- Confusion Matrix ---
    gmeans = np.sqrt(tpr * (1 - fpr))
    best_gmean_index = np.argmax(gmeans)
    thresholds = roc_curve(gt_labels, img_scores)[2]
    optimal_threshold = thresholds[best_gmean_index]
    predictions = (img_scores >= optimal_threshold).astype(int)
    cm = confusion_matrix(gt_labels, predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Normal", "Anomalous"]
    )
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format="g")
    plt.title(f"Confusion Matrix at Threshold = {optimal_threshold:.2f}")
    plt.savefig(os.path.join(class_save_dir, "confusion_matrix.png"))
    plt.close()


def plot_mean_anomaly_maps(class_save_dir, gt_labels, anomaly_maps_raw, score_maps):
    """Calculates and saves a 2x2 plot of the mean anomaly maps."""
    # (Implementation is the same as before, code omitted for brevity)
    logging.info("Generating mean anomaly maps...")
    normal_indices = np.where(gt_labels == 0)[0]
    anomalous_indices = np.where(gt_labels == 1)[0]

    mean_raw_normal = np.zeros_like(anomaly_maps_raw[0])
    mean_upscaled_normal = np.zeros_like(score_maps[0])
    mean_raw_anomalous = np.zeros_like(anomaly_maps_raw[0])
    mean_upscaled_anomalous = np.zeros_like(score_maps[0])

    if len(normal_indices) > 0:
        mean_raw_normal = np.mean(anomaly_maps_raw[normal_indices], axis=0)
        mean_upscaled_normal = np.mean(score_maps[normal_indices], axis=0)
    if len(anomalous_indices) > 0:
        mean_raw_anomalous = np.mean(anomaly_maps_raw[anomalous_indices], axis=0)
        mean_upscaled_anomalous = np.mean(score_maps[anomalous_indices], axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle("Mean Anomaly Maps", fontsize=20)
    # Top Row: Raw Maps
    im1 = axes[0, 0].imshow(mean_raw_normal, cmap="jet")
    axes[0, 0].set_title("Mean Raw Map (Normal Images)")
    fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    im2 = axes[0, 1].imshow(mean_raw_anomalous, cmap="jet")
    axes[0, 1].set_title("Mean Raw Map (Anomalous Images)")
    fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    # Bottom Row: Upscaled Maps
    im3 = axes[1, 0].imshow(mean_upscaled_normal, cmap="jet")
    axes[1, 0].set_title("Mean Upscaled Map (Normal Images)")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    im4 = axes[1, 1].imshow(mean_upscaled_anomalous, cmap="jet")
    axes[1, 1].set_title("Mean Upscaled Map (Anomalous Images)")
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    save_path = os.path.join(class_save_dir, "mean_anomaly_maps_comparison.png")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path)
    plt.close(fig)


def plot_individual_visualizations(
    test_images,
    raw_maps,
    norm_scores,
    img_scores,
    save_dir,
    test_filepaths,
    per_image_stats=None,
    optimal_threshold=None,
):
    """Generates and saves a 4-panel visualization for each test image into subfolders."""
    logging.info(f"Generating {len(test_images)} individual visualizations...")
    for i in range(len(test_images)):
        img_original, raw_map, final_heatmap, score = (
            test_images[i],
            raw_maps[i],
            norm_scores[i],
            img_scores[i],
        )
        img_denormalized = denormalize_image_for_display(img_original)

        filepath = test_filepaths[i]
        defect_type = os.path.basename(os.path.dirname(filepath))
        image_filename = os.path.splitext(os.path.basename(filepath))[0]

        # Determine subfolder and create it
        subfolder = "normal" if defect_type == "good" else "anomalous"
        target_dir = os.path.join(save_dir, subfolder)
        os.makedirs(target_dir, exist_ok=True)

        # *** MODIFIED: Re-added the defect_type to the filename ***
        save_filename = f"{defect_type}_{image_filename}.png"

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # 1. Original Image
        axes[0].imshow(img_denormalized)
        axes[0].set_title("1. Original Image")

        # 2. Patch Score Distribution
        axes[1].hist(raw_map.ravel(), bins=60, color="skyblue", edgecolor="black")
        axes[1].set_title("2. Patch Score Distribution")
        axes[1].set_xlabel("Mahalanobis Distance")
        axes[1].set_ylabel("Frequency")
        axes[1].grid(axis="y", alpha=0.6)

        # 3. Raw Anomaly Map
        im = axes[2].imshow(raw_map, cmap="jet")
        axes[2].set_title("3. Raw Anomaly Map")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        # 4. Final Heatmap Overlay
        axes[3].imshow(img_denormalized)
        axes[3].imshow(final_heatmap, cmap="jet", alpha=0.5)
        axes[3].set_title("4. Final Heatmap Overlay")
        # Overlay a compact stats box in the top-left of the last panel
        stats_lines = [f"Score: {score:.3f}"]
        if per_image_stats is not None:
            s = per_image_stats[i]
            stats_lines.append(f"Max: {s['max']:.3f}")
            stats_lines.append(f"MeanTop1%: {s['mean_top1pct']:.3f}")
            stats_lines.append(f"P95: {s['p95']:.3f}")
            stats_lines.append(f"Frac>{s['threshold']:.3f}: {s['frac_above']*100:.1f}%")
        if optimal_threshold is not None:
            pred_label = "Anomalous" if score >= optimal_threshold else "Normal"
            stats_lines.append(f"Pred: {pred_label}")

        stats_text = "\n".join(stats_lines)
        axes[3].text(
            0.02,
            0.98,
            stats_text,
            transform=axes[3].transAxes,
            fontsize=10,
            color="white",
            verticalalignment="top",
            bbox=dict(facecolor="black", alpha=0.6, pad=6),
        )

        # Turn off axes for image plots only
        axes[0].axis("off")
        axes[2].axis("off")
        axes[3].axis("off")

        plt.tight_layout()
        fig.savefig(
            os.path.join(target_dir, save_filename), dpi=100, bbox_inches="tight"
        )
        plt.close(fig)


def plot_patch_score_distributions(class_save_dir, gt_labels, anomaly_maps_raw):
    """Generates separate and overlaid patch score distribution plots with fixed axes."""
    logging.info("Generating detailed patch score distribution plots...")
    bins = 300
    normal_indices = np.where(gt_labels == 0)[0]
    abnormal_indices = np.where(gt_labels == 1)[0]

    all_normal_patch_scores = None
    all_abnormal_patch_scores = None

    # --- Data Preparation ---
    if len(normal_indices) > 0:
        all_normal_patch_scores = anomaly_maps_raw[normal_indices].flatten()
    if len(abnormal_indices) > 0:
        all_abnormal_patch_scores = anomaly_maps_raw[abnormal_indices].flatten()

    # --- NEW: Dynamically determine x-axis maximum based on the absolute max score ---
    combined_scores = []
    if all_normal_patch_scores is not None:
        combined_scores.append(all_normal_patch_scores)
    if all_abnormal_patch_scores is not None:
        combined_scores.append(all_abnormal_patch_scores)

    # *** MODIFIED: Use np.max() to include all values, and add 5% padding ***
    x_max = np.max(np.concatenate(combined_scores)) * 1.05 if combined_scores else 1.0

    # --- 1. Plot for Normal Patches ---
    if all_normal_patch_scores is not None:
        fig = plt.figure(figsize=(10, 6))
        plt.hist(
            all_normal_patch_scores,
            bins=bins,
            label="Normal Patches",
            color="lightblue",
            alpha=0.7,
            density=True,
            edgecolor="black",
        )
        plt.title("Distribution of All Patch Scores (Normal Images)")
        plt.xlabel("Patch-level Mahalanobis Distance (Anomaly Score)")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.grid(axis="y", alpha=0.5)
        plt.xlim(0, x_max)  # Set x-axis limits
        plt.savefig(os.path.join(class_save_dir, "patch_score_distribution_normal.png"))
        plt.close(fig)

    # --- 2. Plot for Abnormal Patches ---
    if all_abnormal_patch_scores is not None:
        fig = plt.figure(figsize=(10, 6))
        plt.hist(
            all_abnormal_patch_scores,
            bins=bins,
            label="Abnormal Patches",
            color="red",
            alpha=0.7,
            density=True,
            edgecolor="black",
        )
        plt.title("Distribution of All Patch Scores (Abnormal Images)")
        plt.xlabel("Patch-level Mahalanobis Distance (Anomaly Score)")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.grid(axis="y", alpha=0.5)
        plt.xlim(0, x_max)  # Set x-axis limits
        plt.savefig(
            os.path.join(class_save_dir, "patch_score_distribution_abnormal.png")
        )
        plt.close(fig)

    # --- 3. Plot for Overlaid Patches ---
    if all_normal_patch_scores is not None and all_abnormal_patch_scores is not None:
        fig = plt.figure(figsize=(12, 7))
        plt.hist(
            all_normal_patch_scores,
            bins=bins,
            label="Normal Patches",
            color="lightblue",
            alpha=0.7,
            density=True,
            edgecolor="black",
        )
        plt.hist(
            all_abnormal_patch_scores,
            bins=bins,
            label="Abnormal Patches",
            color="red",
            alpha=0.4,
            density=True,
        )
        plt.title("Distribution of All Patch Scores: Normal vs. Abnormal")
        plt.xlabel("Patch-level Mahalanobis Distance (Anomaly Score)")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.grid(axis="y", alpha=0.2)
        plt.xlim(0, x_max)  # Set x-axis limits
        plt.savefig(
            os.path.join(class_save_dir, "patch_score_distribution_comparative.png")
        )
        plt.close(fig)


# ==========================================================================================
# CORE LOGIC
# ==========================================================================================


def run_class_processing(args, class_name, model, device, random_feature_indices):
    """Main function to process a single class: load data, train, test, and save results."""
    global INTERMEDIATE_FEATURE_MAPS

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
    
    logging.info("------------------- EXPERIMENT CONFIGURATION -------------------")
    
    # --- Dataset & Preprocessing Details ---
    logging.info(f"[DATASET] Class Name: {class_name}")
    logging.info(f"[DATASET] Data Path: {args.data_path}")
    logging.info(f"[DATASET] Train Set Size: {len(data_manager.train)} images")
    logging.info(f"[DATASET] Test Set Size: {len(data_manager.test)} images")
    normal_test_count = sum(1 for label in data_manager.test.labels if label == 0)
    abnormal_test_count = len(data_manager.test.labels) - normal_test_count
    logging.info(f"[DATASET] Test Set Composition: {normal_test_count} Normal, {abnormal_test_count} Abnormal")
    logging.info(f"[PREPROC] Image Resize Target: {args.resize}")
    logging.info(f"[PREPROC] Center Crop Size: {args.cropsize}")
    
    # --- Augmentation Settings ---
    if args.augmentations_enabled:
        logging.info("[AUGMENT] Status: ENABLED")
        logging.info(f"[AUGMENT] Horizontal Flip: {args.horizontal_flip}")
        logging.info(f"[AUGMENT] Vertical Flip: {args.vertical_flip}")
        logging.info(f"[AUGMENT] Application Probability: {args.augmentation_prob}")
    else:
        logging.info("[AUGMENT] Status: DISABLED")

    # --- Model & Embedding Details ---
    logging.info(f"[MODEL] Architecture: {args.model_architecture}")
    # Temporarily run one dummy image to get feature map sizes
    dummy_input = torch.randn(1, 3, args.cropsize, args.cropsize).to(device)
    with torch.no_grad():
        _ = model(dummy_input)
    
    # This directly answers your question about the "amount of distances"
    map_shapes = [f.shape for f in INTERMEDIATE_FEATURE_MAPS]
    final_h, final_w = map_shapes[-1][-2], map_shapes[-1][-1]
    total_patches = final_h * final_w
    logging.info(f"[MODEL] Intermediate Feature Map Shapes (B,C,H,W): {map_shapes}")
    logging.info(f"[MODEL] Final Anomaly Map Grid Size (H x W): {final_h} x {final_w}")
    logging.info(f"[MODEL] Total Patches per Image: {total_patches}")
    
    INTERMEDIATE_FEATURE_MAPS = [] # Clear hooks after dummy pass
    
    total_dim = sum(shape[1] for shape in map_shapes)
    logging.info(f"[EMBED] Total Feature Dimension (Concatenated Channels): {total_dim}")
    logging.info(f"[EMBED] Reduced Feature Dimension (Randomly Selected): {len(random_feature_indices)}")

    # --- PaDiM & Run Parameters ---
    logging.info(f"[PADIM] Covariance Regularization (epsilon): 0.01")
    logging.info(f"[SYSTEM] Random Seed: {args.seed}")
    logging.info(f"[SYSTEM] Device: {device}")
    logging.info(f"[SYSTEM] Mahalanobis on GPU: {args.mahalanobis_on_gpu}")
    logging.info("------------------------------------------------------------------")
    
    train_dataloader = DataLoader(
        data_manager.train, batch_size=args.batch_size, pin_memory=True, shuffle=True
    )
    test_dataloader = DataLoader(
        data_manager.test, batch_size=args.batch_size, pin_memory=True
    )
    logging.info(f"Results for this class will be saved in: {class_save_dir}")

    # --- 2. Learn Distribution ---
    # (Implementation is the same as before, code omitted for brevity)
    dist_path = os.path.join(class_save_dir, "learned_distribution.pkl")
    if not os.path.exists(dist_path):
        logging.info("No cached distribution found. Learning from scratch...")
        train_feature_maps = OrderedDict(
            [("layer1", []), ("layer2", []), ("layer3", [])]
        )

        model.eval()
        for image_batch, _, _ in train_dataloader:
            with torch.no_grad():
                _ = model(image_batch.to(device))
            for layer, feat in zip(
                train_feature_maps.keys(), INTERMEDIATE_FEATURE_MAPS
            ):
                train_feature_maps[layer].append(feat.cpu().detach())
            INTERMEDIATE_FEATURE_MAPS = []

        for layer, feat_list in train_feature_maps.items():
            train_feature_maps[layer] = torch.cat(feat_list, 0)

        embedding_vectors = train_feature_maps["layer1"]
        for layer in ["layer2", "layer3"]:
            embedding_vectors = concatenate_embeddings(
                embedding_vectors, train_feature_maps[layer]
            )

        embedding_vectors = torch.index_select(
            embedding_vectors, 1, random_feature_indices
        )

        b, c, h, w = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(b, c, h * w)
        mean_vectors = torch.mean(embedding_vectors, dim=0).numpy()
        cov_matrices = torch.zeros(c, c, h * w).numpy()
        identity = np.identity(c)
        for i in range(h * w):
            cov_matrices[:, :, i] = (
                np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False)
                + 0.01 * identity
            )

        learned_distribution = [mean_vectors, cov_matrices]

        if args.save_distribution:
            with open(dist_path, "wb") as f:
                pickle.dump(learned_distribution, f)
            logging.info(f"Saved learned distribution to: {dist_path}")
        else:
            logging.info(
                "`--save_distribution` is false. Distribution will not be saved."
            )

    else:
        logging.info(f"Loading cached distribution from: {dist_path}")
        with open(dist_path, "rb") as f:
            learned_distribution = pickle.load(f)

    # --- 3. Evaluation & Anomaly Scoring ---
    # (Implementation is the same as before, code omitted for brevity)
    logging.info("Extracting features from the test set...")
    test_feature_maps = OrderedDict([("layer1", []), ("layer2", []), ("layer3", [])])
    test_images_list, ground_truth_labels = [], []

    model.eval()
    for image_batch, labels, _ in test_dataloader:
        test_images_list.extend(image_batch.cpu().detach().numpy())
        ground_truth_labels.extend(labels.cpu().detach().numpy())
        with torch.no_grad():
            _ = model(image_batch.to(device))
        for layer, feat in zip(test_feature_maps.keys(), INTERMEDIATE_FEATURE_MAPS):
            test_feature_maps[layer].append(feat.cpu().detach())
        INTERMEDIATE_FEATURE_MAPS = []

    for layer, feat_list in test_feature_maps.items():
        test_feature_maps[layer] = torch.cat(feat_list, 0)

    embedding_vectors_test = test_feature_maps["layer1"]
    for layer in ["layer2", "layer3"]:
        embedding_vectors_test = concatenate_embeddings(
            embedding_vectors_test, test_feature_maps[layer]
        )

    embedding_vectors_test = torch.index_select(
        embedding_vectors_test, 1, random_feature_indices
    )

    logging.info("Calculating Mahalanobis distances...")
    # Branch: allow CPU fallback for Mahalanobis if requested
    if getattr(args, "mahalanobis_on_gpu", False):
        logging.info("Calculating Mahalanobis distances on GPU...")
        # --- 1. Move all necessary data to the GPU ---
        mean_gpu = torch.tensor(
            learned_distribution[0], device=device, dtype=torch.float32
        )
        cov_gpu = torch.tensor(
            learned_distribution[1], device=device, dtype=torch.float32
        )
        # Permute and invert covariance matrices just once
        inv_cov_gpu = torch.linalg.inv(cov_gpu.permute(2, 0, 1))
        # Permute mean vectors just once
        mean_gpu = mean_gpu.permute(1, 0)

        # --- B. Process the test set one batch at a time ---
        test_images_list, ground_truth_labels, all_distances = [], [], []
        model.eval()
        for image_batch, labels, _ in test_dataloader:
            # Store metadata (small, so this is fine)
            test_images_list.extend(image_batch.cpu().detach().numpy())
            ground_truth_labels.extend(labels.cpu().detach().numpy())

            # --- Step 1: Feature Extraction for the current batch ---
            with torch.no_grad():
                _ = model(image_batch.to(device))

            # The hooks in INTERMEDIATE_FEATURE_MAPS now hold features for this batch only
            embedding_vectors_batch = INTERMEDIATE_FEATURE_MAPS[0]
            for i in range(1, len(INTERMEDIATE_FEATURE_MAPS)):
                embedding_vectors_batch = concatenate_embeddings(
                    embedding_vectors_batch, INTERMEDIATE_FEATURE_MAPS[i]
                )

            embedding_vectors_batch = torch.index_select(
                embedding_vectors_batch, 1, random_feature_indices.to(device)
            )
            INTERMEDIATE_FEATURE_MAPS = []  # CRITICAL: Clear list for the next batch

            # --- Step 2: Mahalanobis Distance for the current batch ---
            b, c, h, w = embedding_vectors_batch.size()
            num_patches = h * w
            embedding_vectors_batch = embedding_vectors_batch.view(
                b, c, num_patches
            ).permute(0, 2, 1)

            diff = (
                embedding_vectors_batch - mean_gpu
            )  # Broadcasting handles the math here

            dist_squared = torch.sum(
                torch.einsum("bpc,pcd->bpd", diff, inv_cov_gpu) * diff, dim=2
            )
            distances_batch = torch.sqrt(dist_squared)

            # --- Step 3: Store the small result and discard the large feature tensor ---
            all_distances.append(distances_batch.cpu().numpy())
            # The large `embedding_vectors_batch` tensor is now cleared from memory
    else:
        logging.info("Calculating Mahalanobis distances on CPU (fallback)...")
        # CPU implementation (process patches and invert cov on CPU)
        b_all = len(test_feature_maps[list(test_feature_maps.keys())[0]])
        # We'll recompute distances in a loop over patch locations similar to the CPU runner
        embedding_vectors_test = embedding_vectors_test.cpu()
        mean_vectors = learned_distribution[0]
        cov_matrices = learned_distribution[1]

        distances = []
        # embedding_vectors_test shape: (b, c, h, w)
        b, c, h, w = embedding_vectors_test.size()
        embedding_reshaped = embedding_vectors_test.view(b, c, h * w)
        # For each patch location, compute Mahalanobis on CPU
        for i_patch in range(h * w):
            mean_patch = mean_vectors[:, i_patch]
            cov_inv_patch = np.linalg.inv(cov_matrices[:, :, i_patch])
            diff = embedding_reshaped[:, :, i_patch].numpy() - mean_patch
            # diff shape: (b, c)
            dist = np.sqrt(np.einsum("bi,ij,bj->b", diff, cov_inv_patch, diff))
            distances.append(dist)
        distances = np.array(distances).transpose(1, 0)

    # --- C. Post-processing after the loop is finished ---
    # Concatenate the small distance results from all batches (GPU path)
    # or use the precomputed `distances` from the CPU fallback.
    if "all_distances" in locals() and len(all_distances) > 0:
        distances = np.concatenate(all_distances, axis=0)
    elif "distances" in locals():
        # `distances` was already computed by the CPU fallback branch.
        pass
    else:
        raise RuntimeError(
            "No Mahalanobis distances were computed. Check GPU/CPU Mahalanobis branches."
        )

    # Reshape the final array of distances into anomaly maps
    b = len(test_images_list)
    # h and w are correctly determined from the last batch's feature map dimensions
    anomaly_maps_raw = distances.reshape(b, h, w)

    # Normalize the raw anomaly maps to [0,1] for plotting and spatial measures.
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
    logging.info("Calculating metrics and saving results...")
    max_score, min_score = score_maps.max(), score_maps.min()
    normalized_scores = (score_maps - min_score) / (max_score - min_score)
    image_level_scores = normalized_scores.reshape(normalized_scores.shape[0], -1).max(
        axis=1
    )
    ground_truth_labels = np.asarray(ground_truth_labels)

    # AUC / PR-AUC
    image_roc_auc = roc_auc_score(ground_truth_labels, image_level_scores)
    try:
        image_pr_auc = average_precision_score(ground_truth_labels, image_level_scores)
    except Exception:
        image_pr_auc = None

    fpr, tpr, thresholds = roc_curve(ground_truth_labels, image_level_scores)
    logging.info(f"Image-level ROC AUC for class '{class_name}': {image_roc_auc:.4f}")

    # Determine an operational threshold (maximize G-mean) and predictions
    gmeans = np.sqrt(tpr * (1 - fpr))
    best_idx = np.argmax(gmeans)
    thresholds_from_roc = roc_curve(ground_truth_labels, image_level_scores)[2]
    optimal_threshold = thresholds_from_roc[best_idx]
    predictions = (image_level_scores >= optimal_threshold).astype(int)

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
    # --- Compute per-image statistics used for overlays and group summaries ---
    # Compute a threshold based on the normal images (99th percentile) in the
    # normalized score space. Fall back to overall 99th percentile if no normals.
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

        # Spatial features on the normalized upscaled map
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

    # Save per-image metrics to CSV
    per_image_csv_path = os.path.join(class_save_dir, "per_image_metrics.csv")
    import csv

    with open(per_image_csv_path, "w", newline="") as csvfile:
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

    # NOTE: Per-class JSON report will be created after group-level metrics
    # (cohen_d, wass, mw_p, auc CI) are computed further down to ensure all
    # referenced variables are defined. See later in this function where the
    # report is written after those calculations.

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
    # *** NEW: Call the new plotting function ***
    plot_patch_score_distributions(
        class_save_dir, ground_truth_labels, anomaly_maps_raw
    )
    logging.info(f"All individual results for class '{class_name}' saved.")

    # --- 6. Calculate Final Metrics & Prepare Data for Master CSV ---
    gmeans = np.sqrt(tpr * (1 - fpr))
    optimal_threshold = thresholds[np.argmax(gmeans)]
    predictions = (image_level_scores >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(ground_truth_labels, predictions).ravel()

    # --- NEW: Calculate Precision, Recall, and F1-Score ---
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-6
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

    logging.info(
        f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1_score:.4f}"
    )
    # --- NEW: Group-level effect sizes and distribution measures ---
    normal_scores = image_level_scores[ground_truth_labels == 0]
    anomalous_scores = image_level_scores[ground_truth_labels == 1]

    # Cohen's d (pooled)
    def cohens_d(a, b):
        nx, ny = len(a), len(b)
        dof = nx + ny - 2
        pooled_std = np.sqrt(
            ((nx - 1) * np.var(a, ddof=1) + (ny - 1) * np.var(b, ddof=1)) / dof
        )
        return (np.mean(a) - np.mean(b)) / (pooled_std + 1e-12)

    cohen_d = None
    try:
        if len(normal_scores) > 1 and len(anomalous_scores) > 1:
            cohen_d = float(cohens_d(anomalous_scores, normal_scores))
    except Exception:
        cohen_d = None

    # Wasserstein distance
    try:
        wass = float(sps.wasserstein_distance(normal_scores, anomalous_scores))
    except Exception:
        wass = None

    # Mann-Whitney U test
    try:
        mw_stat, mw_p = sps.mannwhitneyu(
            normal_scores, anomalous_scores, alternative="two-sided"
        )
    except Exception:
        mw_stat, mw_p = None, None

    # Bootstrap CI for ROC AUC
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
        low = np.percentile(bootstrapped_scores, 2.5)
        high = np.percentile(bootstrapped_scores, 97.5)
        return low, high

    try:
        auc_ci_low, auc_ci_high = bootstrap_auc(
            ground_truth_labels, image_level_scores, n_bootstrap=200
        )
    except Exception:
        auc_ci_low, auc_ci_high = None, None

    # --- Append new metrics to the results.txt file ---
    with open(os.path.join(class_save_dir, "results.txt"), "a") as f:
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

    # --- Save a per-class JSON summary report (now that group metrics are available) ---
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
        import json

        with open(class_report_path, "w") as jf:
            json.dump(class_report, jf, indent=2)
    except Exception:
        logging.exception("Failed to write class_report.json")
    # --- NEW: Add new metrics to the dictionary for the master CSV ---
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
    logging.info(f"Running on device: {device}")
    logging.info(f"Script arguments: {vars(args)}")

    # --- 2. Model & Feature Selection ---
    if args.model_architecture == "wide_resnet50_2":
        model = wide_resnet50_2(weights="DEFAULT") # Using modern weights call
        total_dim, reduced_dim = 1792, 550
    elif args.model_architecture == "resnet18":
        model = resnet18(weights="DEFAULT") # Using modern weights call
        total_dim, reduced_dim = 448, 100
    elif args.model_architecture == "efficientnet_b5":
        # --- UPDATED: Use the official torchvision loader ---
        model = efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
        # For EfficientNet-B5, the output channels of blocks 2, 4, and 6 are:
        # Block 2: 40 channels
        # Block 4: 112 channels
        # Block 6: 320 channels
        # Total = 40 + 112 + 320 = 472
        total_dim, reduced_dim = 472, 400


    model.to(device)
    model.eval()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    random_feature_indices = torch.tensor(
        random.sample(range(0, total_dim), reduced_dim)
    )

    if 'resnet' in args.model_architecture:
        model.layer1[-1].register_forward_hook(hook_function)
        model.layer2[-1].register_forward_hook(hook_function)
        model.layer3[-1].register_forward_hook(hook_function)
    elif 'efficientnet' in args.model_architecture:
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
