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

# --- Machine Learning and Deep Learning ---
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

# --- Visualization ---
import matplotlib.pyplot as plt

# --- Custom Dataloader ---
# Assumes bowtie.py is in a 'datasets' subfolder
import datasets.bowtie as bowtie

# --- Global Variables ---
INTERMEDIATE_FEATURE_MAPS = []

# ==========================================================================================
# HELPER FUNCTIONS
# ==========================================================================================

def setup_logging(log_path, log_name):
    """Configures a master logger for the entire experiment run."""
    log_file = os.path.join(log_path, f"{log_name}.log")
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file, mode='w'),
                            logging.StreamHandler()
                        ])
    logging.info(f"Logging initialized. Log file at: {log_file}")

def hook_function(module, input, output):
    """A simple hook that appends the output of a layer to a global list."""
    INTERMEDIATE_FEATURE_MAPS.append(output)

def denormalize_image_for_display(tensor_image):
    """Reverses normalization for viewing."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    denormalized_img = (((tensor_image.transpose(1, 2, 0) * std) + mean) * 255.0).astype(np.uint8)
    return denormalized_img

def concatenate_embeddings(larger_map, smaller_map):
    """Aligns and concatenates two feature maps of different spatial resolutions."""
    b, c1, h1, w1 = larger_map.size()
    _, c2, h2, w2 = smaller_map.size()
    stride = int(h1 / h2)
    unfolded = F.unfold(larger_map, kernel_size=stride, dilation=1, stride=stride)
    unfolded = unfolded.view(b, c1, -1, h2, w2)
    output_tensor = torch.zeros(b, c1 + c2, unfolded.size(2), h2, w2)
    for i in range(unfolded.size(2)):
        patch = unfolded[:, :, i, :, :]
        output_tensor[:, :, i, :, :] = torch.cat((patch, smaller_map), 1)
    output_tensor = output_tensor.view(b, -1, h2 * w2)
    final_embedding = F.fold(output_tensor, kernel_size=stride, output_size=(h1, w1), stride=stride)
    return final_embedding

# ==========================================================================================
# PLOTTING FUNCTIONS
# ==========================================================================================

def plot_summary_visuals(class_save_dir, class_name, image_roc_auc, fpr, tpr, gt_labels, img_scores):
    """Handles the creation and saving of all summary visual plots for a single class."""
    # --- ROC Curve, Score Distribution, Confusion Matrix ---
    # (Implementation is the same as before, code omitted for brevity)
    # --- ROC Curve ---
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label=f"{class_name} ROC AUC: {image_roc_auc:.3f}")
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
        plt.hist(abnormal_scores, bins=50, label="Abnormal Scores", color="red", alpha=0.7)
    plt.xlabel("Image-Level Anomaly Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Scores for Normal vs. Abnormal Images")
    plt.legend()
    plt.savefig(os.path.join(class_save_dir, "score_distribution.png"))
    plt.close()

    # --- Confusion Matrix ---
    gmeans = np.sqrt(tpr * (1 - fpr))
    best_gmean_index = np.argmax(gmeans)
    thresholds = roc_curve(gt_labels, img_scores)[2]
    optimal_threshold = thresholds[best_gmean_index]
    predictions = (img_scores >= optimal_threshold).astype(int)
    cm = confusion_matrix(gt_labels, predictions)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomalous"])
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
    im1 = axes[0, 0].imshow(mean_raw_normal, cmap='jet')
    axes[0, 0].set_title("Mean Raw Map (Normal Images)")
    fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    im2 = axes[0, 1].imshow(mean_raw_anomalous, cmap='jet')
    axes[0, 1].set_title("Mean Raw Map (Anomalous Images)")
    fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    # Bottom Row: Upscaled Maps
    im3 = axes[1, 0].imshow(mean_upscaled_normal, cmap='jet')
    axes[1, 0].set_title("Mean Upscaled Map (Normal Images)")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    im4 = axes[1, 1].imshow(mean_upscaled_anomalous, cmap='jet')
    axes[1, 1].set_title("Mean Upscaled Map (Anomalous Images)")
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    for ax in axes.ravel():
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    save_path = os.path.join(class_save_dir, "mean_anomaly_maps_comparison.png")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path)
    plt.close(fig)

def plot_individual_visualizations(test_images, raw_maps, norm_scores, img_scores, save_dir, test_filepaths):
    """Generates and saves a 4-panel visualization for each test image into subfolders."""
    logging.info(f"Generating {len(test_images)} individual visualizations...")
    
    for i in range(len(test_images)):
        img_original, raw_map, final_heatmap, score = test_images[i], raw_maps[i], norm_scores[i], img_scores[i]
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
        axes[1].hist(raw_map.ravel(), bins=60, color='skyblue', edgecolor='black')
        axes[1].set_title("2. Patch Score Distribution")
        axes[1].set_xlabel("Mahalanobis Distance")
        axes[1].set_ylabel("Frequency")
        axes[1].grid(axis='y', alpha=0.6)

        # 3. Raw Anomaly Map
        im = axes[2].imshow(raw_map, cmap="jet")
        axes[2].set_title("3. Raw Anomaly Map")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        # 4. Final Heatmap Overlay
        axes[3].imshow(img_denormalized)
        axes[3].imshow(final_heatmap, cmap="jet", alpha=0.5)
        axes[3].set_title("4. Final Heatmap Overlay")
        axes[3].text(5, 20, f"Score: {score:.3f}", color="white", backgroundcolor="black")

        # Turn off axes for image plots only
        axes[0].axis('off')
        axes[2].axis('off')
        axes[3].axis('off')
        
        plt.tight_layout()
        fig.savefig(os.path.join(target_dir, save_filename), dpi=100, bbox_inches='tight')
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
        plt.hist(all_normal_patch_scores, bins=bins, label='Normal Patches', color='lightblue', alpha=0.7, density=True, edgecolor='black')
        plt.title('Distribution of All Patch Scores (Normal Images)')
        plt.xlabel('Patch-level Mahalanobis Distance (Anomaly Score)')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(axis='y', alpha=0.5)
        plt.xlim(0, x_max)  # Set x-axis limits
        plt.savefig(os.path.join(class_save_dir, "patch_score_distribution_normal.png"))
        plt.close(fig)
    
    # --- 2. Plot for Abnormal Patches ---
    if all_abnormal_patch_scores is not None:
        fig = plt.figure(figsize=(10, 6))
        plt.hist(all_abnormal_patch_scores, bins=bins, label='Abnormal Patches', color='red', alpha=0.7, density=True, edgecolor='black')
        plt.title('Distribution of All Patch Scores (Abnormal Images)')
        plt.xlabel('Patch-level Mahalanobis Distance (Anomaly Score)')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(axis='y', alpha=0.5)
        plt.xlim(0, x_max)  # Set x-axis limits
        plt.savefig(os.path.join(class_save_dir, "patch_score_distribution_abnormal.png"))
        plt.close(fig)
        
    # --- 3. Plot for Overlaid Patches ---
    if all_normal_patch_scores is not None and all_abnormal_patch_scores is not None:
        fig = plt.figure(figsize=(12, 7))
        plt.hist(all_normal_patch_scores, bins=bins, label='Normal Patches', color='lightblue', alpha=0.7, density=True, edgecolor='black')
        plt.hist(all_abnormal_patch_scores, bins=bins, label='Abnormal Patches', color='red', alpha=0.4, density=True)
        plt.title('Distribution of All Patch Scores: Normal vs. Abnormal')
        plt.xlabel('Patch-level Mahalanobis Distance (Anomaly Score)')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(axis='y', alpha=0.2)
        plt.xlim(0, x_max)  # Set x-axis limits
        plt.savefig(os.path.join(class_save_dir, "patch_score_distribution_comparative.png"))
        plt.close(fig)

# ==========================================================================================
# CORE LOGIC
# ==========================================================================================

def run_class_processing(args, class_name, model, device, random_feature_indices):
    """ Main function to process a single class: load data, train, test, and save results. """
    global INTERMEDIATE_FEATURE_MAPS
    
    logging.info(f"--- Starting processing for CLASS: {class_name} ---")

    # --- 1. Setup & Dataloading ---
    class_save_dir = os.path.join(args.master_save_dir, class_name)
    os.makedirs(class_save_dir, exist_ok=True)
    data_manager = bowtie.BowtieDataManager(
        dataset_path=args.data_path, class_name=class_name, resize=args.resize,
        cropsize=args.cropsize, seed=args.seed
    )
    train_dataloader = DataLoader(data_manager.train, batch_size=args.batch_size, pin_memory=True, shuffle=True)
    test_dataloader = DataLoader(data_manager.test, batch_size=args.batch_size, pin_memory=True)
    logging.info(f"Results for this class will be saved in: {class_save_dir}")

    # --- 2. Learn Distribution ---
    # (Implementation is the same as before, code omitted for brevity)
    dist_path = os.path.join(class_save_dir, "learned_distribution.pkl")
    if not os.path.exists(dist_path):
        logging.info("No cached distribution found. Learning from scratch...")
        train_feature_maps = OrderedDict([("layer1", []), ("layer2", []), ("layer3", [])])

        model.eval()
        for image_batch, _, _ in train_dataloader:
            with torch.no_grad():
                _ = model(image_batch.to(device))
            for layer, feat in zip(train_feature_maps.keys(), INTERMEDIATE_FEATURE_MAPS):
                train_feature_maps[layer].append(feat.cpu().detach())
            INTERMEDIATE_FEATURE_MAPS = []

        for layer, feat_list in train_feature_maps.items():
            train_feature_maps[layer] = torch.cat(feat_list, 0)
        
        embedding_vectors = train_feature_maps["layer1"]
        for layer in ["layer2", "layer3"]:
            embedding_vectors = concatenate_embeddings(embedding_vectors, train_feature_maps[layer])
        
        embedding_vectors = torch.index_select(embedding_vectors, 1, random_feature_indices)
        
        b, c, h, w = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(b, c, h * w)
        mean_vectors = torch.mean(embedding_vectors, dim=0).numpy()
        cov_matrices = torch.zeros(c, c, h * w).numpy()
        identity = np.identity(c)
        for i in range(h * w):
            cov_matrices[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * identity
        
        learned_distribution = [mean_vectors, cov_matrices]
        
        if args.save_distribution:
            with open(dist_path, "wb") as f:
                pickle.dump(learned_distribution, f)
            logging.info(f"Saved learned distribution to: {dist_path}")
        else:
            logging.info("`--save_distribution` is false. Distribution will not be saved.")
            
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
        embedding_vectors_test = concatenate_embeddings(embedding_vectors_test, test_feature_maps[layer])

    embedding_vectors_test = torch.index_select(embedding_vectors_test, 1, random_feature_indices)
    
    logging.info("Calculating Mahalanobis distances...")
    b, c, h, w = embedding_vectors_test.size()
    embedding_vectors_test = embedding_vectors_test.view(b, c, h * w).cpu()
    mean_vectors = learned_distribution[0]
    cov_matrices = learned_distribution[1]
    
    distances = []
    for i in range(h * w):
        mean_patch = mean_vectors[:, i]
        cov_inv_patch = np.linalg.inv(cov_matrices[:, :, i])
        diff = embedding_vectors_test[:, :, i].numpy() - mean_patch
        dist = np.sqrt(np.einsum('bi,ij,bj->b', diff, cov_inv_patch, diff))
        distances.append(dist)
        
    anomaly_maps_raw = np.array(distances).transpose(1, 0).reshape(b, h, w)
    
    score_maps = F.interpolate(torch.tensor(anomaly_maps_raw).unsqueeze(1), size=args.cropsize,
                               mode="bilinear", align_corners=False).squeeze().numpy()
    
    for i in range(score_maps.shape[0]):
        score_maps[i] = gaussian_filter(score_maps[i], sigma=4)

    # --- 4. Calculate Metrics ---
    logging.info("Calculating metrics and saving results...")
    max_score, min_score = score_maps.max(), score_maps.min()
    normalized_scores = (score_maps - min_score) / (max_score - min_score)
    image_level_scores = normalized_scores.reshape(normalized_scores.shape[0], -1).max(axis=1)
    ground_truth_labels = np.asarray(ground_truth_labels)
    image_roc_auc = roc_auc_score(ground_truth_labels, image_level_scores)
    fpr, tpr, thresholds = roc_curve(ground_truth_labels, image_level_scores)
    logging.info(f"Image-level ROC AUC for class '{class_name}': {image_roc_auc:.4f}")
    with open(os.path.join(class_save_dir, "results.txt"), "w") as f:
        f.write(f"Image-level ROC AUC: {image_roc_auc:.4f}\n")

    # --- 5. Generate and Save All Visualizations ---
    plot_summary_visuals(class_save_dir, class_name, image_roc_auc, fpr, tpr, ground_truth_labels, image_level_scores)
    plot_mean_anomaly_maps(class_save_dir, ground_truth_labels, anomaly_maps_raw, score_maps)
    plot_individual_visualizations(
        test_images=test_images_list, raw_maps=anomaly_maps_raw, norm_scores=normalized_scores,
        img_scores=image_level_scores, save_dir=os.path.join(class_save_dir, "visualizations"),
        test_filepaths=data_manager.test.image_filepaths
    )
    # *** NEW: Call the new plotting function ***
    plot_patch_score_distributions(class_save_dir, ground_truth_labels, anomaly_maps_raw)
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

    logging.info(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1_score:.4f}")
    
    # --- NEW: Append new metrics to the results.txt file ---
    with open(os.path.join(class_save_dir, "results.txt"), "a") as f:
        f.write(f"Optimal Threshold: {optimal_threshold:.4f}\n")
        f.write(f"True Negatives: {tn}\n")
        f.write(f"False Positives: {fp}\n")
        f.write(f"False Negatives: {fn}\n")
        f.write(f"True Positives: {tp}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1_score:.4f}\n")

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
    }

# ==========================================================================================
# SCRIPT ENTRYPOINT
# ==========================================================================================

def main():
    parser = argparse.ArgumentParser(description="PaDiM Anomaly Detection Experiment Runner")
    parser.add_argument('--model_architecture', type=str, default="wide_resnet50_2", choices=["wide_resnet50_2", "resnet18"])
    parser.add_argument('--data_path', type=str, default="../anomaly_detection/data/BowTie-New/original", help="Root path to the dataset.")
    parser.add_argument('--base_results_dir', type=str, default="./results", help="Directory to save all experiment results.")
    parser.add_argument('--resize', type=int, default=512)
    parser.add_argument('--cropsize', type=int, default=400)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--save_distribution', action='store_true', help="Save the learned distribution model to a .pkl file.")
    parser.add_argument('--batch_size', type=int, default=32, help="Set the batch size for training and testing.")
    parser.add_argument(
        "--results_subdir",
        type=str,
        default="",
        help="Optional subfolder inside the base results directory to save results (e.g. 'custom_folder'). If empty, saves directly in base_results_dir.)",
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
        args.master_save_dir = os.path.join(
            args.base_results_dir, args.results_subdir
        )
    else:
        args.master_save_dir = os.path.join(args.base_results_dir, experiment_name)
    os.makedirs(args.master_save_dir, exist_ok=True)

    setup_logging(args.master_save_dir, "experiment_log")
    logging.info(f"--- Starting Experiment: {experiment_name} ---")
    logging.info(f"Running on device: {device}")
    logging.info(f"Script arguments: {vars(args)}")

    # --- 2. Model & Feature Selection ---
    # (Implementation is the same as before, code omitted for brevity)
    if args.model_architecture == "wide_resnet50_2":
        model = wide_resnet50_2(pretrained=True, progress=True)
        total_dim, reduced_dim = 1792, 550
    else: # resnet18
        model = resnet18(pretrained=True, progress=True)
        total_dim, reduced_dim = 448, 100
    
    model.to(device)
    model.eval()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    random_feature_indices = torch.tensor(random.sample(range(0, total_dim), reduced_dim))
    
    model.layer1[-1].register_forward_hook(hook_function)
    model.layer2[-1].register_forward_hook(hook_function)
    model.layer3[-1].register_forward_hook(hook_function)

    # --- 3. Loop Through All Classes ---
    all_class_results = []
    for class_name in bowtie.CLASS_NAMES:
        try:
            class_summary = run_class_processing(args, class_name, model, device, random_feature_indices)
            all_class_results.append(class_summary)
        except Exception as e:
            logging.error(f"!!! FAILED to process class {class_name}. Error: {e}", exc_info=True)

    # --- 4. Save Master Results ---
    # (Implementation is the same as before, code omitted for brevity)
    if not all_class_results:
        logging.warning("No classes were processed successfully. Master results CSV will not be generated.")
        return

    master_df = pd.DataFrame(all_class_results)
    csv_path = os.path.join(args.master_save_dir, "master_results.csv")
    master_df.to_csv(csv_path, index=False)
    
    logging.info(f"--- Experiment Complete ---")
    logging.info(f"Master results saved to: {csv_path}")
    logging.info("\n" + master_df.to_string())

if __name__ == "__main__":
    main()