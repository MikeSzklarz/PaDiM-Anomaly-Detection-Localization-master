import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
from torch.amp import autocast


def get_batch_embeddings(
    batch, hooks, random_feature_indices, model, main_device, target_device
):
    with torch.no_grad(), autocast(device_type=main_device.type):
        _ = model(batch.to(main_device))

    embedding = hooks[0]
    for i in range(1, len(hooks)):
        embedding = concatenate_embeddings(embedding, hooks[i])

    embedding = torch.index_select(embedding, 1, random_feature_indices.to(main_device))
    hooks.clear()
    return embedding.to(target_device)


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
    # --- Score distribution: overlayed density histplot with KDE and median lines ---
    plt.figure(figsize=(10, 6))
    normal_scores = img_scores[gt_labels == 0]
    abnormal_scores = img_scores[gt_labels == 1]

    # Build DataFrame for seaborn hue plotting (safe when one group may be empty)
    scores_list = []
    labels_list = []
    if len(normal_scores) > 0:
        scores_list.append(np.asarray(normal_scores))
        labels_list.append(np.zeros(len(normal_scores), dtype=int))
    if len(abnormal_scores) > 0:
        scores_list.append(np.asarray(abnormal_scores))
        labels_list.append(np.ones(len(abnormal_scores), dtype=int))

    if scores_list:
        scores_combined = np.concatenate(scores_list)
        labels_combined = np.concatenate(labels_list)
        score_df = pd.DataFrame({"score": scores_combined, "label": labels_combined})

        palette = {0: "C0", 1: "C3"}
        sns.histplot(
            data=score_df,
            x="score",
            hue="label",
            bins=50,
            kde=True,
            element="step",
            stat="count",
            common_norm=False,
            palette=palette,
        )

        # dashed median lines for each group
        try:
            if len(normal_scores) > 0:
                med0 = np.median(normal_scores)
                plt.axvline(med0, color=palette[0], linestyle="--", linewidth=1.5)
            if len(abnormal_scores) > 0:
                med1 = np.median(abnormal_scores)
                plt.axvline(med1, color=palette[1], linestyle="--", linewidth=1.5)
        except Exception:
            pass

        # friendly legend
        handles, labels = plt.gca().get_legend_handles_labels()
        if labels:
            mapped = [
                (
                    "Normal"
                    if l in ["0", "0.0"]
                    else "Anomalous" if l in ["1", "1.0"] else l
                )
                for l in labels
            ]
            plt.legend(handles, mapped)

    else:
        # fallback: empty groups -- draw empty histogram axes
        plt.hist([], bins=50)

    plt.xlabel("Image-Level Anomaly Score")
    plt.ylabel("Count")
    plt.title("Distribution of Scores for Normal vs. Abnormal Images")
    plt.xlim(0, 1)
    plt.savefig(os.path.join(class_save_dir, "score_distribution.png"))
    plt.close()

    gmeans = np.sqrt(tpr * (1 - fpr))
    best_gmean_index = np.argmax(gmeans)
    thresholds = (
        np.array(plt.mlab.transpose(plt.mlab.griddata([0], [0], [0])))
        if False
        else None
    )
    # fallback: recompute thresholds from roc_curve
    try:
        from sklearn.metrics import roc_curve

        thresholds = roc_curve(gt_labels, img_scores)[2]
    except Exception:
        thresholds = np.linspace(0, 1, num=len(gmeans))

    optimal_threshold = thresholds[best_gmean_index]
    predictions = (img_scores >= optimal_threshold).astype(int)
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
    im1 = axes[0, 0].imshow(mean_raw_normal, cmap="jet")
    axes[0, 0].set_title("Mean Raw Map (Normal Images)")
    fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    im2 = axes[0, 1].imshow(mean_raw_anomalous, cmap="jet")
    axes[0, 1].set_title("Mean Raw Map (Anomalous Images)")
    fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
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
    global_raw_min = np.min(raw_maps)
    global_raw_max = np.max(raw_maps)
    global_norm_min = np.min(norm_scores)
    global_norm_max = np.max(norm_scores)

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

        subfolder = "normal" if defect_type == "good" else "anomalous"
        target_dir = os.path.join(save_dir, subfolder)
        os.makedirs(target_dir, exist_ok=True)

        save_filename = f"{defect_type}_{image_filename}.png"

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(img_denormalized)
        axes[0].set_title("1. Original Image")
        axes[1].hist(raw_map.ravel(), bins=60, color="skyblue", edgecolor="black")
        axes[1].set_title("2. Patch Score Distribution")
        axes[1].set_xlabel("Mahalanobis Distance")
        axes[1].set_ylabel("Frequency")
        axes[1].grid(axis="y", alpha=0.6)
        im = axes[2].imshow(
            raw_map, cmap="jet", vmin=global_raw_min, vmax=global_raw_max
        )
        axes[2].set_title("3. Raw Anomaly Map")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        axes[3].imshow(img_denormalized)
        axes[3].imshow(
            final_heatmap,
            cmap="jet",
            alpha=0.5,
            vmin=global_norm_min,
            vmax=global_norm_max,
        )
        axes[3].set_title("4. Final Heatmap Overlay")
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
    if len(normal_indices) > 0:
        all_normal_patch_scores = anomaly_maps_raw[normal_indices].flatten()
    if len(abnormal_indices) > 0:
        all_abnormal_patch_scores = anomaly_maps_raw[abnormal_indices].flatten()

    combined_scores = []
    if all_normal_patch_scores is not None:
        combined_scores.append(all_normal_patch_scores)
    if all_abnormal_patch_scores is not None:
        combined_scores.append(all_abnormal_patch_scores)

    x_max = np.max(np.concatenate(combined_scores)) * 1.05 if combined_scores else 1.0

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
        plt.xlim(0, x_max)
        plt.savefig(os.path.join(class_save_dir, "patch_score_distribution_normal.png"))
        plt.close(fig)

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
        plt.xlim(0, x_max)
        plt.savefig(
            os.path.join(class_save_dir, "patch_score_distribution_abnormal.png")
        )
        plt.close(fig)

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
        plt.xlim(0, x_max)
        plt.savefig(
            os.path.join(class_save_dir, "patch_score_distribution_comparative.png")
        )
        plt.close(fig)
