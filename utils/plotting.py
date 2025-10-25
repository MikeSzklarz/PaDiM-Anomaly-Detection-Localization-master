import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as sps
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay

from .misc import denormalize_image_for_display


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
    """Create & save ROC, score distribution and confusion matrix visuals."""
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

    # Score distribution
    plt.figure(figsize=(10, 6))
    normal_scores = img_scores[gt_labels == 0]
    abnormal_scores = img_scores[gt_labels == 1]

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

        try:
            if len(normal_scores) > 0:
                med0 = np.median(normal_scores)
                plt.axvline(med0, color=palette[0], linestyle="--", linewidth=1.5)
            if len(abnormal_scores) > 0:
                med1 = np.median(abnormal_scores)
                plt.axvline(med1, color=palette[1], linestyle="--", linewidth=1.5)
        except Exception:
            pass

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
        plt.hist([], bins=50)

    plt.xlabel("Image-Level Anomaly Score")
    plt.ylabel("Count")
    plt.title("Distribution of Scores for Normal vs. Abnormal Images")
    plt.xlim(0, 1)
    plt.savefig(os.path.join(class_save_dir, "score_distribution.png"))
    plt.close()

    gmeans = np.sqrt(tpr * (1 - fpr))
    best_gmean_index = np.argmax(gmeans)
    try:
        thresholds = roc_curve(gt_labels, img_scores)[2]
    except Exception:
        thresholds = np.linspace(0, 1, num=len(gmeans))

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


def plot_mean_anomaly_maps(
    class_save_dir,
    gt_labels,
    anomaly_maps_raw,
    score_maps,
    img_scores=None,
    optimal_threshold=None,
):
    """Saves a 2x4 figure comparing mean anomaly maps for TN/FP/FN/TP.

    Layout:
      - Top row: raw anomaly maps (TN | FP | FN | TP)
      - Bottom row: upscaled / score maps (TN | FP | FN | TP)

    Args:
        class_save_dir: directory to save the image
        gt_labels: array-like of ground-truth labels (0=normal,1=anomalous)
        anomaly_maps_raw: array-like of raw patch maps per image
        score_maps: array-like of upscaled score maps per image
        img_scores: optional array-like of image-level scores (required to determine TP/FP/FN/TN)
        optimal_threshold: optional threshold for image-level classification; if None defaults to 0.5
    """
    logging.info("Generating mean anomaly maps for TN/FP/FN/TP...")

    if img_scores is None:
        logging.warning(
            "plot_mean_anomaly_maps called without img_scores; defaulting to grouping by gt only (will show TN and TP columns only)."
        )

    # Determine threshold to compute image-level predictions
    if optimal_threshold is None:
        used_threshold = 0.5
        logging.info(
            "No optimal_threshold provided; using default threshold=0.5 for image-level classification."
        )
    else:
        used_threshold = optimal_threshold

    # Compute predicted labels if image scores are present
    if img_scores is not None:
        preds = (np.asarray(img_scores) >= used_threshold).astype(int)
    else:
        preds = None

    # Prepare containers for the four outcome types in order [TN, FP, FN, TP]
    outcome_names = ["TN", "FP", "FN", "TP"]
    mean_raw = {k: np.zeros_like(anomaly_maps_raw[0]) for k in outcome_names}
    mean_up = {k: np.zeros_like(score_maps[0]) for k in outcome_names}

    # Determine indices for each outcome
    gt = np.asarray(gt_labels)
    n = len(gt)

    # Helper to safely compute mean if indices non-empty
    def _mean_for(indices, arr):
        if len(indices) == 0:
            return np.zeros_like(arr[0])
        return np.mean(arr[indices], axis=0)

    # TN: gt=0 & pred=0
    if preds is not None:
        tn_idx = np.where((gt == 0) & (preds == 0))[0]
        fp_idx = np.where((gt == 0) & (preds == 1))[0]
        fn_idx = np.where((gt == 1) & (preds == 0))[0]
        tp_idx = np.where((gt == 1) & (preds == 1))[0]
    else:
        # If preds not available, group by gt only: treat all normal as TN and anomalous as TP
        tn_idx = np.where(gt == 0)[0]
        fp_idx = np.array([], dtype=int)
        fn_idx = np.array([], dtype=int)
        tp_idx = np.where(gt == 1)[0]

    idx_map = {"TN": tn_idx, "FP": fp_idx, "FN": fn_idx, "TP": tp_idx}

    for name in outcome_names:
        inds = idx_map[name]
        mean_raw[name] = _mean_for(inds, anomaly_maps_raw)
        mean_up[name] = _mean_for(inds, score_maps)

    # Create 2x4 figure: columns are TN,FP,FN,TP; top row raw, bottom row upscaled
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Mean Anomaly Maps (top: raw, bottom: upscaled)", fontsize=18)

    for col, name in enumerate(outcome_names):
        inds = idx_map[name]
        count = len(inds)

        ax_raw = axes[0, col]
        im_raw = ax_raw.imshow(mean_raw[name], cmap="jet")
        ax_raw.set_title(f"{name} (n={count})")
        fig.colorbar(im_raw, ax=ax_raw, fraction=0.046, pad=0.02)

        ax_up = axes[1, col]
        im_up = ax_up.imshow(mean_up[name], cmap="jet")
        ax_up.set_title(f"{name} (n={count})")
        fig.colorbar(im_up, ax=ax_up, fraction=0.046, pad=0.02)

        # Hide axis ticks for readability
        ax_raw.axes.xaxis.set_visible(False)
        ax_raw.axes.yaxis.set_visible(False)
        ax_up.axes.xaxis.set_visible(False)
        ax_up.axes.yaxis.set_visible(False)

    save_path = os.path.join(class_save_dir, "mean_anomaly_maps_by_outcome.png")
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
    """Generate 4-panel visualizations per test image and save under `save_dir`.

    Files will be saved in a two-level folder structure:
      save_dir/<normal|anomalous>/<TP|TN|FP|FN>/

    Uses `denormalize_image_for_display` from `utils.misc` for image rendering.
    """
    logging.info(f"Generating {len(test_images)} individual visualizations...")

    # If user didn't provide an optimal threshold, default to 0.5 and warn.
    if optimal_threshold is None:
        logging.warning(
            "No optimal_threshold provided to plot_individual_visualizations; defaulting to 0.5 for classification (TP/TN/FP/FN)."
        )
        used_threshold = 0.5
    else:
        used_threshold = optimal_threshold

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

        # Map dataset folder label to binary ground truth: 'good' -> 0, else 1
        gt_label = 0 if defect_type == "good" else 1
        pred_label = 1 if score >= used_threshold else 0

        if gt_label == 0 and pred_label == 0:
            class_tag = "TN"
        elif gt_label == 0 and pred_label == 1:
            class_tag = "FP"
        elif gt_label == 1 and pred_label == 0:
            class_tag = "FN"
        else:
            class_tag = "TP"

        subfolder = "normal" if defect_type == "good" else "anomalous"
        target_dir = os.path.join(save_dir, subfolder, class_tag)
        os.makedirs(target_dir, exist_ok=True)

        save_filename = f"{defect_type}_{image_filename}_{class_tag}.png"

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(img_denormalized)
        axes[0].set_title("1. Original Image")
        axes[1].hist(raw_map.ravel(), bins=60, color="skyblue", edgecolor="black")
        axes[1].set_title("2. Patch Score Distribution")
        axes[1].set_xlabel("Mahalanobis Distance")
        axes[1].set_ylabel("Frequency")
        axes[1].grid(axis="y", alpha=0.6)

        im = axes[2].imshow(raw_map, cmap="jet")
        axes[2].set_title("3. Raw Anomaly Map")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        axes[3].imshow(img_denormalized)
        axes[3].imshow(final_heatmap, cmap="jet", alpha=0.5)
        axes[3].set_title("4. Final Heatmap Overlay")
        stats_lines = [f"Score: {score:.3f}"]
        if per_image_stats is not None:
            s = per_image_stats[i]
            stats_lines.append(f"Max: {s['max']:.3f}")
            stats_lines.append(f"MeanTop1%: {s['mean_top1pct']:.3f}")
            stats_lines.append(f"P95: {s['p95']:.3f}")
            stats_lines.append(f"Frac>{s['threshold']:.3f}: {s['frac_above']*100:.1f}%")
        if optimal_threshold is not None:
            pred_label_text = "Anomalous" if score >= optimal_threshold else "Normal"
            stats_lines.append(f"Pred: {pred_label_text}")
        else:
            # If we defaulted, note the used threshold and predicted label
            pred_label_text = "Anomalous" if pred_label == 1 else "Normal"
            stats_lines.append(f"Pred (th=0.5): {pred_label_text}")

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
    """Create patch-level histograms for normal, abnormal and their overlay."""
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


def compute_and_save_blob_analysis(
    class_save_dir,
    analysis_results,
    test_filepaths,
    gt_labels,
    img_scores=None,
    optimal_threshold=None,
    num_quadrants=10,
):
    """Flatten per-image blob analysis and save CSVs.

    Produces two CSVs in `class_save_dir`:
      - blob_analysis.csv : one row per blob with detailed attributes
      - blob_image_summary.csv : one row per image summarizing blob counts & stats

    Parameters
    ----------
    class_save_dir : str
        Directory to save CSV files.
    analysis_results : list
        List of per-image dicts matching the notebook structure. Each dict is
        expected to have keys like 'image_filename', 'blobs' (list of blob dicts),
        and optionally 'image_gt_label'. Blob dicts should contain area, bbox,
        mean_intensity, max_intensity, geometric_centroid, weighted_centroid, distance_from_center, quadrant, etc.
    test_filepaths : list[str]
        Original filepaths for the test images (used to infer folder/class names if missing from analysis_results).
    gt_labels : array-like
        Ground-truth labels per image (0=normal,1=anomalous).
    img_scores : array-like, optional
        Image-level anomaly scores used to compute TP/TN/FP/FN when combined with optimal_threshold.
    optimal_threshold : float, optional
        Threshold to turn image scores into binary predictions. If None and img_scores provided,
        defaults to 0.5.
    num_quadrants : int
        Number of radial quadrants used when quadrant info isn't already present.

    Returns
    -------
    tuple(pd.DataFrame, pd.DataFrame)
        (blob_df, image_summary_df) dataframes that were written to disk.
    """
    os.makedirs(class_save_dir, exist_ok=True)

    # Determine predictions per-image if scores provided
    preds = None
    if img_scores is not None:
        if optimal_threshold is None:
            used_threshold = 0.5
        else:
            used_threshold = optimal_threshold
        preds = (np.asarray(img_scores) >= used_threshold).astype(int)

    # Build flattened blob rows
    blob_rows = []
    image_summaries = []

    # Ensure gt_labels is numpy array
    gt_arr = np.asarray(gt_labels)

    for idx, img_summary in enumerate(analysis_results):
        # Provide fallbacks if notebook-style fields are missing
        if isinstance(img_summary, dict):
            image_filename = img_summary.get("image_filename")
            blobs = img_summary.get("blobs", [])
            img_gt = img_summary.get("image_gt_label")
        else:
            # If user passed a simple list of blobs per image, adapt
            image_filename = None
            blobs = img_summary
            img_gt = None

        if image_filename is None:
            # fallback to test_filepaths list
            try:
                image_filename = os.path.basename(test_filepaths[idx])
            except Exception:
                image_filename = f"img_{idx}"

        # infer gt_label
        if img_gt is None:
            try:
                img_gt_val = int(gt_arr[idx])
            except Exception:
                img_gt_val = None
        else:
            img_gt_val = int(img_gt)

        pred_val = int(preds[idx]) if preds is not None else None

        # determine image-level outcome tag for all blobs in this image
        if img_gt_val is None or pred_val is None:
            image_outcome = None
        else:
            if img_gt_val == 0 and pred_val == 0:
                image_outcome = "TN"
            elif img_gt_val == 0 and pred_val == 1:
                image_outcome = "FP"
            elif img_gt_val == 1 and pred_val == 0:
                image_outcome = "FN"
            else:
                image_outcome = "TP"

        # Per-image aggregates used for image_summary_df
        image_blob_count = 0
        sum_blob_area = 0.0

        for blob in blobs:
            image_blob_count += 1
            # Accept blob as dict and gracefully handle missing keys
            row = {
                "image_idx": idx,
                "image_filename": image_filename,
                "image_gt_label": img_gt_val,
                "image_pred_label": pred_val,
                "image_outcome": image_outcome,
            }
            # Copy known blob keys if present, else fill with None
            row["blob_label"] = (
                blob.get("blob_label") if isinstance(blob, dict) else None
            )
            row["area_pixels"] = (
                blob.get("area_pixels") if isinstance(blob, dict) else None
            )
            row["bbox_x"] = (
                blob.get("bounding_box", {}).get("x")
                if isinstance(blob, dict)
                else None
            )
            row["bbox_y"] = (
                blob.get("bounding_box", {}).get("y")
                if isinstance(blob, dict)
                else None
            )
            row["bbox_width"] = (
                blob.get("bounding_box", {}).get("width")
                if isinstance(blob, dict)
                else None
            )
            row["bbox_height"] = (
                blob.get("bounding_box", {}).get("height")
                if isinstance(blob, dict)
                else None
            )
            row["mean_intensity"] = (
                blob.get("mean_intensity") if isinstance(blob, dict) else None
            )
            row["max_intensity"] = (
                blob.get("max_intensity") if isinstance(blob, dict) else None
            )
            geom = blob.get("geometric_centroid") if isinstance(blob, dict) else None
            if geom is not None and len(geom) >= 2:
                row["geometric_centroid_x"] = geom[0]
                row["geometric_centroid_y"] = geom[1]
            else:
                row["geometric_centroid_x"] = None
                row["geometric_centroid_y"] = None
            wcent = blob.get("weighted_centroid") if isinstance(blob, dict) else None
            if wcent is not None and len(wcent) >= 2:
                row["weighted_centroid_x"] = wcent[0]
                row["weighted_centroid_y"] = wcent[1]
            else:
                row["weighted_centroid_x"] = None
                row["weighted_centroid_y"] = None
            row["distance_from_center"] = (
                blob.get("distance_from_center") if isinstance(blob, dict) else None
            )
            row["quadrant"] = blob.get("quadrant") if isinstance(blob, dict) else None

            sum_blob_area += (
                float(row["area_pixels"]) if row["area_pixels"] is not None else 0.0
            )

            blob_rows.append(row)

        # Record per-image summary
        image_summaries.append(
            {
                "image_idx": idx,
                "image_filename": image_filename,
                "image_gt_label": img_gt_val,
                "image_pred_label": pred_val,
                "image_outcome": image_outcome,
                "blob_count": image_blob_count,
                "total_blob_area": sum_blob_area,
            }
        )

    # Convert to DataFrames
    blob_df = pd.DataFrame(blob_rows)
    image_summary_df = pd.DataFrame(image_summaries)

    # Ensure we have an image filename and an image_name (basename without extension)
    def _ensure_image_name(row, idx):
        fname = row.get("image_filename") if isinstance(row, dict) else None
        if not fname or fname is None or fname == "":
            try:
                fname = test_filepaths[idx]
            except Exception:
                fname = f"img_{idx}"
        image_name = os.path.splitext(os.path.basename(fname))[0]
        return fname, image_name

    # Add image_filename and image_name to the image_summary_df (fill from provided or test_filepaths)
    image_fnames = []
    image_names = []
    for i, rec in enumerate(image_summaries):
        fname, iname = _ensure_image_name(rec, i)
        image_fnames.append(fname)
        image_names.append(iname)
    image_summary_df["image_filename"] = image_fnames
    image_summary_df["image_name"] = image_names

    # If image-level scores are provided, attach them and predicted label/outcome were already set
    if img_scores is not None:
        image_summary_df["image_score"] = np.asarray(img_scores)[
            : len(image_summary_df)
        ]

    # If blob_df is empty, create an empty grouped structure; otherwise aggregate per-image stats
    if not blob_df.empty:
        # normalize numeric columns
        numeric_cols = [
            "area_pixels",
            "mean_intensity",
            "max_intensity",
            "weighted_centroid_x",
            "weighted_centroid_y",
            "distance_from_center",
        ]
        for c in numeric_cols:
            if c in blob_df.columns:
                blob_df[c] = pd.to_numeric(blob_df[c], errors="coerce")

        grouped = (
            blob_df.groupby("image_idx")
            .agg(
                blob_count=("blob_label", "count"),
                total_blob_area=("area_pixels", "sum"),
                mean_blob_area=("area_pixels", "mean"),
                max_blob_area=("area_pixels", "max"),
                mean_of_mean_intensity=("mean_intensity", "mean"),
                max_intensity=("max_intensity", "max"),
                mean_weighted_centroid_x=("weighted_centroid_x", "mean"),
                mean_weighted_centroid_y=("weighted_centroid_y", "mean"),
                mean_distance_from_center=("distance_from_center", "mean"),
            )
            .reset_index()
        )

        # quadrant counts per image (if quadrant column present)
        if "quadrant" in blob_df.columns:
            try:
                quad_counts = (
                    blob_df.groupby(["image_idx", "quadrant"])
                    .size()
                    .unstack(fill_value=0)
                )
                quad_counts = quad_counts.add_prefix("quadrant_")
                quad_counts = quad_counts.reset_index()
                grouped = grouped.merge(quad_counts, on="image_idx", how="left")
            except Exception:
                logging.exception("Failed to compute quadrant counts per image")

    else:
        # No blobs detected at all; create an empty grouped dataframe with image_idx covering all images
        grouped = pd.DataFrame({"image_idx": image_summary_df["image_idx"].tolist()})

    # Merge aggregated blob stats into the image summary frame
    merged = image_summary_df.merge(grouped, on="image_idx", how="left")

    # Fill missing numeric aggregate values sensibly
    fill_zero_cols = [
        "blob_count",
        "total_blob_area",
        "mean_blob_area",
        "max_blob_area",
        "mean_of_mean_intensity",
        "max_intensity",
        "mean_weighted_centroid_x",
        "mean_weighted_centroid_y",
        "mean_distance_from_center",
    ]
    for c in fill_zero_cols:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0)

    # Reorder columns for readability
    image_summary_cols = [
        "image_idx",
        "image_name",
        "image_filename",
        "image_gt_label",
        "image_score",
        "image_pred_label",
        "image_outcome",
        "blob_count",
        "total_blob_area",
        "mean_blob_area",
        "max_blob_area",
        "mean_of_mean_intensity",
        "max_intensity",
        "mean_weighted_centroid_x",
        "mean_weighted_centroid_y",
        "mean_distance_from_center",
    ]
    # include any quadrant columns if present
    quad_cols = [c for c in merged.columns if c.startswith("quadrant_")]
    image_summary_cols.extend(quad_cols)

    # Ensure all selected columns exist in merged, otherwise skip missing ones
    image_summary_cols = [c for c in image_summary_cols if c in merged.columns]

    final_image_summary_df = merged[image_summary_cols]

    blob_csv_path = os.path.join(class_save_dir, "blob_analysis.csv")
    image_summary_path = os.path.join(class_save_dir, "blob_image_summary.csv")

    try:
        blob_df.to_csv(blob_csv_path, index=False)
        logging.info(f"Saved blob analysis CSV to: {blob_csv_path}")
    except Exception:
        logging.exception(f"Failed to write blob analysis CSV to: {blob_csv_path}")

    try:
        final_image_summary_df.to_csv(image_summary_path, index=False)
        logging.info(f"Saved blob image summary CSV to: {image_summary_path}")
    except Exception:
        logging.exception(f"Failed to write image summary CSV to: {image_summary_path}")

    return blob_df, final_image_summary_df
