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
import sklearn
from torch.amp import autocast
from sklearn.metrics import average_precision_score
import scipy.stats as sps

# --- Visualization ---
import matplotlib.pyplot as plt

# --- Custom Dataloader ---
# Assumes bowtie.py is in a 'datasets' subfolder
import datasets.bowtie as bowtie
import copy

# Bring commonly used helpers from smaller utils modules to keep this script slim.
from utils.embeddings import get_batch_embeddings, concatenate_embeddings
from utils.misc import denormalize_image_for_display
from utils.plotting import (
    plot_summary_visuals,
    plot_mean_anomaly_maps,
    plot_individual_visualizations,
    plot_patch_score_distributions,
    save_feature_importance,
    plot_roc_pr,
    save_confusion_matrix,
    tsne_plot,
    explain_with_shap,
)

import trainer
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from utils.misc import feature_factory_from_blob_df
from sklearn.model_selection import train_test_split

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


def run_class_processing(
    args, class_name, model, device, random_feature_indices, hooks=None
):
    """Compatibility shim that delegates to trainer.run_class_processing.

    Args:
        args: argparse Namespace or config object
        class_name: str
        model: torch.nn.Module
        device: torch.device
        random_feature_indices: torch.Tensor
        hooks: optional list to receive intermediate features (falls back to INTERMEDIATE_FEATURE_MAPS)

    Returns:
        Whatever trainer.run_class_processing returns.
    """
    hooks_to_use = hooks if hooks is not None else INTERMEDIATE_FEATURE_MAPS
    return trainer.run_class_processing(
        args, class_name, model, device, random_feature_indices, hooks_to_use
    )


def main():
    parser = argparse.ArgumentParser(
        description="PaDiM Anomaly Detection Experiment Runner"
    )
    parser.add_argument("--model_architecture", type=str, default="wide_resnet50_2")
    parser.add_argument(
        "--data_path",
        type=str,
        default="../anomaly_detection/data/BowTie-New/original",
        help="Root path to the dataset.",
    )
    parser.add_argument("--base_results_dir", type=str, default="./results")
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--cropsize", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--save_distribution", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--augmentations_enabled", action="store_true")
    parser.add_argument("--horizontal_flip", action="store_true")
    parser.add_argument("--vertical_flip", action="store_true")
    parser.add_argument("--augmentation_prob", type=float, default=0.5)
    parser.add_argument("--results_subdir", type=str, default="")
    parser.add_argument("--test_class_name", type=str, default=None)
    parser.add_argument("--test_class_path", type=str, default=None)
    parser.add_argument("--mahalanobis_on_gpu", action="store_true")
    parser.add_argument("--stats_on_gpu", action="store_true")
    parser.add_argument("--test_sample_ratio", type=float, default=0.2)
    parser.add_argument(
        "--use_blob_classifier",
        action="store_true",
        help="Train and use a RandomForest on aggregated blob features to override PaDiM predictions",
    )
    parser.add_argument(
        "--retrain_blob_classifier",
        action="store_true",
        help="Force retraining of the blob classifier even if saved artifacts exist in the class folder",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_name = (
        f"{args.model_architecture}_resize-{args.resize}_crop-{args.cropsize}"
    )
    if args.results_subdir:
        args.master_save_dir = os.path.join(args.base_results_dir, args.results_subdir)
    else:
        args.master_save_dir = os.path.join(args.base_results_dir, experiment_name)
    os.makedirs(args.master_save_dir, exist_ok=True)

    setup_logging(args.master_save_dir, "experiment_log")
    logging.info(f"--- Starting Experiment: {experiment_name} ---")

    # Model selection
    if args.model_architecture == "wide_resnet50_2":
        model = wide_resnet50_2(weights="DEFAULT")
        total_dim, reduced_dim = 1792, 550
    elif args.model_architecture == "resnet18":
        model = resnet18(weights="DEFAULT")
        total_dim, reduced_dim = 448, 100
    elif args.model_architecture == "efficientnet_b5":
        model = efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
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

    # Loop through classes and call trainer
    all_class_results = []
    discovered_classes = bowtie.get_class_names(args.data_path)

    is_direct_class_folder = os.path.isdir(
        os.path.join(args.data_path, "train")
    ) and os.path.isdir(os.path.join(args.data_path, "test"))

    if is_direct_class_folder:
        class_name = os.path.basename(os.path.abspath(args.data_path))
        logging.info(
            f"Data path appears to be a single class folder. Running for class: {class_name}"
        )
        args_copy = copy.copy(args)
        args_copy.data_path = os.path.dirname(os.path.abspath(args.data_path))
        try:
            class_summary = trainer.run_class_processing(
                args_copy,
                class_name,
                model,
                device,
                random_feature_indices,
                INTERMEDIATE_FEATURE_MAPS,
            )
            all_class_results.append(class_summary)

            # Optional: Train/use blob-level RandomForest to override PaDiM predictions
            if getattr(args, "use_blob_classifier", False):
                try:
                    # Determine class save dir the same way trainer does
                    if (
                        getattr(args_copy, "test_class_path", None)
                        or getattr(args_copy, "test_class_name", None) is not None
                    ):
                        if getattr(args_copy, "test_class_path", None):
                            test_abs = os.path.abspath(args_copy.test_class_path)
                            test_class_name = os.path.basename(test_abs)
                        else:
                            test_class_name = args_copy.test_class_name
                        class_save_dir = os.path.join(
                            args_copy.master_save_dir,
                            f"train_{class_name}_test_{test_class_name}",
                        )
                    else:
                        class_save_dir = os.path.join(
                            args_copy.master_save_dir, class_name
                        )

                    blob_csv = os.path.join(class_save_dir, "blob_analysis.csv")
                    model_path = os.path.join(
                        class_save_dir, "blob_classifier_model.joblib"
                    )
                    scaler_path = os.path.join(
                        class_save_dir, "blob_classifier_scaler.joblib"
                    )
                    feat_path = os.path.join(
                        class_save_dir, "blob_classifier_features.json"
                    )

                    # Preferred: check for a training blob CSV exported for the training split
                    train_blob_csv = os.path.join(
                        class_save_dir, "blob_analysis_train.csv"
                    )
                    test_blob_csv = os.path.join(class_save_dir, "blob_analysis.csv")

                    model_exists = (
                        os.path.exists(model_path)
                        and os.path.exists(scaler_path)
                        and os.path.exists(feat_path)
                    )

                    clf = None
                    scaler = None
                    saved_feats = None
                    # When True we evaluated on an internal holdout from the test CSV
                    eval_on_holdout = False
                    # When True we evaluated on an internal holdout from the test CSV
                    eval_on_holdout = False

                    # Train on dedicated training blob CSV if available
                    if os.path.exists(train_blob_csv):
                        logging.info(
                            f"Training blob classifier from training CSV: {train_blob_csv}"
                        )
                        # Version tag for logging: training from dedicated train CSV
                        logging.info("Blob classifier mode: TRAIN_FROM_TRAIN_CSV")
                        X_train_df, y_train, feat_names_train = (
                            feature_factory_from_blob_df(train_blob_csv)
                        )
                        if y_train is None or (len(np.unique(y_train)) < 2):
                            logging.warning(
                                f"Insufficient labels in {train_blob_csv} to train blob classifier for {class_name}."
                            )
                        else:
                            logging.info(
                                "Blob classifier training: starting fit on training CSV."
                            )
                            try:
                                logging.info(
                                    f"Training CSV rows: {len(X_train_df)}, features: {len(X_train_df.columns)}"
                                )
                                # class distribution
                                if hasattr(y_train, "value_counts"):
                                    logging.info(
                                        f"Training class distribution: {y_train.value_counts().to_dict()}"
                                    )
                                else:
                                    logging.info(
                                        f"Training classes (unique counts): {np.unique(y_train, return_counts=True)}"
                                    )
                            except Exception:
                                logging.exception("Failed to log training CSV summary")
                            X_train_vals = X_train_df.fillna(0).values
                            scaler = StandardScaler().fit(X_train_vals)
                            clf = RandomForestClassifier(
                                n_estimators=200, random_state=args.seed
                            )
                            clf.fit(scaler.transform(X_train_vals), y_train.values)
                            # Log model metadata
                            try:
                                logging.info(
                                    f"Trained RandomForest (sklearn {sklearn.__version__}) with params: {clf.get_params()}"
                                )
                            except Exception:
                                logging.exception("Failed to log classifier params")
                            os.makedirs(class_save_dir, exist_ok=True)
                            joblib.dump(clf, model_path)
                            joblib.dump(scaler, scaler_path)
                            with open(feat_path, "w") as fh:
                                json.dump(list(X_train_df.columns), fh)
                            saved_feats = list(X_train_df.columns)
                            logging.info(
                                f"Saved blob classifier artifacts to: model={model_path}, scaler={scaler_path}, feats={feat_path}"
                            )

                    # If we still don't have a trained model, consider loading existing ones
                    if (
                        not clf
                        and model_exists
                        and not getattr(args, "retrain_blob_classifier", False)
                    ):
                        # Version tag for logging: loading existing trained artifacts
                        logging.info("Blob classifier mode: LOAD_EXISTING_ARTIFACTS")
                        try:
                            clf = joblib.load(model_path)
                            scaler = joblib.load(scaler_path)
                            with open(feat_path, "r") as fh:
                                saved_feats = json.load(fh)
                            try:
                                logging.info(
                                    f"Loaded existing blob classifier artifacts. model={model_path}, scaler={scaler_path}, feats={feat_path}"
                                )
                                if hasattr(clf, "get_params"):
                                    logging.info(
                                        f"Loaded RandomForest (sklearn {sklearn.__version__}) params: {clf.get_params()}"
                                    )
                            except Exception:
                                logging.exception("Failed to log loaded classifier metadata")
                        except Exception:
                            logging.exception(
                                "Failed to load existing blob classifier artifacts; will attempt to (re)train if possible."
                            )

                    # If no train CSV, fallback: split the available test blob CSV into train/holdout
                    if (clf is None) and os.path.exists(test_blob_csv):
                        # Version tag for logging: fallback internal split from test blob CSV
                        logging.info("Blob classifier mode: FALLBACK_SPLIT_FROM_TEST_CSV")
                        df_blob = pd.read_csv(test_blob_csv)
                        X_all, y_all, feat_names_all = feature_factory_from_blob_df(
                            df_blob
                        )
                        if (
                            y_all is None
                            or (len(np.unique(y_all)) < 2)
                            or (len(X_all) < 10)
                        ):
                            logging.warning(
                                f"Insufficient or single-class data in {test_blob_csv}; skipping blob classifier for {class_name}."
                            )
                        else:
                            # safe stratified split
                            try:
                                X_train, X_hold, y_train, y_hold = train_test_split(
                                    X_all,
                                    y_all,
                                    test_size=0.3,
                                    random_state=args.seed,
                                    stratify=y_all,
                                )
                            except Exception:
                                X_train, X_hold, y_train, y_hold = train_test_split(
                                    X_all,
                                    y_all,
                                    test_size=0.3,
                                    random_state=args.seed,
                                    stratify=None,
                                )
                                try:
                                    logging.info(
                                        f"Performed fallback split: total={len(X_all)}, train={len(X_train)}, holdout={len(X_hold)}"
                                    )
                                    if hasattr(y_all, "value_counts"):
                                        logging.info(
                                            f"Overall class distribution: {y_all.value_counts().to_dict()}"
                                        )
                                    if hasattr(y_train, "value_counts"):
                                        logging.info(
                                            f"Train class distribution: {y_train.value_counts().to_dict()}"
                                        )
                                    if hasattr(y_hold, "value_counts"):
                                        logging.info(
                                            f"Holdout class distribution: {y_hold.value_counts().to_dict()}"
                                        )
                                except Exception:
                                    logging.exception("Failed to log split distributions")
                            X_train_vals = X_train.fillna(0).values
                            scaler = StandardScaler().fit(X_train_vals)
                            clf = RandomForestClassifier(
                                n_estimators=200, random_state=args.seed
                            )
                            clf.fit(scaler.transform(X_train_vals), y_train.values)
                            os.makedirs(class_save_dir, exist_ok=True)
                            joblib.dump(clf, model_path)
                            joblib.dump(scaler, scaler_path)
                            saved_feats = list(X_train.columns)
                            # prepare evaluation set
                            X_ordered = X_hold.reindex(columns=saved_feats).fillna(0)
                            y_eval = y_hold
                            X_eval_scaled = scaler.transform(X_ordered.values)
                            eval_preds = clf.predict(X_eval_scaled).astype(int)
                            new_preds = eval_preds
                            X_scaled = X_eval_scaled
                            y = y_eval
                            eval_on_holdout = True

                    # If we have a model and a test CSV, run evaluation on the test CSV
                    if clf is not None and os.path.exists(test_blob_csv) and not eval_on_holdout:
                        try:
                            # load test blobs and evaluate
                            X_test_df, y_test, _ = feature_factory_from_blob_df(
                                test_blob_csv
                            )
                            logging.info("Evaluating on full test CSV because no internal holdout was used.")
                            if saved_feats is None:
                                # try to derive from X_test
                                saved_feats = list(X_test_df.columns)
                            X_ordered = X_test_df.reindex(columns=saved_feats).fillna(0)
                            X_pred = X_ordered.values
                            X_scaled = scaler.transform(X_pred)
                            new_preds = clf.predict(X_scaled).astype(int)
                            y = y_test
                        except Exception:
                            logging.exception(
                                "Failed to evaluate blob classifier on test CSV"
                            )
                    else:
                        if clf is None:
                            logging.info(
                                "No blob classifier trained or available for this class."
                            )
                        elif not os.path.exists(test_blob_csv):
                            logging.warning(
                                f"Test blob CSV not found at {test_blob_csv}; cannot evaluate blob classifier for {class_name}."
                            )

                    # If we obtained predictions and labels, compute metrics and diagnostics
                    if (clf is not None) and (
                        ("new_preds" in locals()) and (y is not None)
                    ):
                        try:
                            # If y is a pandas Series, ensure ordering alignment
                            if hasattr(y, "values"):
                                y_vals = y.values
                            else:
                                y_vals = np.asarray(list(y))
                            tn, fp, fn, tp = confusion_matrix(y_vals, new_preds).ravel()
                        except Exception:
                            tn = fp = fn = tp = 0
                        epsilon = 1e-6
                        precision = tp / (tp + fp + epsilon)
                        recall = tp / (tp + fn + epsilon)
                        f1_score = (
                            2 * (precision * recall) / (precision + recall + epsilon)
                        )
                        class_summary.update(
                            {
                                "precision": round(precision, 4),
                                "recall": round(recall, 4),
                                "f1_score": round(f1_score, 4),
                                "true_negatives": int(tn),
                                "false_positives": int(fp),
                                "false_negatives": int(fn),
                                "true_positives": int(tp),
                            }
                        )
                        logging.info(
                            f"Blob classifier predictions used to override metrics for class {class_name}."
                        )
                        try:
                            logging.info(
                                f"Blob classifier results: precision={precision:.4f}, recall={recall:.4f}, f1={f1_score:.4f}, TN={tn}, FP={fp}, FN={fn}, TP={tp}"
                            )
                        except Exception:
                            logging.exception("Failed to log blob classifier summary stats")

                        # Diagnostics: save preds CSV, report, plots
                        try:
                            out_dir = class_save_dir
                            probs = (
                                clf.predict_proba(X_scaled)[:, 1]
                                if hasattr(clf, "predict_proba")
                                else None
                            )
                            preds_df = pd.DataFrame(
                                {
                                    "image_name": list(X_ordered.index),
                                    "true_label": (
                                        list(y_vals)
                                        if y is not None
                                        else [None] * len(X_ordered)
                                    ),
                                    "pred": list(new_preds),
                                    "prob": (
                                        list(probs)
                                        if probs is not None
                                        else [None] * len(X_ordered)
                                    ),
                                }
                            )
                            preds_df.to_csv(
                                os.path.join(
                                    out_dir, "blob_classifier_predictions.csv"
                                ),
                                index=False,
                            )

                            report = {
                                "used_blob_classifier": True,
                                "n_images": int(X_ordered.shape[0]),
                                "n_features": int(X_ordered.shape[1]),
                            }
                            if hasattr(clf, "feature_importances_"):
                                importances = clf.feature_importances_
                                feat_list = list(X_ordered.columns)
                                top_idx = np.argsort(importances)[::-1][:10]
                                report["top_features"] = [
                                    {
                                        "feature": feat_list[i],
                                        "importance": float(importances[i]),
                                    }
                                    for i in top_idx
                                ]
                            with open(
                                os.path.join(out_dir, "blob_classifier_report.json"),
                                "w",
                            ) as fh:
                                json.dump(report, fh, indent=2)

                            try:
                                if hasattr(clf, "feature_importances_"):
                                    save_feature_importance(
                                        clf.feature_importances_,
                                        list(X_ordered.columns),
                                        out_dir,
                                    )
                            except Exception:
                                logging.exception(
                                    "Failed to save RF feature importances"
                                )

                            try:
                                if (y is not None) and (probs is not None):
                                    plot_roc_pr(y_vals, probs, out_dir)
                            except Exception:
                                logging.exception(
                                    "Failed to plot ROC/PR for blob classifier"
                                )

                            try:
                                if y is not None:
                                    save_confusion_matrix(
                                        y_vals,
                                        new_preds,
                                        out_dir,
                                        labels=["normal", "anomaly"],
                                    )
                            except Exception:
                                logging.exception(
                                    "Failed to save confusion matrix for blob classifier"
                                )

                            try:
                                tsne_plot(
                                    X_ordered,
                                    pd.Series(y_vals) if y is not None else None,
                                    new_preds,
                                    out_dir,
                                    sample_frac=1.0,
                                )
                            except Exception:
                                logging.exception(
                                    "Failed to create t-SNE for blob features"
                                )

                            try:
                                explain_with_shap(
                                    clf,
                                    X_ordered,
                                    out_dir,
                                    list(X_ordered.columns),
                                    y=pd.Series(y_vals) if y is not None else None,
                                )
                            except Exception:
                                logging.exception(
                                    "Failed to compute SHAP explanations for blob classifier"
                                )
                        except Exception:
                            logging.exception(
                                "Failed to write blob classifier diagnostics"
                            )
                except Exception:
                    logging.exception(
                        f"Blob classifier training/eval failed for {class_name}"
                    )
        except Exception as e:
            logging.exception(f"Error processing class {class_name}: {e}")
    else:
        if not discovered_classes:
            logging.error(
                f"No class subfolders found in data path: {args.data_path}. Make sure the dataset root contains one folder per class (each with train/ and test/)."
            )
            return

        for class_name in discovered_classes:
            try:
                class_summary = trainer.run_class_processing(
                    args,
                    class_name,
                    model,
                    device,
                    random_feature_indices,
                    INTERMEDIATE_FEATURE_MAPS,
                )
                all_class_results.append(class_summary)
                # Optional blob-classifier training/eval (override PaDiM predictions)
                if getattr(args, "use_blob_classifier", False):
                    try:
                        if (
                            getattr(args, "test_class_path", None)
                            or getattr(args, "test_class_name", None) is not None
                        ):
                            if getattr(args, "test_class_path", None):
                                test_abs = os.path.abspath(args.test_class_path)
                                test_class_name = os.path.basename(test_abs)
                            else:
                                test_class_name = args.test_class_name
                            class_save_dir = os.path.join(
                                args.master_save_dir,
                                f"train_{class_name}_test_{test_class_name}",
                            )
                        else:
                            class_save_dir = os.path.join(
                                args.master_save_dir, class_name
                            )

                        blob_csv = os.path.join(class_save_dir, "blob_analysis.csv")
                        model_path = os.path.join(
                            class_save_dir, "blob_classifier_model.joblib"
                        )
                        scaler_path = os.path.join(
                            class_save_dir, "blob_classifier_scaler.joblib"
                        )
                        feat_path = os.path.join(
                            class_save_dir, "blob_classifier_features.json"
                        )

                        if os.path.exists(blob_csv):
                            # Preferred: check for a training blob CSV exported for the training split
                            train_blob_csv = os.path.join(
                                class_save_dir, "blob_analysis_train.csv"
                            )
                            test_blob_csv = os.path.join(
                                class_save_dir, "blob_analysis.csv"
                            )

                            model_exists = (
                                os.path.exists(model_path)
                                and os.path.exists(scaler_path)
                                and os.path.exists(feat_path)
                            )

                            clf = None
                            scaler = None
                            saved_feats = None

                            # Train on dedicated training blob CSV if available
                            if os.path.exists(train_blob_csv):
                                logging.info(
                                    f"Training blob classifier from training CSV: {train_blob_csv}"
                                )
                                # Version tag for logging: training from dedicated train CSV
                                logging.info("Blob classifier mode: TRAIN_FROM_TRAIN_CSV")
                                X_train_df, y_train, feat_names_train = (
                                    feature_factory_from_blob_df(train_blob_csv)
                                )
                                if y_train is None or (len(np.unique(y_train)) < 2):
                                    logging.warning(
                                        f"Insufficient labels in {train_blob_csv} to train blob classifier for {class_name}."
                                    )
                                else:
                                    logging.info(
                                        "Blob classifier training: starting fit on training CSV."
                                    )
                                    try:
                                        logging.info(
                                            f"Training CSV rows: {len(X_train_df)}, features: {len(X_train_df.columns)}"
                                        )
                                        if hasattr(y_train, "value_counts"):
                                            logging.info(
                                                f"Training class distribution: {y_train.value_counts().to_dict()}"
                                            )
                                        else:
                                            logging.info(
                                                f"Training classes (unique counts): {np.unique(y_train, return_counts=True)}"
                                            )
                                    except Exception:
                                        logging.exception("Failed to log training CSV summary")
                                    X_train_vals = X_train_df.fillna(0).values
                                    scaler = StandardScaler().fit(X_train_vals)
                                    clf = RandomForestClassifier(
                                        n_estimators=200, random_state=args.seed
                                    )
                                    clf.fit(
                                        scaler.transform(X_train_vals), y_train.values
                                    )
                                    try:
                                        logging.info(
                                            f"Trained RandomForest (sklearn {sklearn.__version__}) with params: {clf.get_params()}"
                                        )
                                    except Exception:
                                        logging.exception("Failed to log classifier params")
                                    os.makedirs(class_save_dir, exist_ok=True)
                                    joblib.dump(clf, model_path)
                                    joblib.dump(scaler, scaler_path)
                                    with open(feat_path, "w") as fh:
                                        json.dump(list(X_train_df.columns), fh)
                                    saved_feats = list(X_train_df.columns)
                                    logging.info(
                                        f"Saved blob classifier artifacts to: model={model_path}, scaler={scaler_path}, feats={feat_path}"
                                    )

                            # If we still don't have a trained model, consider loading existing ones
                            if (
                                not clf
                                and model_exists
                                and not getattr(args, "retrain_blob_classifier", False)
                            ):
                                # Version tag for logging: loading existing trained artifacts
                                logging.info("Blob classifier mode: LOAD_EXISTING_ARTIFACTS")
                                try:
                                    clf = joblib.load(model_path)
                                    scaler = joblib.load(scaler_path)
                                    with open(feat_path, "r") as fh:
                                        saved_feats = json.load(fh)
                                    try:
                                        logging.info(
                                            f"Loaded existing blob classifier artifacts. model={model_path}, scaler={scaler_path}, feats={feat_path}"
                                        )
                                        if hasattr(clf, "get_params"):
                                            logging.info(
                                                f"Loaded RandomForest (sklearn {sklearn.__version__}) params: {clf.get_params()}"
                                            )
                                    except Exception:
                                        logging.exception("Failed to log loaded classifier metadata")
                                except Exception:
                                    logging.exception(
                                        "Failed to load existing blob classifier artifacts; will attempt to (re)train if possible."
                                    )

                            # If no train CSV, fallback: split the available test blob CSV into train/holdout
                            if (clf is None) and os.path.exists(test_blob_csv):
                                # Version tag for logging: fallback internal split from test blob CSV
                                logging.info("Blob classifier mode: FALLBACK_SPLIT_FROM_TEST_CSV")
                                df_blob = pd.read_csv(test_blob_csv)
                                X_all, y_all, feat_names_all = (
                                    feature_factory_from_blob_df(df_blob)
                                )
                                if (
                                    y_all is None
                                    or (len(np.unique(y_all)) < 2)
                                    or (len(X_all) < 10)
                                ):
                                    logging.warning(
                                        f"Insufficient or single-class data in {test_blob_csv}; skipping blob classifier for {class_name}."
                                    )
                                else:
                                    # safe stratified split
                                    try:
                                        X_train, X_hold, y_train, y_hold = (
                                            train_test_split(
                                                X_all,
                                                y_all,
                                                test_size=0.3,
                                                random_state=args.seed,
                                                stratify=y_all,
                                            )
                                        )
                                    except Exception:
                                        X_train, X_hold, y_train, y_hold = (
                                            train_test_split(
                                                X_all,
                                                y_all,
                                                test_size=0.3,
                                                random_state=args.seed,
                                                stratify=None,
                                            )
                                        )
                                    try:
                                        logging.info(
                                            f"Performed fallback split: total={len(X_all)}, train={len(X_train)}, holdout={len(X_hold)}"
                                        )
                                        if hasattr(y_all, "value_counts"):
                                            logging.info(
                                                f"Overall class distribution: {y_all.value_counts().to_dict()}"
                                            )
                                        if hasattr(y_train, "value_counts"):
                                            logging.info(
                                                f"Train class distribution: {y_train.value_counts().to_dict()}"
                                            )
                                        if hasattr(y_hold, "value_counts"):
                                            logging.info(
                                                f"Holdout class distribution: {y_hold.value_counts().to_dict()}"
                                            )
                                    except Exception:
                                        logging.exception("Failed to log split distributions")
                                    X_train_vals = X_train.fillna(0).values
                                    scaler = StandardScaler().fit(X_train_vals)
                                    clf = RandomForestClassifier(
                                        n_estimators=200, random_state=args.seed
                                    )
                                    clf.fit(
                                        scaler.transform(X_train_vals), y_train.values
                                    )
                                    os.makedirs(class_save_dir, exist_ok=True)
                                    joblib.dump(clf, model_path)
                                    joblib.dump(scaler, scaler_path)
                                    saved_feats = list(X_train.columns)
                                    # prepare evaluation set
                                    X_ordered = X_hold.reindex(
                                        columns=saved_feats
                                    ).fillna(0)
                                    y_eval = y_hold
                                    X_eval_scaled = scaler.transform(X_ordered.values)
                                    eval_preds = clf.predict(X_eval_scaled).astype(int)
                                    new_preds = eval_preds
                                    X_scaled = X_eval_scaled
                                    y = y_eval
                                    eval_on_holdout = True

                            # If we have a model and a test CSV, run evaluation on the test CSV
                            if clf is not None and os.path.exists(test_blob_csv) and not eval_on_holdout:
                                try:
                                    # load test blobs and evaluate
                                    X_test_df, y_test, _ = feature_factory_from_blob_df(
                                        test_blob_csv
                                    )
                                    logging.info("Evaluating on full test CSV because no internal holdout was used.")
                                    try:
                                        logging.info(
                                            f"Evaluating blob classifier on test CSV: {test_blob_csv} (n={len(X_test_df)})"
                                        )
                                        if hasattr(y_test, "value_counts"):
                                            logging.info(
                                                f"Test class distribution: {y_test.value_counts().to_dict()}"
                                            )
                                    except Exception:
                                        logging.exception("Failed to log test CSV summary")
                                    if saved_feats is None:
                                        # try to derive from X_test
                                        saved_feats = list(X_test_df.columns)
                                    X_ordered = X_test_df.reindex(
                                        columns=saved_feats
                                    ).fillna(0)
                                    X_pred = X_ordered.values
                                    X_scaled = scaler.transform(X_pred)
                                    new_preds = clf.predict(X_scaled).astype(int)
                                    y = y_test
                                except Exception:
                                    logging.exception(
                                        "Failed to evaluate blob classifier on test CSV"
                                    )
                            else:
                                if clf is None:
                                    logging.info(
                                        "No blob classifier trained or available for this class."
                                    )
                                elif not os.path.exists(test_blob_csv):
                                    logging.warning(
                                        f"Test blob CSV not found at {test_blob_csv}; cannot evaluate blob classifier for {class_name}."
                                    )

                            # If we obtained predictions and labels, compute metrics and diagnostics
                            if (clf is not None) and (
                                ("new_preds" in locals()) and (y is not None)
                            ):
                                try:
                                    # If y is a pandas Series, ensure ordering alignment
                                    if hasattr(y, "values"):
                                        y_vals = y.values
                                    else:
                                        y_vals = np.asarray(list(y))
                                    tn, fp, fn, tp = confusion_matrix(
                                        y_vals, new_preds
                                    ).ravel()
                                except Exception:
                                    tn = fp = fn = tp = 0
                                epsilon = 1e-6
                                precision = tp / (tp + fp + epsilon)
                                recall = tp / (tp + fn + epsilon)
                                f1_score = (
                                    2
                                    * (precision * recall)
                                    / (precision + recall + epsilon)
                                )
                                class_summary.update(
                                    {
                                        "precision": round(precision, 4),
                                        "recall": round(recall, 4),
                                        "f1_score": round(f1_score, 4),
                                        "true_negatives": int(tn),
                                        "false_positives": int(fp),
                                        "false_negatives": int(fn),
                                        "true_positives": int(tp),
                                    }
                                )
                                logging.info(
                                    f"Blob classifier predictions used to override metrics for class {class_name}."
                                )
                                try:
                                    logging.info(
                                        f"Blob classifier predictions used to override metrics for class {class_name}."
                                    )
                                    logging.info(
                                        f"Blob classifier results: precision={precision:.4f}, recall={recall:.4f}, f1={f1_score:.4f}, TN={tn}, FP={fp}, FN={fn}, TP={tp}"
                                    )
                                except Exception:
                                    logging.exception("Failed to log blob classifier summary stats")

                                # Diagnostics: save preds CSV, report, plots
                                try:
                                    out_dir = class_save_dir
                                    probs = (
                                        clf.predict_proba(X_scaled)[:, 1]
                                        if hasattr(clf, "predict_proba")
                                        else None
                                    )
                                    preds_df = pd.DataFrame(
                                        {
                                            "image_name": list(X_ordered.index),
                                            "true_label": (
                                                list(y_vals)
                                                if y is not None
                                                else [None] * len(X_ordered)
                                            ),
                                            "pred": list(new_preds),
                                            "prob": (
                                                list(probs)
                                                if probs is not None
                                                else [None] * len(X_ordered)
                                            ),
                                        }
                                    )
                                    preds_df.to_csv(
                                        os.path.join(
                                            out_dir, "blob_classifier_predictions.csv"
                                        ),
                                        index=False,
                                    )

                                    report = {
                                        "used_blob_classifier": True,
                                        "n_images": int(X_ordered.shape[0]),
                                        "n_features": int(X_ordered.shape[1]),
                                    }
                                    if hasattr(clf, "feature_importances_"):
                                        importances = clf.feature_importances_
                                        feat_list = list(X_ordered.columns)
                                        top_idx = np.argsort(importances)[::-1][:10]
                                        report["top_features"] = [
                                            {
                                                "feature": feat_list[i],
                                                "importance": float(importances[i]),
                                            }
                                            for i in top_idx
                                        ]
                                    with open(
                                        os.path.join(
                                            out_dir, "blob_classifier_report.json"
                                        ),
                                        "w",
                                    ) as fh:
                                        json.dump(report, fh, indent=2)

                                    try:
                                        if hasattr(clf, "feature_importances_"):
                                            save_feature_importance(
                                                clf.feature_importances_,
                                                list(X_ordered.columns),
                                                out_dir,
                                            )
                                    except Exception:
                                        logging.exception(
                                            "Failed to save RF feature importances"
                                        )

                                    try:
                                        if (y is not None) and (probs is not None):
                                            plot_roc_pr(y_vals, probs, out_dir)
                                    except Exception:
                                        logging.exception(
                                            "Failed to plot ROC/PR for blob classifier"
                                        )

                                    try:
                                        if y is not None:
                                            save_confusion_matrix(
                                                y_vals,
                                                new_preds,
                                                out_dir,
                                                labels=["normal", "anomaly"],
                                            )
                                    except Exception:
                                        logging.exception(
                                            "Failed to save confusion matrix for blob classifier"
                                        )

                                    try:
                                        tsne_plot(
                                            X_ordered,
                                            (
                                                pd.Series(y_vals)
                                                if y is not None
                                                else None
                                            ),
                                            new_preds,
                                            out_dir,
                                            sample_frac=1.0,
                                        )
                                    except Exception:
                                        logging.exception(
                                            "Failed to create t-SNE for blob features"
                                        )

                                    try:
                                        explain_with_shap(
                                            clf,
                                            X_ordered,
                                            out_dir,
                                            list(X_ordered.columns),
                                            y=(
                                                pd.Series(y_vals)
                                                if y is not None
                                                else None
                                            ),
                                        )
                                    except Exception:
                                        logging.exception(
                                            "Failed to compute SHAP explanations for blob classifier"
                                        )
                                except Exception:
                                    logging.exception(
                                        "Failed to write blob classifier diagnostics"
                                    )
                                # --- Blob classifier diagnostics & verbose logging ---
                                try:
                                    out_dir = class_save_dir
                                    probs = (
                                        clf.predict_proba(X_scaled)[:, 1]
                                        if hasattr(clf, "predict_proba")
                                        else None
                                    )
                                    preds_df = pd.DataFrame(
                                        {
                                            "image_name": list(X_ordered.index),
                                            "true_label": (
                                                list(y.values)
                                                if y is not None
                                                else [None] * len(X_ordered)
                                            ),
                                            "pred": list(new_preds),
                                            "prob": (
                                                list(probs)
                                                if probs is not None
                                                else [None] * len(X_ordered)
                                            ),
                                        }
                                    )
                                    preds_df.to_csv(
                                        os.path.join(
                                            out_dir, "blob_classifier_predictions.csv"
                                        ),
                                        index=False,
                                    )

                                    report = {
                                        "used_blob_classifier": True,
                                        "n_images": int(X_ordered.shape[0]),
                                        "n_features": int(X_ordered.shape[1]),
                                    }
                                    if hasattr(clf, "feature_importances_"):
                                        importances = clf.feature_importances_
                                        feat_list = list(X_ordered.columns)
                                        top_idx = np.argsort(importances)[::-1][:10]
                                        report["top_features"] = [
                                            {
                                                "feature": feat_list[i],
                                                "importance": float(importances[i]),
                                            }
                                            for i in top_idx
                                        ]
                                    with open(
                                        os.path.join(
                                            out_dir, "blob_classifier_report.json"
                                        ),
                                        "w",
                                    ) as fh:
                                        json.dump(report, fh, indent=2)

                                    try:
                                        if hasattr(clf, "feature_importances_"):
                                            save_feature_importance(
                                                clf.feature_importances_,
                                                list(X_ordered.columns),
                                                out_dir,
                                            )
                                    except Exception:
                                        logging.exception(
                                            "Failed to save RF feature importances"
                                        )

                                    try:
                                        if (y is not None) and (probs is not None):
                                            plot_roc_pr(y.values, probs, out_dir)
                                    except Exception:
                                        logging.exception(
                                            "Failed to plot ROC/PR for blob classifier"
                                        )

                                    try:
                                        if y is not None:
                                            save_confusion_matrix(
                                                y.values,
                                                new_preds,
                                                out_dir,
                                                labels=["normal", "anomaly"],
                                            )
                                    except Exception:
                                        logging.exception(
                                            "Failed to save confusion matrix for blob classifier"
                                        )

                                    try:
                                        tsne_plot(
                                            X_ordered,
                                            y,
                                            new_preds,
                                            out_dir,
                                            sample_frac=1.0,
                                        )
                                    except Exception:
                                        logging.exception(
                                            "Failed to create t-SNE for blob features"
                                        )

                                    try:
                                        explain_with_shap(
                                            clf,
                                            X_ordered,
                                            out_dir,
                                            list(X_ordered.columns),
                                            y=y,
                                        )
                                    except Exception:
                                        logging.exception(
                                            "Failed to compute SHAP explanations for blob classifier"
                                        )
                                except Exception:
                                    logging.exception(
                                        "Failed to write blob classifier diagnostics"
                                    )
                        else:
                            logging.warning(
                                f"Expected blob CSV not found at: {blob_csv}; cannot train/use blob classifier for {class_name}."
                            )
                    except Exception:
                        logging.exception(
                            f"Blob classifier training/eval failed for {class_name}"
                        )
            except Exception as e:
                logging.error(
                    f"!!! FAILED to process class {class_name}. Error: {e}",
                    exc_info=True,
                )

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
