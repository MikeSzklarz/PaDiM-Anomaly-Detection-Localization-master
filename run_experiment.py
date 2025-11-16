import os
import random
import logging
import argparse

import pandas as pd

import torch
from torchvision.models import (
    wide_resnet50_2,
    resnet18,
    efficientnet_b5,
    EfficientNet_B5_Weights,
)

import datasets.bowtie as bowtie
import copy

import trainer

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
        "--dataset",
        type=str,
        default="bowtie",
        choices=["bowtie", "mvtec"],
        help="Which dataset loader to use."
    )
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
