# --- Standard Library Imports ---
import os  # Provides a way of using operating system dependent functionality, like creating directories.
import argparse  # For parsing command-line arguments, allowing flexible script configuration.
import random  # Used for generating random numbers, specifically for reproducible feature selection.
import pickle  # For serializing and de-serializing Python object structures, used here to save and load pre-calculated features.
from collections import (
    OrderedDict,
)  # A dictionary subclass that remembers the order in which items were inserted.

# --- Third-party Library Imports ---
import numpy as np  # Fundamental package for numerical computing with Python.
import torch  # The main PyTorch library for tensor computations and neural networks.
import torch.nn.functional as F  # Provides common neural network functions like interpolation (resizing).
from torch.utils.data import (
    DataLoader,
)  # Utility for creating batches of data from a Dataset.
from torchvision.models import (
    wide_resnet50_2,
    resnet18,
)  # Pre-trained computer vision models from torchvision.
from tqdm import tqdm  # A library to create smart, fast progress bars for loops.

# --- Scientific and Plotting Imports ---
from scipy.spatial.distance import (
    mahalanobis,
)  # Calculates the Mahalanobis distance, a key part of the PaDiM algorithm.
from scipy.ndimage import (
    gaussian_filter,
)  # Applies a Gaussian filter, used for smoothing the anomaly map.
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
)  # Metrics for evaluating the model's performance.
import matplotlib.pyplot as plt  # The primary library for creating visualizations.
import matplotlib  # Matplotlib's base package, used here for color normalization in plots.

# --- Custom Local Imports ---
# This assumes you have a 'datasets' folder with 'bowtie.py' defining your dataset class.
import datasets.bowtie as bowtie_dataset

# --- Device Configuration ---
# Check if a CUDA-enabled GPU is available for computation.
# Using a GPU significantly speeds up the deep learning operations.
IS_CUDA_AVAILABLE = torch.cuda.is_available()
# Set the primary computation device based on availability.
DEVICE = torch.device("cuda" if IS_CUDA_AVAILABLE else "cpu")
print(f"--- Running on device: {DEVICE} ---")


def parse_command_line_args():
    """
    Parses and returns command-line arguments for the script.

    This function sets up the argument parser to allow the user to specify
    the data path, save path, and model architecture from the command line,
    making the script more flexible and easier to run with different configurations.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="PaDiM Anomaly Detection")

    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/bowtie",
        help="Path to the root directory of the dataset.",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="./results/padim_bowtie",
        help="Path to the directory where results and model data will be saved.",
    )

    parser.add_argument(
        "--arch",
        type=str,
        choices=["resnet18", "wide_resnet50_2"],
        default="wide_resnet50_2",
        help="The architecture of the pre-trained feature extractor to use.",
    )

    return parser.parse_args()


def concatenate_embeddings(embedding_layer_1, embedding_layer_2):
    """
    Concatenates two feature maps of different spatial resolutions from a CNN.

    This function is crucial for combining features from different stages of the network.
    It uses `unfold` and `fold` operations to intelligently resize and merge the feature maps.
    The first feature map (from an earlier layer) is downsampled to match the spatial
    dimensions of the second feature map (from a deeper layer) before concatenation.

    Args:
        embedding_layer_1 (torch.Tensor): The first feature map (larger spatial size).
        embedding_layer_2 (torch.Tensor): The second feature map (smaller spatial size).

    Returns:
        torch.Tensor: The resulting concatenated feature map.
    """
    # Get the dimensions of the input tensors.
    batch_size_1, channels_1, height_1, width_1 = embedding_layer_1.size()
    _, channels_2, height_2, width_2 = embedding_layer_2.size()

    # Calculate the stride required to match the spatial dimensions.
    # This assumes height_1 and width_1 are multiples of height_2 and width_2.
    stride = int(height_1 / height_2)

    # Use `unfold` to extract sliding local blocks from the first embedding.
    # This effectively reshapes the tensor to align with the smaller feature map's dimensions.
    unfolded_embedding_1 = F.unfold(
        embedding_layer_1, kernel_size=stride, dilation=1, stride=stride
    )

    # Reshape the unfolded tensor to prepare for concatenation.
    unfolded_embedding_1 = unfolded_embedding_1.view(
        batch_size_1, channels_1, -1, height_2, width_2
    )

    # Create a new tensor to hold the combined embeddings.
    concatenated_tensor = torch.zeros(
        batch_size_1,
        channels_1 + channels_2,
        unfolded_embedding_1.size(2),
        height_2,
        width_2,
    )

    # Loop through the patches and concatenate features channel-wise.
    for i in range(unfolded_embedding_1.size(2)):
        patch = unfolded_embedding_1[:, :, i, :, :]
        concatenated_tensor[:, :, i, :, :] = torch.cat((patch, embedding_layer_2), 1)

    # Reshape the concatenated tensor back into a 2D feature map representation.
    concatenated_tensor = concatenated_tensor.view(batch_size_1, -1, height_2 * width_2)

    # Use `fold` to reconstruct the feature map to its original larger spatial size.
    final_embedding = F.fold(
        concatenated_tensor,
        kernel_size=stride,
        output_size=(height_1, width_1),
        stride=stride,
    )

    return final_embedding


def denormalize_image(image_tensor):
    """
    Denormalizes an image tensor for visualization.

    The input tensor is assumed to have been normalized using ImageNet standards.
    This function reverses that process to convert the tensor back into a displayable
    image format (0-255 uint8).

    Args:
        image_tensor (numpy.ndarray): A normalized image tensor in (C, H, W) format.

    Returns:
        numpy.ndarray: The denormalized image in (H, W, C) format.
    """
    # ImageNet standard normalization values.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Transpose from (C, H, W) to (H, W, C), reverse normalization, and scale to 0-255.
    denormalized_image = (
        ((image_tensor.transpose(1, 2, 0) * std) + mean) * 255.0
    ).astype(np.uint8)

    return denormalized_image


def visualize_and_save_results(
    test_images_list,
    normalized_anomaly_maps,
    image_level_anomaly_scores,
    save_directory,
    class_name,
):
    """
    Generates and saves visualization images for each test sample.

    For each test image, this function creates a figure with three subplots:
    1. The original input image.
    2. A heatmap of the anomaly scores, overlaid on the original image.
    3. The original image with the overall anomaly score printed on it.

    Args:
        test_images_list (list): A list of original test image tensors.
        normalized_anomaly_maps (numpy.ndarray): A batch of normalized anomaly maps.
        image_level_anomaly_scores (numpy.ndarray): An array of anomaly scores, one for each image.
        save_directory (str): The directory where the output images will be saved.
        class_name (str): The name of the class being processed, used for file naming.
    """
    num_images = len(normalized_anomaly_maps)

    # Determine the color scale range for the heatmaps across the entire batch.
    heatmap_max_value = normalized_anomaly_maps.max() * 255.0
    heatmap_min_value = normalized_anomaly_maps.min() * 255.0
    color_normalizer = matplotlib.colors.Normalize(
        vmin=heatmap_min_value, vmax=heatmap_max_value
    )

    for i in range(num_images):
        original_image = test_images_list[i]
        original_image = denormalize_image(
            original_image
        )  # Convert back to viewable format

        # Scale heatmap to 0-255 for visualization
        single_image_heatmap = normalized_anomaly_maps[i] * 255

        # --- Create the plot ---
        figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        figure.subplots_adjust(right=0.9)  # Adjust subplot to make room for colorbar

        # Turn off axes for all subplots
        for axis in axes:
            axis.axes.xaxis.set_visible(False)
            axis.axes.yaxis.set_visible(False)

        # Subplot 1: Original Image
        axes[0].imshow(original_image)
        axes[0].title.set_text("Original Image")

        # Subplot 2: Heatmap Overlay
        # Display the heatmap itself
        heatmap_plot = axes[1].imshow(
            single_image_heatmap, cmap="jet", norm=color_normalizer
        )
        # Overlay the original image with transparency
        axes[1].imshow(original_image, cmap="gray", alpha=0.5, interpolation="none")
        axes[1].title.set_text("Anomaly Heatmap")

        # Subplot 3: Result with Score
        axes[2].imshow(original_image)
        axes[2].title.set_text("Result")
        # Display the final image-level anomaly score on the plot
        axes[2].text(
            5,
            20,
            f"Anomaly Score: {image_level_anomaly_scores[i]:.3f}",
            color="white",
            backgroundcolor="black",
            fontsize=10,
        )

        # --- Add Colorbar ---
        # Define position for the colorbar axis
        left, bottom, width, height = 0.92, 0.15, 0.015, 1 - 2 * 0.15
        colorbar_axis = figure.add_axes([left, bottom, width, height])

        # Create and configure the colorbar
        colorbar = plt.colorbar(heatmap_plot, cax=colorbar_axis)
        colorbar.ax.tick_params(labelsize=8)
        font_properties = {
            "family": "serif",
            "color": "black",
            "weight": "normal",
            "size": 8,
        }
        colorbar.set_label("Anomaly Score", fontdict=font_properties)

        # Save the final figure to a file
        output_filepath = os.path.join(save_directory, f"{class_name}_{i:03d}.png")
        figure.savefig(output_filepath, dpi=100)
        plt.close(figure)


def main():
    """
    The main execution function for the PaDiM algorithm.

    This function orchestrates the entire process:
    1.  Parses arguments and sets up the model and environment.
    2.  Loops through each class in the dataset.
    3.  For each class:
        a. Extracts features from the 'good' training images.
        b. Builds a statistical model (multivariate Gaussian) of normalcy.
        c. Extracts features from test images.
        d. Computes anomaly scores using Mahalanobis distance.
        e. Evaluates performance using ROC AUC.
        f. Saves visualizations.
    4.  Reports the average performance across all classes.
    """
    # --- 1. Initialization and Setup ---
    command_line_args = parse_command_line_args()

    print("Loading pre-trained model...")
    # Select the model architecture and define its feature dimensions.
    # 'total_feature_dimension' is the combined number of channels from the selected layers.
    # 'reduced_feature_dimension' is the number of channels after random selection.
    if command_line_args.arch == "resnet18":
        feature_extractor_model = resnet18(pretrained=True, progress=True)
        total_feature_dimension = 448
        reduced_feature_dimension = 100
    elif command_line_args.arch == "wide_resnet50_2":
        feature_extractor_model = wide_resnet50_2(pretrained=True, progress=True)
        total_feature_dimension = 1792
        reduced_feature_dimension = 550

    feature_extractor_model.to(DEVICE)  # Move model to the selected device (GPU or CPU)
    feature_extractor_model.eval()  # Set model to evaluation mode (disables dropout, batchnorm updates, etc.)

    # Set seeds for reproducibility of results.
    random.seed(1024)
    torch.manual_seed(1024)
    if IS_CUDA_AVAILABLE:
        torch.cuda.manual_seed_all(1024)

    # --- 2. Feature Selection and Hook Setup ---
    # Randomly select a subset of feature dimensions. This is a key part of PaDiM
    # that reduces memory and computation while maintaining performance.
    random_feature_indices = torch.tensor(
        random.sample(range(0, total_feature_dimension), reduced_feature_dimension)
    )

    # PyTorch "hooks" are used to intercept and capture the output of intermediate layers
    # during the forward pass, without modifying the model's code.
    feature_map_outputs = []

    def get_intermediate_layer_output(module, input_data, output_data):
        """Hook function to capture the output of a layer."""
        feature_map_outputs.append(output_data)

    # Register the hook on the final block of the first three layers of the ResNet model.
    feature_extractor_model.layer1[-1].register_forward_hook(
        get_intermediate_layer_output
    )
    feature_extractor_model.layer2[-1].register_forward_hook(
        get_intermediate_layer_output
    )
    feature_extractor_model.layer3[-1].register_forward_hook(
        get_intermediate_layer_output
    )

    # --- 3. Prepare for Evaluation and Visualization ---
    # Create directories to store cached features and results.
    temp_save_dir = os.path.join(
        command_line_args.save_path, f"temp_{command_line_args.arch}"
    )
    os.makedirs(temp_save_dir, exist_ok=True)

    # Set up a plot to aggregate ROC curves for all classes.
    figure, roc_auc_plot_axis = plt.subplots(1, 1, figsize=(10, 10))

    all_class_roc_aucs = []

    # --- 4. Main Loop: Process Each Class in the Dataset ---
    for current_class_name in bowtie_dataset.CLASS_NAMES:
        print(f"\n--- Processing Class: {current_class_name} ---")

        # --- 4a. Setup DataLoaders ---
        train_dataset = bowtie_dataset.BowtieDataset(
            command_line_args.data_path, class_name=current_class_name, is_train=True
        )
        train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)

        test_dataset = bowtie_dataset.BowtieDataset(
            command_line_args.data_path, class_name=current_class_name, is_train=False
        )
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)

        # Dictionaries to store feature maps from each layer for training and testing.
        train_feature_maps = OrderedDict(
            [("layer1", []), ("layer2", []), ("layer3", [])]
        )
        test_feature_maps = OrderedDict(
            [("layer1", []), ("layer2", []), ("layer3", [])]
        )

        # --- 4b. Build Statistical Model of Normalcy (Training Phase) ---
        # To save time, check if the training features have already been computed and saved.
        cached_train_features_filepath = os.path.join(
            temp_save_dir, f"train_{current_class_name}.pkl"
        )

        if not os.path.exists(cached_train_features_filepath):
            print(
                f"Calculating and caching training features for {current_class_name}..."
            )
            # Loop through the training data (which contains only 'good' samples).
            for images, _, _ in tqdm(
                train_dataloader,
                f"| Feature Extraction | Train | {current_class_name} |",
            ):
                with torch.no_grad():
                    # Perform a forward pass. The hooks will automatically capture the layer outputs.
                    _ = feature_extractor_model(images.to(DEVICE))

                # Assign the captured outputs to the corresponding layers in our dictionary.
                for layer_name, feature_map in zip(
                    train_feature_maps.keys(), feature_map_outputs
                ):
                    train_feature_maps[layer_name].append(feature_map.cpu().detach())

                # Clear the list for the next batch.
                feature_map_outputs = []

            # Concatenate all batch features into a single tensor for each layer.
            for layer_name, feature_list in train_feature_maps.items():
                train_feature_maps[layer_name] = torch.cat(feature_list, 0)

            # Concatenate features from all three layers into a single embedding vector.
            concatenated_embedding_vectors = train_feature_maps["layer1"]
            for layer_name in ["layer2", "layer3"]:
                concatenated_embedding_vectors = concatenate_embeddings(
                    concatenated_embedding_vectors, train_feature_maps[layer_name]
                )

            # --- Apply Random Feature Reduction ---
            concatenated_embedding_vectors = torch.index_select(
                concatenated_embedding_vectors, 1, random_feature_indices
            )

            # --- Calculate Mean and Covariance (The Core of PaDiM Training) ---
            # This builds the statistical model of what "normal" features look like.
            batch_size, channels, height, width = concatenated_embedding_vectors.size()
            # Reshape for statistical calculation: (batch, channels, height*width)
            concatenated_embedding_vectors = concatenated_embedding_vectors.view(
                batch_size, channels, height * width
            )

            # Calculate the mean embedding for each patch location across all training samples.
            mean_of_train_features = torch.mean(
                concatenated_embedding_vectors, dim=0
            ).numpy()

            # Calculate the covariance matrix for each patch location.
            covariance_of_train_features = torch.zeros(
                channels, channels, height * width
            ).numpy()
            identity_matrix = np.identity(channels)

            for i in range(height * width):  # For each patch location
                # Calculate covariance matrix and add a small value (epsilon * I) for numerical stability.
                # This ensures the matrix is invertible.
                patch_embeddings = concatenated_embedding_vectors[:, :, i].numpy()
                covariance_of_train_features[:, :, i] = (
                    np.cov(patch_embeddings, rowvar=False) + 0.01 * identity_matrix
                )

            # `learned_distribution` contains the mean and covariance that define normalcy.
            learned_distribution = [
                mean_of_train_features,
                covariance_of_train_features,
            ]

            # Save (cache) the learned distribution to a file to avoid re-computation.
            with open(cached_train_features_filepath, "wb") as file:
                pickle.dump(learned_distribution, file)
        else:
            print(
                f"Loading cached training features from: {cached_train_features_filepath}"
            )
            with open(cached_train_features_filepath, "rb") as file:
                learned_distribution = pickle.load(file)

        # --- 4c. Anomaly Scoring (Testing Phase) ---
        print(f"Performing anomaly detection on test set for {current_class_name}...")
        ground_truth_labels_list = []
        test_images_list = []

        # Extract features for all test images.
        for images, labels, _ in tqdm(
            test_dataloader, f"| Feature Extraction | Test | {current_class_name} |"
        ):
            test_images_list.extend(images.cpu().detach().numpy())
            ground_truth_labels_list.extend(labels.cpu().detach().numpy())

            with torch.no_grad():
                _ = feature_extractor_model(images.to(DEVICE))

            for layer_name, feature_map in zip(
                test_feature_maps.keys(), feature_map_outputs
            ):
                test_feature_maps[layer_name].append(feature_map.cpu().detach())

            feature_map_outputs = []

        for layer_name, feature_list in test_feature_maps.items():
            test_feature_maps[layer_name] = torch.cat(feature_list, 0)

        # Concatenate and reduce features for the test set, same as for the training set.
        concatenated_embedding_vectors_test = test_feature_maps["layer1"]
        for layer_name in ["layer2", "layer3"]:
            concatenated_embedding_vectors_test = concatenate_embeddings(
                concatenated_embedding_vectors_test, test_feature_maps[layer_name]
            )

        concatenated_embedding_vectors_test = torch.index_select(
            concatenated_embedding_vectors_test, 1, random_feature_indices
        )

        # --- Calculate Mahalanobis Distance to find anomalies ---
        batch_size, channels, height, width = concatenated_embedding_vectors_test.size()
        concatenated_embedding_vectors_test = concatenated_embedding_vectors_test.view(
            batch_size, channels, height * width
        ).numpy()

        mahalanobis_distances_list = []

        for i in range(height * width):  # For each patch location
            mean = learned_distribution[0][:, i]
            # Pre-calculate the inverse of the covariance matrix for efficiency.
            inverse_covariance = np.linalg.inv(learned_distribution[1][:, :, i])

            # Calculate the Mahalanobis distance for each test sample's patch from the learned "normal" distribution.
            distances = [
                mahalanobis(sample[:, i], mean, inverse_covariance)
                for sample in concatenated_embedding_vectors_test
            ]
            mahalanobis_distances_list.append(distances)

        # Reshape the list of distances into an anomaly map for each image.
        mahalanobis_distances_list = (
            np.array(mahalanobis_distances_list)
            .transpose(1, 0)
            .reshape(batch_size, height, width)
        )

        # Upsample the anomaly map to the original image size and apply Gaussian smoothing.
        anomaly_map = torch.tensor(mahalanobis_distances_list)
        anomaly_map = (
            F.interpolate(
                anomaly_map.unsqueeze(1),
                size=images.size(2),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .numpy()
        )

        for i in range(anomaly_map.shape[0]):
            anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)

        # --- 4d. Evaluation ---
        # Normalize scores to a 0-1 range for consistency.
        max_score = anomaly_map.max()
        min_score = anomaly_map.min()
        normalized_anomaly_map = (anomaly_map - min_score) / (max_score - min_score)

        # The final anomaly score for an entire image is the maximum score found in its anomaly map.
        image_level_anomaly_scores = normalized_anomaly_map.reshape(
            normalized_anomaly_map.shape[0], -1
        ).max(axis=1)

        # Calculate the ROC AUC score to measure how well the model distinguishes normal from anomalous images.
        ground_truth_labels_list = np.asarray(ground_truth_labels_list)
        false_positive_rate, true_positive_rate, _ = roc_curve(
            ground_truth_labels_list, image_level_anomaly_scores
        )
        image_level_roc_auc = roc_auc_score(
            ground_truth_labels_list, image_level_anomaly_scores
        )

        all_class_roc_aucs.append(image_level_roc_auc)
        print(
            f"Image-level ROC AUC for {current_class_name}: {image_level_roc_auc:.3f}"
        )

        # Add this class's ROC curve to the main plot.
        roc_auc_plot_axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{current_class_name} ROC AUC: {image_level_roc_auc:.3f}",
        )

        # --- 4e. Visualization ---
        visualization_save_directory = os.path.join(
            command_line_args.save_path, f"pictures_{command_line_args.arch}"
        )
        os.makedirs(visualization_save_directory, exist_ok=True)
        visualize_and_save_results(
            test_images_list,
            normalized_anomaly_map,
            image_level_anomaly_scores,
            visualization_save_directory,
            current_class_name,
        )

    # --- 5. Final Report and Cleanup ---
    mean_roc_auc = np.mean(all_class_roc_aucs)
    print(
        f"\n--- Average Image-level ROC AUC across all classes: {mean_roc_auc:.3f} ---"
    )

    # Finalize and save the aggregated ROC curve plot.
    roc_auc_plot_axis.title.set_text(f"Average Image ROC AUC: {mean_roc_auc:.3f}")
    roc_auc_plot_axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(os.path.join(command_line_args.save_path, "roc_curve.png"), dpi=100)


# This is the standard entry point for a Python script.
if __name__ == "__main__":
    main()
