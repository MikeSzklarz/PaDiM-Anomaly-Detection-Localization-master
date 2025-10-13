import os
import random  # <-- Added for random sampling
from PIL import Image
import logging
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


def get_class_names(dataset_path: str):
    """Return a sorted list of class folder names found directly under dataset_path.

    This allows experiments to work with datasets where some classes have been
    combined into a single folder (for example: <dataset_root>/combined/{train,test})
    or with the usual structure (<dataset_root>/116/{train,test}, <dataset_root>/117/...).

    If the dataset_path does not exist or contains no subdirectories, an empty
    list is returned.
    """
    try:
        entries = [
            d
            for d in sorted(os.listdir(dataset_path))
            if os.path.isdir(os.path.join(dataset_path, d))
        ]
    except Exception as e:
        logging.error(f"Could not list classes in '{dataset_path}': {e}")
        return []
    return entries


class BowtieDataset(Dataset):
    """
    Custom PyTorch Dataset for the Bowtie dataset.

    - For training, it loads all 'good' images from the train path.
    - For testing, it loads all abnormal images from the test path and
      randomly samples a portion of 'good' images from the train path
      to serve as the normal test set.
    """

    def __init__(
        self,
        dataset_path: str,
        class_name: str = "116",
        is_train: bool = True,
        resize: int = 512,
        cropsize: int = 224,
        image_extension: str = ".jpg",
        normal_test_sample_ratio: float = 0.2,
        seed: int = 1024,
        # Augmentation options (only applied when is_train==True)
        augmentations_enabled: bool = False,
        horizontal_flip: bool = False,
        vertical_flip: bool = False,
        augmentation_prob: float = 0.5,
    ):
        """
        Args:
            dataset_path (str): Path to the root of the dataset directory.
            class_name (str): The specific class folder to load from (e.g., "116").
            is_train (bool): If True, loads from the 'train' directory. If False, creates a test set.
            resize (int): The size to which images will be resized.
            cropsize (int): The size of the center crop taken after resizing.
            image_extension (str): The file extension of the images to load (e.g., ".jpg", ".png").
            normal_test_sample_ratio (float): Fraction of normal training data to use as normal test data.
        """
        # It's possible the caller has restructured the dataset (combined folders,
        # etc). We don't enforce a global CLASS_NAMES list anymore. Instead,
        # perform a quick sanity check that the expected class folder exists and
        # contains a 'train' directory.
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            raise ValueError(
                f"Class directory does not exist: {class_dir}. "
                "Use get_class_names(dataset_path) to discover available classes."
            )
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        self.image_extension = image_extension
        self.normal_test_sample_ratio = normal_test_sample_ratio

        self.image_filepaths, self.labels = self.load_dataset_folder()
        # Build transformation pipeline. Optionally include augmentations for training.
        transform_list = [T.Resize(resize, Image.Resampling.LANCZOS)]

        if is_train and augmentations_enabled:
            aug_ops = []
            if horizontal_flip:
                # We'll use RandomHorizontalFlip inside RandomApply so the overall
                # chance for *any* augmentation to happen is controlled by augmentation_prob
                aug_ops.append(T.RandomHorizontalFlip(p=1.0))
            if vertical_flip:
                aug_ops.append(T.RandomVerticalFlip(p=1.0))
            if aug_ops:
                transform_list.append(T.RandomApply(aug_ops, p=augmentation_prob))

        transform_list.extend(
            [
                T.CenterCrop(cropsize),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.transformations = T.Compose(transform_list)

    def __getitem__(self, idx):
        filepath, label = self.image_filepaths[idx], self.labels[idx]
        image = Image.open(filepath).convert("RGB")
        transformed_image = self.transformations(image)
        empty_mask = torch.zeros([1, self.cropsize, self.cropsize])
        return transformed_image, label, empty_mask

    def __len__(self):
        return len(self.image_filepaths)

    def load_dataset_folder(self):
        image_filepaths, labels = [], []

        if self.is_train:
            # --- TRAINING SET: Load all 'good' images from the train folder ---
            train_good_dir = os.path.join(
                self.dataset_path, self.class_name, "train", "good"
            )
            image_filenames = sorted(
                [
                    os.path.join(train_good_dir, f)
                    for f in os.listdir(train_good_dir)
                    if f.endswith(self.image_extension)
                ]
            )
            image_filepaths.extend(image_filenames)
            labels.extend([0] * len(image_filenames))
        else:
            # --- TEST SET: Load abnormal images from test AND sample normal images from train ---
            # 1. Load ABNORMAL images from the test folder
            test_dir = os.path.join(self.dataset_path, self.class_name, "test")
            defect_types = [
                d
                for d in sorted(os.listdir(test_dir))
                if os.path.isdir(os.path.join(test_dir, d))
            ]

            for defect_type in defect_types:
                # Skip any 'good' folder if it exists in test, as we sample from train
                if defect_type == "good":
                    continue

                defect_type_dir = os.path.join(test_dir, defect_type)
                image_filenames = sorted(
                    [
                        os.path.join(defect_type_dir, f)
                        for f in os.listdir(defect_type_dir)
                        if f.endswith(self.image_extension)
                    ]
                )
                image_filepaths.extend(image_filenames)
                labels.extend([1] * len(image_filenames))

            # 2. Sample NORMAL images from the training 'good' folder
            train_good_dir = os.path.join(
                self.dataset_path, self.class_name, "train", "good"
            )
            normal_image_files = sorted(
                [
                    os.path.join(train_good_dir, f)
                    for f in os.listdir(train_good_dir)
                    if f.endswith(self.image_extension)
                ]
            )

            num_to_sample = int(len(normal_image_files) * self.normal_test_sample_ratio)
            # Ensure we don't try to sample more than available
            num_to_sample = min(num_to_sample, len(normal_image_files))

            print(
                f"Sampling {num_to_sample} normal images from training set for testing."
            )

            # Use random.sample to get a random subset without replacement
            sampled_normal_files = random.sample(normal_image_files, num_to_sample)

            image_filepaths.extend(sampled_normal_files)
            labels.extend([0] * len(sampled_normal_files))

        assert len(image_filepaths) == len(
            labels
        ), "Number of images and labels should be the same"
        return list(image_filepaths), list(labels)


class _BowtieSubset(Dataset):
    """
    An internal, simple PyTorch Dataset. It is initialized with a pre-defined
    list of image files and their corresponding labels. This class does no
    data discovery or splitting; it only handles loading and transforming.
    """

    def __init__(
        self,
        image_files: list,
        labels: list,
        resize: int,
        cropsize: int,
        # Augmentation options (only applied for training subset)
        augmentations_enabled: bool = False,
        horizontal_flip: bool = False,
        vertical_flip: bool = False,
        augmentation_prob: float = 0.5,
    ):
        self.image_filepaths = image_files
        self.labels = labels
        self.cropsize = cropsize
        # Store augmentation config for inspection
        self.augmentations_enabled = augmentations_enabled
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.augmentation_prob = augmentation_prob

        transform_list = [T.Resize(resize, Image.Resampling.LANCZOS)]
        if augmentations_enabled:
            aug_ops = []
            if horizontal_flip:
                aug_ops.append(T.RandomHorizontalFlip(p=1.0))
            if vertical_flip:
                aug_ops.append(T.RandomVerticalFlip(p=1.0))
            if aug_ops:
                transform_list.append(T.RandomApply(aug_ops, p=augmentation_prob))

        transform_list.extend(
            [
                T.CenterCrop(cropsize),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.transformations = T.Compose(transform_list)

    def __getitem__(self, idx):
        filepath = self.image_filepaths[idx]
        label = self.labels[idx]

        image = Image.open(filepath).convert("RGB")
        transformed_image = self.transformations(image)
        empty_mask = torch.zeros([1, self.cropsize, self.cropsize])
        return transformed_image, label, empty_mask

    def __len__(self):
        return len(self.image_filepaths)


class BowtieDataManager:
    """
    A data manager class that handles finding, splitting, and preparing the
    Bowtie train and test datasets.
    """

    def __init__(
        self,
        dataset_path: str,
        class_name: str,
        normal_test_sample_ratio: float = 0.2,
        image_extension: str = ".jpg",
        resize: int = 512,
        cropsize: int = 224,
        seed: int = 1024,
        augmentations_enabled: bool = False,
        horizontal_flip: bool = False,
        vertical_flip: bool = False,
        augmentation_prob: float = 0.5,
    ):
        """
        Initializes the data manager, which finds all files, performs a
        clean train/test split, and creates the dataset objects.
        """
        random.seed(seed)  # Set seed for reproducible splits

        # --- 1. Discover and partition all data files ---
        train_files, train_labels, test_files, test_labels = (
            self._discover_and_split_files(
                dataset_path, class_name, normal_test_sample_ratio, image_extension
            )
        )

        # --- 2. Create and store the train and test dataset attributes ---
        self.train = _BowtieSubset(
            image_files=train_files,
            labels=train_labels,
            resize=resize,
            cropsize=cropsize,
            augmentations_enabled=augmentations_enabled,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            augmentation_prob=augmentation_prob,
        )
        # Ensure test subset has augmentations disabled for consistent evaluation
        self.test = _BowtieSubset(
            image_files=test_files,
            labels=test_labels,
            resize=resize,
            cropsize=cropsize,
            augmentations_enabled=False,
        )

        # *** MODIFIED: Changed from print() to logging.info() ***
        logging.info("Data manager created.")
        logging.info(f"Training set size: {len(self.train)} images")
        logging.info(f"Testing set size: {len(self.test)} images")

    def print_dataset_summary(self):
        """Prints a concise summary of what files and settings are being used.

        Includes counts of total/train/test images, number of abnormal/normal in test,
        and augmentation configuration for the training subset (expected proportion of
        augmented images per epoch = augmentation_prob).
        """
        total_images = len(self.train) + len(self.test)
        num_train = len(self.train)
        num_test = len(self.test)

        # Count normal vs abnormal in test set
        test_labels = self.test.labels
        num_test_normal = int(sum(1 for l in test_labels if l == 0))
        num_test_abnormal = int(sum(1 for l in test_labels if l == 1))

        print("--- Bowtie Dataset Summary ---")
        print(
            f"Dataset path (root): {os.path.abspath(os.path.join(test_labels and self.test.image_filepaths[0] or '.', '..', '..'))}"
        )
        print(f"Total images discovered (train+test): {total_images}")
        print(f" - Train images: {num_train}")
        print(
            f" - Test images:  {num_test} (Normal: {num_test_normal}, Abnormal: {num_test_abnormal})"
        )
        print("")
        print("Training augmentation settings:")
        print(f" - Augmentations enabled: {self.train.augmentations_enabled}")
        print(f" - Horizontal flip:      {self.train.horizontal_flip}")
        print(f" - Vertical flip:        {self.train.vertical_flip}")
        print(f" - Augmentation prob:    {self.train.augmentation_prob}")
        if self.train.augmentations_enabled:
            expected_aug_per_epoch = int(self.train.augmentation_prob * num_train)
            print(
                f" - Expected augmented images per epoch (approx): {expected_aug_per_epoch} of {num_train}"
            )
        print("-------------------------------")

    def _discover_and_split_files(self, dataset_path, class_name, ratio, ext):
        # (The rest of this class remains the same...)

        # Define paths
        normal_dir = os.path.join(dataset_path, class_name, "train", "good")
        abnormal_dir = os.path.join(dataset_path, class_name, "test")

        # Get all normal files, shuffle, and split
        all_normal_files = sorted(
            [
                os.path.join(normal_dir, f)
                for f in os.listdir(normal_dir)
                if f.endswith(ext)
            ]
        )
        random.shuffle(all_normal_files)

        split_idx = int(len(all_normal_files) * (1 - ratio))
        train_normal_files = all_normal_files[:split_idx]
        test_normal_files = all_normal_files[split_idx:]

        # Get all abnormal files
        test_abnormal_files = []
        defect_types = [
            d
            for d in sorted(os.listdir(abnormal_dir))
            if os.path.isdir(os.path.join(abnormal_dir, d))
        ]
        for defect_type in defect_types:
            if defect_type == "good":
                continue
            defect_dir = os.path.join(abnormal_dir, defect_type)
            test_abnormal_files.extend(
                sorted(
                    [
                        os.path.join(defect_dir, f)
                        for f in os.listdir(defect_dir)
                        if f.endswith(ext)
                    ]
                )
            )

        # Create final file and label lists
        train_files = train_normal_files
        train_labels = [0] * len(train_files)
        test_files = test_normal_files + test_abnormal_files
        test_labels = [0] * len(test_normal_files) + [1] * len(test_abnormal_files)

        return train_files, train_labels, test_files, test_labels

    def print_train_files(self):
        """Utility method to print all training file paths and labels."""
        for filepath, label in zip(self.train.image_filepaths, self.train.labels):
            print(f"File: {filepath}, Label: {label}")

    def print_test_files(self):
        """Utility method to print all testing file paths and labels."""
        for filepath, label in zip(self.test.image_filepaths, self.test.labels):
            print(f"File: {filepath}, Label: {label}")
