import os
import random  # <-- Added for random sampling
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

# Define the class names for your Bowtie dataset
CLASS_NAMES = [
    "116",
    # "117",
    # "118",
    # "119",
]


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
        assert class_name in CLASS_NAMES, f"class_name: {class_name}, should be in {CLASS_NAMES}"
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        self.image_extension = image_extension
        self.normal_test_sample_ratio = normal_test_sample_ratio
        

        self.image_filepaths, self.labels = self.load_dataset_folder()

        self.transformations = T.Compose(
            [
                T.Resize(resize, Image.Resampling.LANCZOS),
                T.CenterCrop(cropsize),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

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
            train_good_dir = os.path.join(self.dataset_path, self.class_name, "train", "good")
            image_filenames = sorted([
                os.path.join(train_good_dir, f)
                for f in os.listdir(train_good_dir)
                if f.endswith(self.image_extension)
            ])
            image_filepaths.extend(image_filenames)
            labels.extend([0] * len(image_filenames))
        else:
            # --- TEST SET: Load abnormal images from test AND sample normal images from train ---
            # 1. Load ABNORMAL images from the test folder
            test_dir = os.path.join(self.dataset_path, self.class_name, "test")
            defect_types = [d for d in sorted(os.listdir(test_dir)) if os.path.isdir(os.path.join(test_dir, d))]

            for defect_type in defect_types:
                # Skip any 'good' folder if it exists in test, as we sample from train
                if defect_type == "good":
                    continue
                
                defect_type_dir = os.path.join(test_dir, defect_type)
                image_filenames = sorted([
                    os.path.join(defect_type_dir, f)
                    for f in os.listdir(defect_type_dir)
                    if f.endswith(self.image_extension)
                ])
                image_filepaths.extend(image_filenames)
                labels.extend([1] * len(image_filenames))

            # 2. Sample NORMAL images from the training 'good' folder
            train_good_dir = os.path.join(self.dataset_path, self.class_name, "train", "good")
            normal_image_files = sorted([
                os.path.join(train_good_dir, f)
                for f in os.listdir(train_good_dir)
                if f.endswith(self.image_extension)
            ])
            
            num_to_sample = int(len(normal_image_files) * self.normal_test_sample_ratio)
            # Ensure we don't try to sample more than available
            num_to_sample = min(num_to_sample, len(normal_image_files))
            
            print(f"Sampling {num_to_sample} normal images from training set for testing.")
            
            # Use random.sample to get a random subset without replacement
            sampled_normal_files = random.sample(normal_image_files, num_to_sample)
            
            image_filepaths.extend(sampled_normal_files)
            labels.extend([0] * len(sampled_normal_files))

        assert len(image_filepaths) == len(labels), "Number of images and labels should be the same"
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
    ):
        self.image_filepaths = image_files
        self.labels = labels
        self.cropsize = cropsize

        self.transformations = T.Compose([
            T.Resize(resize, Image.Resampling.LANCZOS),
            T.CenterCrop(cropsize),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

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
        seed: int = 1024
    ):
        """
        Initializes the data manager, which finds all files, performs a
        clean train/test split, and creates the dataset objects.
        """
        random.seed(seed) # Set seed for reproducible splits

        # --- 1. Discover and partition all data files ---
        train_files, train_labels, test_files, test_labels = self._discover_and_split_files(
            dataset_path, class_name, normal_test_sample_ratio, image_extension
        )

        # --- 2. Create and store the train and test dataset attributes ---
        self.train = _BowtieSubset(
            image_files=train_files, labels=train_labels, resize=resize, cropsize=cropsize
        )
        self.test = _BowtieSubset(
            image_files=test_files, labels=test_labels, resize=resize, cropsize=cropsize
        )

        print(f"Data manager created.")
        print(f"Training set size: {len(self.train)} images")
        print(f"Testing set size: {len(self.test)} images")

    def _discover_and_split_files(self, dataset_path, class_name, ratio, ext):
        """A helper method to contain the file discovery and splitting logic."""
        
        # Define paths
        normal_dir = os.path.join(dataset_path, class_name, "train", "good")
        abnormal_dir = os.path.join(dataset_path, class_name, "test")

        # Get all normal files, shuffle, and split
        all_normal_files = sorted([os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith(ext)])
        random.shuffle(all_normal_files)
        
        split_idx = int(len(all_normal_files) * (1 - ratio))
        train_normal_files = all_normal_files[:split_idx]
        test_normal_files = all_normal_files[split_idx:]

        # Get all abnormal files
        test_abnormal_files = []
        defect_types = [d for d in sorted(os.listdir(abnormal_dir)) if os.path.isdir(os.path.join(abnormal_dir, d))]
        for defect_type in defect_types:
            if defect_type == "good": continue
            defect_dir = os.path.join(abnormal_dir, defect_type)
            test_abnormal_files.extend(sorted([os.path.join(defect_dir, f) for f in os.listdir(defect_dir) if f.endswith(ext)]))

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