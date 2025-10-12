import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

CLASS_NAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


class MVTecDataset(Dataset):
    def __init__(
        self,
        dataset_path="D:/dataset/mvtec_anomaly_detection",
        class_name="bottle",
        is_train=True,
        resize=256,
        cropsize=224,
    ):
        assert class_name in CLASS_NAMES, "class_name: {}, should be in {}".format(
            class_name, CLASS_NAMES
        )
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        self.x, self.y = self.load_dataset_folder()

        self.transform_x = T.Compose(
            [
                T.Resize(resize, Image.Resampling.LANCZOS),
                T.CenterCrop(cropsize),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        x = Image.open(x).convert("RGB")
        x = self.transform_x(x)
        mask = torch.zeros([1, self.cropsize, self.cropsize])
        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = "train" if self.is_train else "test"
        x, y = [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        img_types = sorted(os.listdir(img_dir))

        for img_type in img_types:
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted(
                [
                    os.path.join(img_type_dir, f)
                    for f in os.listdir(img_type_dir)
                    if f.endswith(".png")
                ]
            )
            x.extend(img_fpath_list)

            if img_type == "good":
                y.extend([0] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))

        assert len(x) == len(y), "number of x and y should be same"
        return list(x), list(y)
    
    def print_dataset_info(self):
        print("class_name: {}, is_train: {}".format(self.class_name, self.is_train))
        print("number of data: {}".format(len(self.x)))
        
        num_good = len([1 for v in self.y if v == 0])
        num_anomaly = len([1 for v in self.y if v == 1])
        print("number of good: {}, number of anomaly: {}".format(num_good, num_anomaly))
        print("dataset path: {}".format(self.dataset_path))
        
        phase = "train" if self.is_train else "test"
        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        print("image directory: {}".format(img_dir))

