import os, yaml, pickle, shutil, tarfile, glob, sys, pandas as pd
import cv2
import albumentations as A
import matplotlib.pyplot as plt
import PIL
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
from torchvision import transforms


class ThermalBase(Dataset):
    """
    Pytorch Dataset for thermal generation
    """

    def __init__(
        self, csv_file, data_root, size=128, interpolation="bicubic", flip_p=0.5
    ):
        """
        csv file format:
        relative_file_path, img_type, class_id, class_name, ...
        """
        self.df = pd.read_csv(os.path.join(data_root, csv_file))
        self.data_root = data_root

        self.size = size
        self.interpolation = {
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]

        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        relative_file_path, img_type, class_id, class_name = self.df.iloc[i][:4]
        example = dict(
            class_id=class_id,
            class_name=class_name,
            img_type=img_type,
        )

        img = Image.open(os.path.join(self.data_root, relative_file_path))

        # image = self.online_augment(image=image)["image"]
        img = self.normal_augment(img, type=img_type)
        example["image"] = img

        return example

    def normal_augment(self, img, type):
        # TODO: try random/center crop and Karras augmentations
        img = img.convert("L")  # to gray scale
        img = img.resize((self.size, self.size), resample=self.interpolation)

        img = self.flip(img)
        img = np.array(img).astype(np.uint8)
        img = (img / 127.5 - 1.0).astype(np.float32)
        return img


class ThermalTrain(ThermalBase):
    def __init__(self, **kwargs):
        super().__init__(csv_file="train.csv", **kwargs)


class ThermalValidation(ThermalBase):
    def __init__(self, **kwargs):
        super().__init__(csv_file="val.csv", **kwargs)


if __name__ == "__main__":
    # get_image_resolution_from_dir(sys.argv[1])

    data = ThermalTrain(
        data_root="/Users/ducanhnguyen/Desktop/deep_learning_projects/datasets/ThermalGen_ds",
    )
    print(type(data[0]["image"]))
