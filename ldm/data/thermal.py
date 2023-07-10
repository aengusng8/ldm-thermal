import os, yaml, pickle, shutil, tarfile, glob, pandas as pd
import cv2
import albumentations
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
    def __init__(
        self, csv_file, data_root, size=None, interpolation="bicubic", flip_p=0.5
    ):
        """
        csv file format:
        relative_file_path, class_id, class_name
        """
        self.df = pd.read_csv(csv_file, header=None)
        self.data_root = data_root

        self.size = size
        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        relative_file_path, class_id, class_name = self.df.iloc[i]
        example = dict(
            class_id=class_id,
            class_name=class_name,
        )

        image = Image.open(os.path.join(self.data_root, relative_file_path))
        if not image.mode == "RGB":
            print("convert to RGB")
            image = image.convert("RGB")

        # TODO: try random/center crop and Karras augmentations
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example


class ThermalTrain(ThermalBase):
    def __init__(self, **kwargs):
        super().__init__(csv_file="data/thermal/train.csv", **kwargs)


class ThermalValidation(ThermalBase):
    def __init__(self, **kwargs):
        super().__init__(csv_file="data/thermal/val.csv", **kwargs)
