import os, yaml, pickle, shutil, tarfile, glob, pandas as pd
import cv2
import albumentations
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


def get_image_resolution_from_dir(image_dir):
    """
    Get all resolution of all images in the directory
    """

    # recursively get all images
    image_paths = glob.glob(os.path.join(image_dir, "**/*.png"), recursive=True)
    widths = []
    heights = []

    for image_path in tqdm(image_paths):
        image = Image.open(image_path)
        widths.append(image.size[0])
        heights.append(image.size[1])

    print("Total images: ", len(widths))
    assert len(widths) > 0

    resolutions = {
        "max": (max(widths), max(heights)),
        "min": (min(widths), min(heights)),
        "mean": (np.mean(widths), np.mean(heights)),
        "std": (np.std(widths), np.std(heights)),
    }
    print(resolutions)

    plt.hist(widths, bins=100, alpha=0.5, label="width")
    plt.hist(heights, bins=100, alpha=0.5, label="height")
    plt.xlabel("Resolution")
    plt.ylabel("Frequency")

    dataset_name = os.path.basename(image_dir)
    if not os.path.exists("statistics"):
        os.makedirs("statistics")
    plt.savefig(f"statistics/{dataset_name}.png")
    # save resolutions to txt file
    with open(f"statistics/{dataset_name}.txt", "w") as f:
        f.write(str(resolutions))


def create_csv():
    pass


if __name__ == "__main__":
    import sys

    get_image_resolution_from_dir(sys.argv[1])
    # python ldm/data/thermal.py /Users/ducanhnguyen/Downloads/VAIS
