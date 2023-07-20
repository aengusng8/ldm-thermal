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


def create_csv(
    data_root="../datasets/ThermalGen_ds",
    img_type_folders=dict(RGB_v2="rgb", Simulation_v2="sim", ThermalReal="real"),
    class_folders=dict(Civilian="civilian", Military="military"),
    class_mapping=dict(civilian=0, military=1),
):
    """
    Create csv file for thermal generation:
    1. raw CSV
    2. training CSV

    With format like:
    relative_file_path, img_type, class_id, class_name
    """
    df = pd.DataFrame(
        columns=["relative_file_path", "img_type", "class_id", "class_name"]
    )
    assert os.path.exists(data_root)
    # create raw CSV
    for img_type_folder, img_type in img_type_folders.items():
        for class_folder, class_name in class_folders.items():
            assert os.path.exists(
                os.path.join(data_root, img_type_folder, class_folder)
            )
            # get all files in the folder (recursive) regardless of the extension
            files = glob.glob(
                os.path.join(data_root, img_type_folder, class_folder, "**/*.*"),
                recursive=True,
            )
            # get relative path
            files = [os.path.relpath(file, data_root) for file in files]
            class_id = class_mapping[class_name]
            key_to_split = f"{class_id}_{img_type}"

            # concatenate to df
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "relative_file_path": files,
                            "img_type": img_type,
                            "class_id": class_id,
                            "class_name": class_name,
                            "key_to_split": key_to_split,
                        }
                    ),
                ]
            )
    # save raw CSV
    print("we have ", len(df), " images")
    df.to_csv(os.path.join(data_root, "raw.csv"), index=False)


def preprocess_csv(csv_path="../datasets/ThermalGen_ds/raw.csv", train_ratio=0.8):
    """
    Preprocess raw CSV file to create training and validation CSV file
    """
    df = pd.read_csv(csv_path)

    # split by key_to_split
    train_df = pd.DataFrame(columns=df.columns)
    val_df = pd.DataFrame(columns=df.columns)

    print("splitting by key_to_split: ", df["key_to_split"].unique())
    for key in df["key_to_split"].unique():
        df_key = df[df["key_to_split"] == key]
        n_head = int(len(df_key) * train_ratio)
        n_tail = len(df_key) - n_head

        train_df = pd.concat([train_df, df_key.head(n_head)])
        val_df = pd.concat([val_df, df_key.tail(n_tail)])

    print("total images: ", len(df))
    print("train images: ", len(train_df), "(%.3f)" % (len(train_df) / len(df)))
    print("val images: ", len(val_df), "(%.3f)" % (len(val_df) / len(df)))

    train_df.to_csv(os.path.join(os.path.dirname(csv_path), "train.csv"), index=False)
    val_df.to_csv(os.path.join(os.path.dirname(csv_path), "val.csv"), index=False)


# def preprocess_image(csv_path):
#     def offline_augment(type, cfg=None):
#         # just convert to gray image
#         return albumentations.Compose([albumentations.ToGray(p=1.0)], p=1.0)


if __name__ == "__main__":
    create_csv()
    preprocess_csv()
