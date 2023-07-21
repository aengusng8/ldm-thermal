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
    img_type_folders=dict(RGB_v2="rgb", ThermalReal="real"),
    class_folders=dict(Civilian="civilian", Military="military"),
    class_mapping=dict(civilian=0, military=1),
):
    """
    Create csv file for thermal generation:
    1. raw CSV
    2. training CSV

    With format like:
    relative_path, img_type, class_id, class_name
    """
    print("creating csv file...")
    df = pd.DataFrame(columns=["relative_path", "img_type", "class_id", "class_name"])
    assert os.path.exists(data_root)
    # 1. Real images + RGB images
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
            # key_to_split = f"{class_id}_{img_type}"

            # concatenate to df
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "relative_path": files,
                            "img_type": img_type,
                            "class_id": class_id,
                            "class_name": class_name,
                            "is_img": 1,
                            # "key_to_split": key_to_split,
                        }
                    ),
                ]
            )

    # 2. Simulated images: add sub-classes to raw CSV, and images to sim CSV
    sim_folder = "Simulation_v2"
    sim_df = pd.DataFrame(columns=list(df.columns) + ["sub_class"])

    for class_folder, class_name in class_folders.items():
        assert os.path.exists(os.path.join(data_root, sim_folder, class_folder))
        sub_classes = glob.glob(os.path.join(data_root, sim_folder, class_folder, "*"))
        sub_classes = [os.path.basename(sub_class) for sub_class in sub_classes]
        class_id = class_mapping[class_name]

        # add sub-folders to df
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "relative_path": sub_classes,
                        "img_type": "sim",
                        "class_id": class_id,
                        "class_name": class_name,
                        "is_img": 0,
                    }
                ),
            ]
        )

        # get all files in the folder (recursive) regardless of the extension
        files = glob.glob(os.path.join(data_root, sim_folder, class_folder, "**/*.*"))
        # get relative path
        files = [os.path.relpath(file, data_root) for file in files]
        sub_classes = [files.split("/")[-2] for files in files]
        class_id = class_mapping[class_name]

        # concatenate to sim_df
        sim_df = pd.concat(
            [
                sim_df,
                pd.DataFrame(
                    {
                        "relative_path": files,
                        "img_type": "sim",
                        "class_id": class_id,
                        "class_name": class_name,
                        "is_img": 1,
                        "sub_class": sub_classes,
                    }
                ),
            ]
        )

    # save raw CSV
    print("we have ", len(df), " images in a epoch for train and val")
    print("    + real: ", len(df[df["img_type"] == "real"]))
    print("    + rgb: ", len(df[df["img_type"] == "rgb"]))
    print("    + sim (sub-classes): ", len(df[df["img_type"] == "sim"]))
    print("and sim (images): ", len(sim_df))
    df.to_csv(os.path.join(data_root, "raw.csv"), index=False)
    sim_df.to_csv(os.path.join(data_root, "sim.csv"), index=False)


def preprocess_csv(csv_path="../datasets/ThermalGen_ds/raw.csv", train_ratio=0.8):
    """
    Preprocess raw CSV file to create training and validation CSV file
    """
    print("\npreprocessing csv file...")
    df = pd.read_csv(csv_path)

    train_df = pd.DataFrame(columns=df.columns)
    val_df = pd.DataFrame(columns=df.columns)

    # only use a subset of "real" to validate
    real_df = df[df["img_type"] == "real"]
    print("real: ", len(real_df))
    n_head = int(len(real_df) * train_ratio)
    n_tail = len(real_df) - n_head
    train_df = pd.concat([train_df, real_df.head(n_head)])
    val_df = pd.concat([val_df, real_df.tail(n_tail)])

    # use other images, and a subset of "real" to train
    df_key = df[df["img_type"] != "real"]
    train_df = pd.concat([train_df, df_key])

    print("total: ", len(df))
    print(
        f"    + val images (use rgb only): {len(val_df)} ({(round(len(val_df) / len(real_df) * 100))}%)"
    )
    val_df.to_csv(os.path.join(os.path.dirname(csv_path), "val.csv"), index=False)

    print("    + train images before upsampling: ", len(train_df))

    # upsample the real images
    def upsample_real(train_df, must_real_ratio=0.8):
        print("upsampling real images...")
        # 1. calculate the ratio between real and the whole dataset
        n_real = len(train_df[train_df["img_type"] == "real"])
        current_real_ratio = n_real / len(train_df)
        print("   + current 'real' ratio: ", round(current_real_ratio * 100, 3), "%")
        scale = n_real * int(
            (must_real_ratio - current_real_ratio) / current_real_ratio
        )
        print("   + scale: ", scale)
        # 2. upsample the real images
        real_df = train_df[train_df["img_type"] == "real"]
        real_df = pd.concat([real_df] * scale)
        train_df = pd.concat([train_df, real_df])
        # 3. shuffle
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        n_real = len(train_df[train_df["img_type"] == "real"])
        print("after upsampling:")
        print("   + real images: ", n_real)
        print("   + train images  ", len(train_df))
        print(
            "   + current 'real' ratio: ",
            round(n_real / len(train_df) * 100, 3),
            "%",
        )
        print
        return train_df

    train_df = upsample_real(train_df, must_real_ratio=0.7)

    train_df.to_csv(os.path.join(os.path.dirname(csv_path), "train.csv"), index=False)


if __name__ == "__main__":
    create_csv()
    preprocess_csv()
