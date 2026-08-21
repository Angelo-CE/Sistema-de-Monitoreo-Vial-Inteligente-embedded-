#!/usr/bin/env python3

import os
import re
import random
import shutil
from pathlib import Path

import cv2



#config
CCPD_BASE = Path("./datasets/CCPD2019/ccpd_base")  # Change this to your CCPD base path
OUTPUT_DIR = Path("./ccpd_yolo")

TOTAL_IMAGES = 10_000
TRAIN_RATIO = 0.80
SEED = 42

# File extensions accepted
EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ============================================================
# DIRECTORIES
# ============================================================

TRAIN_IMAGES = OUTPUT_DIR / "images" / "train"
VAL_IMAGES = OUTPUT_DIR / "images" / "val"

TRAIN_LABELS = OUTPUT_DIR / "labels" / "train"
VAL_LABELS = OUTPUT_DIR / "labels" / "val"


def create_directories():
    for directory in [
        TRAIN_IMAGES,
        VAL_IMAGES,
        TRAIN_LABELS,
        VAL_LABELS,
    ]:
        directory.mkdir(parents=True, exist_ok=True)



# CCPD BBOX PARSER

def parse_bbox(filename):
    """
    CCPD filename example:

    025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-...

    The third field contains:

        x1&y1_x2&y2

    Returns:
        x1, y1, x2, y2
    """

    parts = filename.split("-")

    if len(parts) < 4:
        raise ValueError(
            "Unexpected CCPD filename format: {}".format(filename)
        )

    bbox_part = parts[2]

    match = re.match(
        r"(\d+)&(\d+)_(\d+)&(\d+)",
        bbox_part
    )

    if match is None:
        raise ValueError(
            "Could not parse bbox from filename: {}".format(filename)
        )

    x1 = int(match.group(1))
    y1 = int(match.group(2))
    x2 = int(match.group(3))
    y2 = int(match.group(4))

    return x1, y1, x2, y2



# YOLO CONVERSION


def bbox_to_yolo(x1, y1, x2, y2, width, height):
    """
    Convert pixel bbox to YOLO format:

        class x_center y_center bbox_width bbox_height

    All coordinates normalized to [0, 1].
    """

    # Safety checks
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    bbox_width = x2 - x1
    bbox_height = y2 - y1

    x_center = x1 + bbox_width / 2.0
    y_center = y1 + bbox_height / 2.0

    return (
        x_center / width,
        y_center / height,
        bbox_width / width,
        bbox_height / height,
    )



# PROCESS IMAGE


def process_image(image_path, destination_image_dir, destination_label_dir):
    """
    Copy image and generate corresponding YOLO label.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            "Could not read image: {}".format(image_path)
        )

    height, width = image.shape[:2]

    x1, y1, x2, y2 = parse_bbox(image_path.name)

    (
        x_center,
        y_center,
        bbox_width,
        bbox_height,
    ) = bbox_to_yolo(
        x1,
        y1,
        x2,
        y2,
        width,
        height,
    )

    # Copy image
    destination_image = (
        destination_image_dir / image_path.name
    )

    shutil.copy2(
        image_path,
        destination_image
    )

    # Create YOLO label
    label_path = (
        destination_label_dir /
        (image_path.stem + ".txt")
    )

    with open(label_path, "w") as f:
        # CCPD has one license plate per image.
        # Class 0 = license_plate
        f.write(
            "0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                x_center,
                y_center,
                bbox_width,
                bbox_height,
            )
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("CCPD base:")
    print(CCPD_BASE)

    if not CCPD_BASE.exists():
        raise FileNotFoundError(
            "CCPD base directory does not exist:\n{}".format(
                CCPD_BASE
            )
        )

    create_directories()

    print("\nSearching images...")

    images = [
        p for p in CCPD_BASE.rglob("*")
        if p.is_file() and
        p.suffix.lower() in EXTENSIONS
    ]

    print("Found {} images.".format(len(images)))

    if len(images) < TOTAL_IMAGES:
        raise RuntimeError(
            "CCPD base contains fewer than {} images.".format(
                TOTAL_IMAGES
            )
        )

    # Reproducible selection
    random.seed(SEED)

    selected = random.sample(
        images,
        TOTAL_IMAGES
    )

    random.shuffle(selected)

    train_count = int(
        TOTAL_IMAGES * TRAIN_RATIO
    )

    train_images = selected[:train_count]
    val_images = selected[train_count:]

    print("\nDataset split:")
    print("Train:", len(train_images))
    print("Val:  ", len(val_images))

 
    # TRAIN
    print("\nProcessing training set...")

    for i, image_path in enumerate(train_images, 1):

        process_image(
            image_path,
            TRAIN_IMAGES,
            TRAIN_LABELS
        )

        if i % 500 == 0:
            print(
                "Train: {}/{}".format(
                    i,
                    len(train_images)
                )
            )

   
    # VALIDATION


    print("\nProcessing validation set...")

    for i, image_path in enumerate(val_images, 1):

        process_image(
            image_path,
            VAL_IMAGES,
            VAL_LABELS
        )

        if i % 500 == 0:
            print(
                "Val: {}/{}".format(
                    i,
                    len(val_images)
                )
            )

    # --------------------------------------------------------
    # DATASET YAML
    # --------------------------------------------------------

    yaml_path = OUTPUT_DIR / "data.yaml"

    with open(yaml_path, "w") as f:
        f.write(
            "path: {}\n".format(
                OUTPUT_DIR.resolve()
            )
        )
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("\n")
        f.write("names:\n")
        f.write("  0: license_plate\n")

    print("\n========================================")
    print("Dataset successfully created")
    print("========================================")

    print("Output:", OUTPUT_DIR.resolve())
    print("YAML:", yaml_path.resolve())


if __name__ == "__main__":
    main()