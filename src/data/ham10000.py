"""Loading and lesion-level splitting for HAM10000 (primary, seven-class task)."""

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_metadata() -> pd.DataFrame:
    """Load HAM10000_metadata and attach each image's file path."""
    df = pd.read_csv(config.HAM10000_METADATA_CSV)
    df["image_path"] = df["image_id"].apply(
        lambda image_id: str(config.HAM10000_IMAGE_DIR / f"{image_id}.jpg")
    )
    return df


def lesion_level_split(
    df: pd.DataFrame, val_fraction: float = 0.15, seed: int = config.RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split HAM10000 into train/validation at the lesion level.

    HAM10000's 10,015 images cover only ~7,470 unique lesions (some lesions were
    photographed more than once). Splitting by image_id would let images of the same
    lesion land in both train and validation, inflating validation performance -- so
    the split is done on unique lesion_id, stratified by each lesion's diagnosis, and
    every image belonging to a chosen lesion follows it into that split.
    """
    lesions = df.drop_duplicates(subset="lesion_id")[["lesion_id", "dx"]]
    train_lesions, val_lesions = train_test_split(
        lesions,
        test_size=val_fraction,
        stratify=lesions["dx"],
        random_state=seed,
    )
    train_df = df[df["lesion_id"].isin(train_lesions["lesion_id"])].reset_index(drop=True)
    val_df = df[df["lesion_id"].isin(val_lesions["lesion_id"])].reset_index(drop=True)
    return train_df, val_df


def load_isic2018_test() -> pd.DataFrame:
    """Load the official ISIC2018 Task 3 held-out test set (lesion-disjoint from HAM10000)."""
    df = pd.read_csv(config.ISIC2018_TEST_GROUNDTRUTH_CSV)
    df["image_path"] = df["image_id"].apply(
        lambda image_id: str(config.ISIC2018_TEST_IMAGE_DIR / f"{image_id}.jpg")
    )
    return df


def load_segmentation_mask(image_id: str) -> np.ndarray:
    """Load the ground-truth binary lesion mask for one HAM10000 image, for the
    quantitative Grad-CAM/SHAP faithfulness check (src.xai.faithfulness).

    Files are named `<image_id>_segmentation.png`, single-channel, values {0, 255}
    (confirmed by direct inspection) -- returned here as a boolean array.
    """
    path = config.HAM10000_SEGMENTATION_DIR / f"{image_id}_segmentation.png"
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"No segmentation mask found at {path}")
    return mask > 0
