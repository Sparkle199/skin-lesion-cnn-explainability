"""Shared fixtures. Tests here use synthetic data shaped like the real, directly-
inspected HAM10000_metadata / ddi_metadata.csv / segmentation-mask schemas -- they do
not require the multi-GB dataset archives to be extracted."""

import numpy as np
import pandas as pd
import pytest

from src import config


@pytest.fixture
def synthetic_ham10000_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_lesions = 200
    classes = config.SEVEN_CLASSES
    # Weighted to loosely mirror the real ~67%-majority-class imbalance.
    lesion_dx = rng.choice(classes, size=n_lesions, p=[0.03, 0.05, 0.11, 0.01, 0.11, 0.67, 0.02])

    rows = []
    for i in range(n_lesions):
        n_images = rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        for j in range(n_images):
            rows.append(
                {
                    "lesion_id": f"HAM_{i:04d}",
                    "image_id": f"ISIC_{i:04d}_{j}",
                    "dx": lesion_dx[i],
                    "dx_type": "histo",
                    "age": 50.0,
                    "sex": "male",
                    "localization": "back",
                    "dataset": "vidir_modern",
                }
            )
    df = pd.DataFrame(rows)
    df["image_path"] = df["image_id"].apply(lambda x: f"/fake/{x}.jpg")
    return df


@pytest.fixture
def synthetic_ddi_df(tmp_path) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 100
    df = pd.DataFrame(
        {
            "DDI_ID": range(n),
            "DDI_file": [f"{i:06d}.png" for i in range(n)],
            "skin_tone": rng.choice([12, 34, 56], size=n),
            "malignant": rng.choice([True, False], size=n, p=[0.26, 0.74]),
            "disease": "melanocytic-nevi",
        }
    )
    return df


@pytest.fixture
def synthetic_lesion_mask() -> np.ndarray:
    """A small boolean mask with a clear foreground/background split, standing in for
    a real HAM10000 segmentation mask (confirmed format: single-channel, {0, 255})."""
    rng = np.random.default_rng(2)
    mask = np.zeros((60, 80), dtype=bool)
    mask[15:45, 20:60] = True
    return mask
