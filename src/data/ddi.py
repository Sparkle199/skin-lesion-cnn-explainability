"""Loading for DDI (secondary, binary malignant/benign task only -- see spec)."""

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_metadata() -> pd.DataFrame:
    """Load ddi_metadata.csv and attach each image's file path and skin-tone group label.

    DDI's `skin_tone` column is three grouped Fitzpatrick bands (12/34/56), not the full
    six-point scale -- decoded here via config.DDI_SKIN_TONE_LABELS for readability in
    the stratified evaluation. `malignant` is already a native boolean; DDI's 78
    `disease` values are not used for the binary task and are not read from here.
    """
    df = pd.read_csv(config.DDI_METADATA_CSV)
    df["image_path"] = df["DDI_file"].apply(lambda filename: str(config.DDI_IMAGE_DIR / filename))
    df["skin_tone_group"] = df["skin_tone"].map(config.DDI_SKIN_TONE_LABELS)
    df["binary_label"] = df["malignant"].map({True: "malignant", False: "benign"})
    return df


def split_ddi(
    df: pd.DataFrame, val_fraction: float = 0.15, seed: int = config.RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DDI into train/validation, stratified by skin-tone group and malignancy.

    DDI has no lesion_id-style grouping to worry about (each row is one image), but it
    still needs its own split: without one, all 656 images would fall into training and
    none would be available for the skin-tone-stratified evaluation the spec calls for.
    Stratifying on the combination of skin_tone_group and binary_label keeps both
    balanced across the two splits, rather than only balancing one at the other's
    expense.
    """
    strata = df["skin_tone_group"] + "_" + df["binary_label"]
    train_df, val_df = train_test_split(
        df, test_size=val_fraction, stratify=strata, random_state=seed
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
