"""Builds the combined HAM10000 + DDI corpus for the secondary binary task.

Per spec/specification.md: DDI's 78 disease labels do not map cleanly onto HAM10000's
seven classes, so DDI is not merged into the primary seven-class task. Instead, both
datasets are relabelled/read as a shared binary malignant/benign target and combined
here, keeping the two source datasets and DDI's skin-tone groups identifiable so the
secondary task's evaluation can be stratified by skin tone.
"""

import pandas as pd

from src import config


def relabel_ham10000_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Derive a malignant/benign label from HAM10000's existing seven-class `dx` column.

    malignant = {akiec, bcc, mel}; benign = {bkl, df, nv, vasc}. This grouping follows
    common HAM10000 literature practice but was not itself specified in the project
    proposal -- confirm it with the supervisor before training (akiec is a borderline,
    precancerous case some papers group differently).
    """
    df = df.copy()
    df["binary_label"] = df["dx"].apply(
        lambda dx: "malignant" if dx in config.MALIGNANT_CLASSES else "benign"
    )
    df["source"] = "ham10000"
    df["skin_tone_group"] = None  # HAM10000 carries no skin-tone label.
    return df


_CORPUS_COLUMNS = ["image_path", "binary_label", "source", "skin_tone_group", "image_id"]


def build_binary_corpus(ham10000_df: pd.DataFrame, ddi_df: pd.DataFrame) -> pd.DataFrame:
    """Combine relabelled HAM10000 with the full DDI dataset for the binary task.

    ham10000_df is expected already split (train or validation) via
    src.data.ham10000.lesion_level_split, so this should be called separately per split
    to avoid leaking HAM10000 lesions across the binary task's train/validation sets.

    Keeps `image_id` (HAM10000-only; None for DDI rows, which have no equivalent
    identifier and no segmentation mask to look up) so that HAM10000-derived rows in the
    combined corpus can still be matched to their ground-truth lesion mask for the
    faithfulness check in src.xai.faithfulness -- an earlier version of this function
    dropped image_id entirely, which silently made that lookup impossible.
    """
    ham_binary = relabel_ham10000_binary(ham10000_df)[_CORPUS_COLUMNS]

    ddi_binary = ddi_df.copy()
    ddi_binary["source"] = "ddi"
    ddi_binary["image_id"] = None
    ddi_binary = ddi_binary[_CORPUS_COLUMNS]

    return pd.concat([ham_binary, ddi_binary], ignore_index=True)
