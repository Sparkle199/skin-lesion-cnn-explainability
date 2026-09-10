"""Selects 5 held-out example images from each of this project's three held-out
evaluation sets (HAM10000 validation split, ISIC2018 Task 3 test set, DDI held-out
split), plus a dedicated 5-image set of dark-skinned (Fitzpatrick V-VI) DDI images,
and copies them into test_samples/ for manually exercising the Streamlit demo's
Upload mode with images no model has been trained on.

Usage: python scripts/build_streamlit_test_samples.py
"""

import shutil
from pathlib import Path

import pandas as pd

from src import config
from src.data import ham10000

OUT_DIR = Path("test_samples")
N_PER_SOURCE = 5


def _one_per_group(df: pd.DataFrame, group_cols, n: int, seed: int) -> pd.DataFrame:
    """One row sampled from each distinct value (or combination of values) of
    group_cols, then trimmed/padded to exactly n rows -- a plain loop rather than
    groupby().apply(), which in some pandas versions silently drops the grouping
    column(s) from the result when the applied function returns a subset of the
    group."""
    keys = list(df.groupby(list(group_cols) if isinstance(group_cols, list) else [group_cols]).groups.keys())
    rng_order = pd.Series(keys).sample(frac=1, random_state=seed).tolist()
    picked_rows = []
    cols = group_cols if isinstance(group_cols, list) else [group_cols]
    for key in rng_order:
        key_tuple = key if isinstance(key, tuple) else (key,)
        mask = pd.Series(True, index=df.index)
        for col, val in zip(cols, key_tuple):
            mask &= df[col] == val
        picked_rows.append(df[mask].sample(1, random_state=seed))
        if len(picked_rows) >= n:
            break
    return pd.concat(picked_rows, ignore_index=True) if picked_rows else df.iloc[0:0]


def pick_ham10000_validation(n: int) -> pd.DataFrame:
    """Exact same lesion-level split used throughout training/evaluation
    (val_fraction=0.15, seed=config.RANDOM_SEED), so this is genuinely the same
    validation split every result in the dissertation was scored on. One image per
    distinct diagnosis where possible, for a varied sample."""
    df = ham10000.load_metadata()
    _, val_df = ham10000.lesion_level_split(df)
    return _one_per_group(val_df, "dx", n, config.RANDOM_SEED)


def pick_isic2018_test(n: int) -> pd.DataFrame:
    df = ham10000.load_isic2018_test()
    return _one_per_group(df, "dx", n, config.RANDOM_SEED)


def pick_ddi_holdout(n: int) -> pd.DataFrame:
    """Reads the pre-materialised 99-image DDI held-out split already extracted to
    data/ddi_holdout_for_testing/ (the same split used for every DDI held-out result
    in Chapter 4), and picks a spread across malignant/benign and skin-tone group.
    Uses the file already copied under FST_*/ rather than the metadata CSV's own
    image_path column, since that column points back at data/raw/ddi with
    Windows-style separators baked in from wherever it was generated."""
    meta_path = Path("data/ddi_holdout_for_testing/ddi_holdout_metadata.csv")
    df = pd.read_csv(meta_path)
    df["holdout_path"] = df.apply(
        lambda r: str(Path("data/ddi_holdout_for_testing") / r["skin_tone_group"] / r["DDI_file"]),
        axis=1,
    )
    return _one_per_group(df, ["binary_label", "skin_tone_group"], n, config.RANDOM_SEED)


def pick_dark_skin(n: int, exclude_files: set[str]) -> pd.DataFrame:
    """A dedicated set of FST V-VI (darkest Fitzpatrick band) images from the DDI
    held-out split -- the specific group the DDI fairness evaluation in Chapter 4
    exists to test, and the one most affected by the zero-shot malignant-recall
    collapse discussed in Chapter 4/5. Skews toward malignant cases since they are
    both the clinically-relevant case and the rarer of the two in this band (7 of 31
    FST V-VI images are malignant), and drops anything already copied to
    ddi_holdout/ so the two folders don't overlap."""
    meta_path = Path("data/ddi_holdout_for_testing/ddi_holdout_metadata.csv")
    df = pd.read_csv(meta_path)
    df["holdout_path"] = df.apply(
        lambda r: str(Path("data/ddi_holdout_for_testing") / r["skin_tone_group"] / r["DDI_file"]),
        axis=1,
    )
    dark = df[(df["skin_tone_group"] == "FST_V_VI") & (~df["DDI_file"].isin(exclude_files))]

    malignant = dark[dark["binary_label"] == "malignant"]
    benign = dark[dark["binary_label"] == "benign"]
    n_malignant = min(len(malignant), max(2, n // 2))
    picked_malignant = _one_per_group(malignant, "disease", n_malignant, config.RANDOM_SEED)
    n_benign = n - len(picked_malignant)
    picked_benign = _one_per_group(benign, "disease", n_benign, config.RANDOM_SEED)
    return pd.concat([picked_malignant, picked_benign], ignore_index=True)


def copy_rows(rows: pd.DataFrame, src_path_col: str, label_cols: list[str], dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in rows.itertuples():
        src_path = Path(getattr(row, src_path_col))
        if not src_path.is_file():
            print(f"  SKIP (file not found): {src_path}")
            continue
        dest_path = dest / src_path.name
        shutil.copyfile(src_path, dest_path)
        manifest.append({col: getattr(row, col) for col in label_cols} | {"file": src_path.name})
    return manifest


def main():
    OUT_DIR.mkdir(exist_ok=True)
    all_manifest = {}

    print("HAM10000 validation split...")
    ham_rows = pick_ham10000_validation(N_PER_SOURCE)
    all_manifest["ham10000_validation"] = copy_rows(
        ham_rows, "image_path", ["image_id", "dx"], OUT_DIR / "ham10000_validation"
    )

    print("ISIC2018 test set...")
    isic_rows = pick_isic2018_test(N_PER_SOURCE)
    all_manifest["isic2018_test"] = copy_rows(
        isic_rows, "image_path", ["image_id", "dx"], OUT_DIR / "isic2018_test"
    )

    print("DDI held-out split...")
    ddi_rows = pick_ddi_holdout(N_PER_SOURCE)
    all_manifest["ddi_holdout"] = copy_rows(
        ddi_rows,
        "holdout_path",
        ["DDI_file", "disease", "binary_label", "skin_tone_group"],
        OUT_DIR / "ddi_holdout",
    )

    print("DDI held-out, dark skin (FST V-VI) only...")
    already_used = {r["DDI_file"] for r in all_manifest["ddi_holdout"]}
    dark_rows = pick_dark_skin(N_PER_SOURCE, already_used)
    all_manifest["ddi_holdout_dark_skin"] = copy_rows(
        dark_rows,
        "holdout_path",
        ["DDI_file", "disease", "binary_label", "skin_tone_group"],
        OUT_DIR / "ddi_holdout_dark_skin",
    )

    manifest_lines = ["# Streamlit test samples: source, file, true label\n"]
    for source, rows in all_manifest.items():
        manifest_lines.append(f"\n## {source} ({len(rows)} images)\n")
        for r in rows:
            manifest_lines.append(f"- {r}\n")
    (OUT_DIR / "MANIFEST.md").write_text("".join(manifest_lines), encoding="utf-8")

    for source, rows in all_manifest.items():
        print(f"{source}: {len(rows)} images copied")
    print(f"\nDone. See {OUT_DIR / 'MANIFEST.md'} for true labels.")


if __name__ == "__main__":
    main()
