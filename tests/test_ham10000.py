import cv2

from src import config
from src.data.ham10000 import lesion_level_split, load_segmentation_mask


def test_lesion_level_split_has_no_lesion_overlap(synthetic_ham10000_df):
    train_df, val_df = lesion_level_split(synthetic_ham10000_df, val_fraction=0.15, seed=0)

    overlap = set(train_df["lesion_id"]) & set(val_df["lesion_id"])
    assert not overlap, f"lesions leaked across splits: {overlap}"


def test_lesion_level_split_preserves_all_images(synthetic_ham10000_df):
    train_df, val_df = lesion_level_split(synthetic_ham10000_df, val_fraction=0.15, seed=0)

    assert len(train_df) + len(val_df) == len(synthetic_ham10000_df)


def test_lesion_level_split_keeps_majority_class_proportion_close(synthetic_ham10000_df):
    train_df, val_df = lesion_level_split(synthetic_ham10000_df, val_fraction=0.15, seed=0)

    # `nv` is the synthetic majority class (~67%, mirroring the real dataset). A
    # lesion-stratified split should keep its share close in both splits; an
    # unstratified split on a small (200-lesion) sample could easily drift far more.
    overall_share = (synthetic_ham10000_df["dx"] == "nv").mean()
    train_share = (train_df["dx"] == "nv").mean()
    val_share = (val_df["dx"] == "nv").mean()

    assert abs(train_share - overall_share) < 0.15
    assert abs(val_share - overall_share) < 0.15


def test_load_segmentation_mask_reads_binary_png(monkeypatch, tmp_path, synthetic_lesion_mask):
    monkeypatch.setattr(config, "HAM10000_SEGMENTATION_DIR", tmp_path)
    mask_path = tmp_path / "ISIC_0000001_segmentation.png"
    cv2.imwrite(str(mask_path), (synthetic_lesion_mask * 255).astype("uint8"))

    loaded = load_segmentation_mask("ISIC_0000001")

    assert loaded.dtype == bool
    assert loaded.shape == synthetic_lesion_mask.shape
    assert (loaded == synthetic_lesion_mask).all()


def test_load_segmentation_mask_missing_file_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "HAM10000_SEGMENTATION_DIR", tmp_path)

    try:
        load_segmentation_mask("does_not_exist")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
