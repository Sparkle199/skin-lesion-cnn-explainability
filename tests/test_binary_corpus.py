from src import config
from src.data.binary_corpus import build_binary_corpus, relabel_ham10000_binary
from src.data.ddi import load_metadata as load_ddi_metadata, split_ddi


def test_relabel_ham10000_binary_matches_expected_mapping(synthetic_ham10000_df):
    relabelled = relabel_ham10000_binary(synthetic_ham10000_df)

    expected = synthetic_ham10000_df["dx"].apply(
        lambda dx: "malignant" if dx in config.MALIGNANT_CLASSES else "benign"
    )
    assert (relabelled["binary_label"].values == expected.values).all()


def test_relabel_ham10000_binary_marks_source_and_no_skin_tone(synthetic_ham10000_df):
    relabelled = relabel_ham10000_binary(synthetic_ham10000_df)

    assert (relabelled["source"] == "ham10000").all()
    assert relabelled["skin_tone_group"].isna().all()


def test_build_binary_corpus_combines_both_sources(monkeypatch, tmp_path, synthetic_ham10000_df, synthetic_ddi_df):
    csv_path = tmp_path / "ddi_metadata.csv"
    synthetic_ddi_df.to_csv(csv_path, index=False)
    monkeypatch.setattr(config, "DDI_METADATA_CSV", csv_path)
    monkeypatch.setattr(config, "DDI_IMAGE_DIR", tmp_path)

    ddi_df = load_ddi_metadata()
    ddi_train, _ = split_ddi(ddi_df, val_fraction=0.15, seed=0)

    combined = build_binary_corpus(synthetic_ham10000_df, ddi_train)

    assert set(combined["source"].unique()) == {"ham10000", "ddi"}
    assert len(combined) == len(synthetic_ham10000_df) + len(ddi_train)
    assert list(combined.columns) == ["image_path", "binary_label", "source", "skin_tone_group", "image_id"]


def test_build_binary_corpus_keeps_ham10000_image_id_for_mask_lookup(
    monkeypatch, tmp_path, synthetic_ham10000_df, synthetic_ddi_df
):
    """image_id must survive into the combined corpus for HAM10000 rows -- otherwise
    src.xai.faithfulness has no way to look up a HAM10000-derived row's ground-truth
    segmentation mask once it's mixed into the binary task's corpus with DDI."""
    csv_path = tmp_path / "ddi_metadata.csv"
    synthetic_ddi_df.to_csv(csv_path, index=False)
    monkeypatch.setattr(config, "DDI_METADATA_CSV", csv_path)
    monkeypatch.setattr(config, "DDI_IMAGE_DIR", tmp_path)

    ddi_df = load_ddi_metadata()
    ddi_train, _ = split_ddi(ddi_df, val_fraction=0.15, seed=0)

    combined = build_binary_corpus(synthetic_ham10000_df, ddi_train)

    ham_rows = combined[combined["source"] == "ham10000"]
    ddi_rows = combined[combined["source"] == "ddi"]
    assert ham_rows["image_id"].notna().all()
    assert set(ham_rows["image_id"]) == set(synthetic_ham10000_df["image_id"])
    assert ddi_rows["image_id"].isna().all()
