from src.data import ddi


def _load(monkeypatch, tmp_path, synthetic_ddi_df):
    csv_path = tmp_path / "ddi_metadata.csv"
    synthetic_ddi_df.to_csv(csv_path, index=False)
    monkeypatch.setattr(ddi.config, "DDI_METADATA_CSV", csv_path)
    monkeypatch.setattr(ddi.config, "DDI_IMAGE_DIR", tmp_path)
    return ddi.load_metadata()


def test_load_metadata_decodes_skin_tone_group(monkeypatch, tmp_path, synthetic_ddi_df):
    df = _load(monkeypatch, tmp_path, synthetic_ddi_df)

    assert set(df["skin_tone_group"].unique()) <= {"FST_I_II", "FST_III_IV", "FST_V_VI"}
    assert (df["skin_tone_group"] == "FST_I_II").sum() == (df["skin_tone"] == 12).sum()


def test_load_metadata_decodes_binary_label(monkeypatch, tmp_path, synthetic_ddi_df):
    df = _load(monkeypatch, tmp_path, synthetic_ddi_df)

    assert set(df["binary_label"].unique()) <= {"malignant", "benign"}
    assert (df["binary_label"] == "malignant").sum() == df["malignant"].sum()


def test_split_ddi_preserves_all_rows(monkeypatch, tmp_path, synthetic_ddi_df):
    df = _load(monkeypatch, tmp_path, synthetic_ddi_df)

    train_df, val_df = ddi.split_ddi(df, val_fraction=0.15, seed=0)

    assert len(train_df) + len(val_df) == len(df)


def test_split_ddi_keeps_both_splits_non_empty(monkeypatch, tmp_path, synthetic_ddi_df):
    df = _load(monkeypatch, tmp_path, synthetic_ddi_df)

    train_df, val_df = ddi.split_ddi(df, val_fraction=0.15, seed=0)

    # The whole point of split_ddi (see agent journal, 2026-07-31) is that DDI images
    # must appear in validation too, or the skin-tone-stratified evaluation has nothing
    # to measure -- so an empty validation split would silently defeat its purpose.
    assert len(val_df) > 0
    assert len(train_df) > 0
