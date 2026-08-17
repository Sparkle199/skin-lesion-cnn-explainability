"""Tests for src.train's pure-pandas data-preparation logic (prepare_data). The
actual train() function needs TensorFlow and is exercised for real by
tests/test_integration_smoke.py on the training pod, not here."""

from src.data import ddi, ham10000
from src.train import TASKS, prepare_data


def _write_ddi_csv(monkeypatch, tmp_path, synthetic_ddi_df):
    """Route ddi.load_metadata() at real CSV/image-dir paths under tmp_path, so it runs
    its actual decoding logic (skin_tone_group, binary_label, image_path) rather than
    returning the fixture's raw, undecoded columns -- matches tests/test_ddi.py's own
    pattern, needed here since prepare_data("binary") feeds ddi.load_metadata()'s output
    straight into binary_corpus.build_binary_corpus(), which requires those decoded
    columns to already exist."""
    csv_path = tmp_path / "ddi_metadata.csv"
    synthetic_ddi_df.to_csv(csv_path, index=False)
    monkeypatch.setattr(ddi.config, "DDI_METADATA_CSV", csv_path)
    monkeypatch.setattr(ddi.config, "DDI_IMAGE_DIR", tmp_path)


def test_prepare_data_seven_class_uses_ham10000_only(monkeypatch, synthetic_ham10000_df):
    monkeypatch.setattr(ham10000, "load_metadata", lambda: synthetic_ham10000_df)

    train_df, val_df = prepare_data("seven_class")

    assert set(train_df["lesion_id"]) & set(val_df["lesion_id"]) == set()
    assert len(train_df) + len(val_df) == len(synthetic_ham10000_df)


def test_prepare_data_binary_combines_ham10000_and_ddi(
    monkeypatch, tmp_path, synthetic_ham10000_df, synthetic_ddi_df
):
    monkeypatch.setattr(ham10000, "load_metadata", lambda: synthetic_ham10000_df)
    _write_ddi_csv(monkeypatch, tmp_path, synthetic_ddi_df)

    train_df, val_df = prepare_data("binary")

    assert set(train_df["source"]) == {"ham10000", "ddi"}
    assert set(val_df["source"]) == {"ham10000", "ddi"}


def test_prepare_data_binary_ham_only_excludes_ddi(monkeypatch, synthetic_ham10000_df, synthetic_ddi_df):
    """The new HAM10000-only pretraining task must never touch DDI -- if it did, the
    zero-shot DDI evaluation (src/evaluate_ddi.py) would be leaking training data into
    its own "held-out" stress test, defeating the entire point of the sequential
    HAM10000-pretrain -> DDI-finetune methodology (see journal, 2026-08-17 entry)."""
    ddi_load_called = False

    def _tracked_ddi_load():
        nonlocal ddi_load_called
        ddi_load_called = True
        return synthetic_ddi_df

    monkeypatch.setattr(ham10000, "load_metadata", lambda: synthetic_ham10000_df)
    monkeypatch.setattr(ddi, "load_metadata", _tracked_ddi_load)

    train_df, val_df = prepare_data("binary_ham_only")

    assert not ddi_load_called, "binary_ham_only must not load DDI at all"
    assert set(train_df["source"]) == {"ham10000"}
    assert set(val_df["source"]) == {"ham10000"}
    assert set(train_df["binary_label"]) <= {"benign", "malignant"}


def test_prepare_data_binary_ham_only_matches_joint_binary_columns(
    monkeypatch, tmp_path, synthetic_ham10000_df, synthetic_ddi_df
):
    """binary_ham_only's DataFrame must be structurally identical (same columns) to
    the HAM10000 half of the joint "binary" corpus, since both feed the same
    make_dataset()/evaluate_run.py code paths."""
    monkeypatch.setattr(ham10000, "load_metadata", lambda: synthetic_ham10000_df)
    _write_ddi_csv(monkeypatch, tmp_path, synthetic_ddi_df)

    _, val_binary = prepare_data("binary")
    _, val_ham_only = prepare_data("binary_ham_only")

    assert list(val_ham_only.columns) == list(val_binary.columns)


def test_binary_ham_only_registered_in_tasks():
    assert "binary_ham_only" in TASKS
    assert TASKS["binary_ham_only"]["classes"] == ["benign", "malignant"]
