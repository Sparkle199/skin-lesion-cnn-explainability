import json

from src.trade_off import build_comparison_table, load_results, rank_by, summarise

# Shaped exactly like src.evaluate_run.evaluate's output for each (architecture, task).
SEVEN_CLASS_RESULT = {
    "architecture": "resnet50",
    "task": "seven_class",
    "validation": {"accuracy": 0.85, "cohens_kappa": 0.78},
    "isic2018_test": {"accuracy": 0.80},
    "faithfulness": {"mean_iou": 0.42, "mean_dice": 0.55, "n": 30},
}
BINARY_RESULT = {
    "architecture": "resnet50",
    "task": "binary",
    "validation": {"accuracy": 0.90},
    "skin_tone_stratified": {
        "FST_I_II": {"n": 20, "accuracy": 0.92},
        "FST_III_IV": {"n": 18, "accuracy": 0.88},
        "FST_V_VI": {"n": 17, "accuracy": 0.83},
    },
}


def _write_results(tmp_path, entries: dict) -> None:
    for (architecture, task), payload in entries.items():
        (tmp_path / f"{architecture}_{task}.json").write_text(json.dumps(payload))


def test_load_results_skips_missing_files(tmp_path):
    _write_results(tmp_path, {("resnet50", "seven_class"): SEVEN_CLASS_RESULT})

    results = load_results(tmp_path)

    assert set(results.keys()) == {("resnet50", "seven_class")}


def test_build_comparison_table_fills_missing_architectures_with_none(tmp_path):
    _write_results(tmp_path, {("resnet50", "seven_class"): SEVEN_CLASS_RESULT})
    results = load_results(tmp_path)

    rows = build_comparison_table(results)

    by_arch = {row["architecture"]: row for row in rows}
    assert by_arch["resnet50"]["seven_class_accuracy"] == 0.85
    assert by_arch["efficientnetb4"]["seven_class_accuracy"] is None
    assert by_arch["vgg16"]["seven_class_accuracy"] is None


def test_build_comparison_table_reads_faithfulness_and_stratified_fields(tmp_path):
    _write_results(
        tmp_path,
        {
            ("resnet50", "seven_class"): SEVEN_CLASS_RESULT,
            ("resnet50", "binary"): BINARY_RESULT,
        },
    )
    results = load_results(tmp_path)

    rows = build_comparison_table(results)
    resnet_row = next(r for r in rows if r["architecture"] == "resnet50")

    assert resnet_row["faithfulness_iou"] == 0.42
    assert resnet_row["binary_accuracy"] == 0.90
    # spread = max(0.92, 0.88, 0.83) - min(...) = 0.92 - 0.83
    assert abs(resnet_row["binary_skin_tone_accuracy_spread"] - (0.92 - 0.83)) < 1e-9


def test_rank_by_skips_none_values():
    rows = [
        {"architecture": "resnet50", "seven_class_accuracy": 0.85},
        {"architecture": "vgg16", "seven_class_accuracy": None},
        {"architecture": "efficientnetb4", "seven_class_accuracy": 0.90},
    ]

    ranking = rank_by(rows, "seven_class_accuracy")

    assert ranking == ["efficientnetb4", "resnet50"]


def test_summarise_flags_trade_off_when_rankings_disagree():
    rows = [
        {"architecture": "resnet50", "seven_class_accuracy": 0.90, "faithfulness_iou": 0.30},
        {"architecture": "vgg16", "seven_class_accuracy": 0.80, "faithfulness_iou": 0.50},
        {"architecture": "efficientnetb4", "seven_class_accuracy": 0.85, "faithfulness_iou": 0.40},
    ]

    summary = summarise(rows)

    assert "resnet50" in summary and "vgg16" in summary
    assert "Trade-off" in summary
    assert "no single architecture" in summary


def test_summarise_notes_single_leader_when_rankings_agree():
    rows = [
        {"architecture": "resnet50", "seven_class_accuracy": 0.90, "faithfulness_iou": 0.55},
        {"architecture": "vgg16", "seven_class_accuracy": 0.80, "faithfulness_iou": 0.40},
    ]

    summary = summarise(rows)

    assert "leads on both accuracy and faithfulness" in summary


def test_summarise_handles_no_results_gracefully():
    rows = [
        {"architecture": "resnet50", "seven_class_accuracy": None, "faithfulness_iou": None},
    ]

    summary = summarise(rows)

    assert "Not enough results" in summary
