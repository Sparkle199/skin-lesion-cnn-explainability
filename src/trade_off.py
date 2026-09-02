"""Trade-off analysis: aggregates src.evaluate_run's per-(architecture, task) JSON
results and produces the accuracy-vs-interpretability comparison called for in the
proposal's Objective 5 / Trade-off Analysis phase.

Reports rankings on each dimension and flags where they disagree, rather than picking a
single "winner" via an opaque combined score: selecting the optimal model is a
judgement call for the student and supervisor to make from the evidence, not something
this script should decide unilaterally (per governance: delegate tasks, not judgement).

Usage: python -m src.trade_off --results-dir results/
"""

import argparse
import json
from pathlib import Path

ARCHITECTURES = ["resnet50", "efficientnetb4", "vgg16"]
TASKS = ["seven_class", "binary"]


def load_results(results_dir: Path) -> dict:
    """Load all available src.evaluate_run JSON outputs, keyed by (architecture, task).

    Missing files are skipped, not errored, since not all six (architecture x task)
    runs may exist yet -- the comparison table then shows gaps explicitly via None
    rather than only working once every run is complete.

    For the seven-class task, a "_faithfulness150" result, where present, is loaded
    alongside the base result and preferred for the faithfulness columns: Chapter 4
    (Table 4.8) established that the original n=30 Grad-CAM sample moved substantially
    once re-run at n=150, so this table should not report the n=30 figure as if it
    were the reliable one when a larger-sample figure is available.
    """
    results = {}
    for architecture in ARCHITECTURES:
        for task in TASKS:
            path = results_dir / f"{architecture}_{task}.json"
            if path.exists():
                with open(path) as f:
                    results[(architecture, task)] = json.load(f)
            if task == "seven_class":
                large_n_path = results_dir / f"{architecture}_{task}_faithfulness150.json"
                if large_n_path.exists():
                    with open(large_n_path) as f:
                        results[(architecture, task, "faithfulness150")] = json.load(f)
    return results


def _skin_tone_spread(binary_results: dict) -> float | None:
    """Max-minus-min accuracy across skin-tone groups in the binary task: a simple,
    interpretable measure of how (in)consistent a model is across skin tones (smaller
    is more consistent). None if stratified results aren't present in this run."""
    strat = binary_results.get("skin_tone_stratified")
    if not strat:
        return None
    accuracies = [group["accuracy"] for group in strat.values()]
    return max(accuracies) - min(accuracies) if accuracies else None


def build_comparison_table(results: dict) -> list[dict]:
    """One row per architecture with the accuracy and interpretability figures needed
    for the trade-off comparison. Missing runs show as None rather than being silently
    omitted, so gaps in what's been run stay visible in the output."""
    rows = []
    for architecture in ARCHITECTURES:
        seven = results.get((architecture, "seven_class"))
        seven_large_n = results.get((architecture, "seven_class", "faithfulness150"))
        binary = results.get((architecture, "binary"))

        faithfulness_source = seven_large_n or seven

        rows.append(
            {
                "architecture": architecture,
                "seven_class_accuracy": seven["validation"]["accuracy"] if seven else None,
                "seven_class_kappa": seven["validation"]["cohens_kappa"] if seven else None,
                "isic2018_test_accuracy": (
                    seven["isic2018_test"]["accuracy"] if seven and "isic2018_test" in seven else None
                ),
                "faithfulness_iou": (
                    faithfulness_source["faithfulness"]["mean_iou"]
                    if faithfulness_source and "faithfulness" in faithfulness_source
                    else None
                ),
                "faithfulness_dice": (
                    faithfulness_source["faithfulness"]["mean_dice"]
                    if faithfulness_source and "faithfulness" in faithfulness_source
                    else None
                ),
                "faithfulness_n": (
                    faithfulness_source["faithfulness"]["n"]
                    if faithfulness_source and "faithfulness" in faithfulness_source
                    else None
                ),
                "binary_accuracy": binary["validation"]["accuracy"] if binary else None,
                "binary_skin_tone_accuracy_spread": _skin_tone_spread(binary) if binary else None,
            }
        )
    return rows


def rank_by(rows: list[dict], key: str, reverse: bool = True) -> list[str]:
    """Architectures ranked by `key` (descending by default), skipping any architecture
    with a missing value for that metric rather than crashing or dropping the whole
    ranking."""
    ranked = sorted(
        (row for row in rows if row[key] is not None), key=lambda r: r[key], reverse=reverse
    )
    return [row["architecture"] for row in ranked]


def summarise(rows: list[dict]) -> str:
    """Plain-text summary: rankings on each dimension, plus an explicit flag for
    whether the same architecture leads on both accuracy and faithfulness or whether
    there's a genuine trade-off -- without picking a winner itself."""
    lines = []

    accuracy_rank = rank_by(rows, "seven_class_accuracy")
    faithfulness_rank = rank_by(rows, "faithfulness_iou")

    lines.append(f"Ranked by seven-class accuracy (highest first): {accuracy_rank}")
    lines.append(
        f"Ranked by Grad-CAM faithfulness / mean IoU (highest first): {faithfulness_rank}"
    )

    if not accuracy_rank or not faithfulness_rank:
        lines.append(
            "Not enough results available yet to compare accuracy against "
            "faithfulness -- run src.evaluate_run for the missing (architecture, task) "
            "pairs first."
        )
    elif accuracy_rank[0] != faithfulness_rank[0]:
        lines.append(
            f"Trade-off: '{accuracy_rank[0]}' leads on accuracy but "
            f"'{faithfulness_rank[0]}' leads on Grad-CAM faithfulness -- no single "
            f"architecture dominates both dimensions. The choice between them depends "
            f"on how much weight accuracy vs. interpretability is given, which is a "
            f"judgement call for the student and supervisor, not this script."
        )
    else:
        lines.append(
            f"'{accuracy_rank[0]}' leads on both accuracy and Grad-CAM faithfulness "
            f"among the six (architecture, task) runs now available."
        )

    lines.append(
        "Note: the faithfulness ranking above is Grad-CAM only, at the larger n=150 "
        "sample where available in preference to the original n=30 sample (see "
        "Chapter 4, Table 4.8, on why n=30 was not trusted on its own). It does not "
        "include SHAP -- the full Grad-CAM-vs-SHAP comparison, which finds no single "
        "architecture wins on faithfulness once SHAP is considered, is reported "
        "separately in Chapter 4 Section 4.5 and Chapter 5 Section 5.2.3."
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results = load_results(results_dir)
    if not results:
        raise SystemExit(f"No src.evaluate_run results found under {results_dir}/ -- run it first.")

    rows = build_comparison_table(results)
    summary = summarise(rows)

    output_path = Path(args.output or f"{results_dir}/trade_off_summary.json")
    with open(output_path, "w") as f:
        json.dump({"comparison_table": rows, "summary": summary}, f, indent=2)

    print(summary)
    print(f"\nFull comparison table written to {output_path}")


if __name__ == "__main__":
    main()
