"""Tier 1 experiments: decision-threshold calibration for the binary DDI models, and
bootstrap confidence intervals testing whether (a) the seven-class architecture ranking
and (b) the joint-vs-zero-shot-vs-finetuned DDI ranking are statistically robust or
within noise, given all prior comparisons were single point estimates from single runs.

Pure numpy/sklearn -- reads src.export_predictions's raw per-example output only, no
TensorFlow dependency, but run via the project venv on the pod for consistency with the
rest of this project's execution history.

Usage: python scripts/tier1_analysis.py
"""

import json

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

RNG = np.random.default_rng(42)
N_BOOTSTRAP = 2000
ARCHITECTURES = ["resnet50", "efficientnetb4", "vgg16"]


def bootstrap_ci(y_true, y_pred, metric_fn, n=N_BOOTSTRAP, alpha=0.05):
    """Percentile bootstrap: resample (y_true, y_pred) pairs together with replacement,
    n times, recompute metric_fn each time, and report the 95% CI as the [2.5, 97.5]
    percentiles of the resampled distribution."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n_samples = len(y_true)
    scores = np.empty(n)
    for i in range(n):
        idx = RNG.integers(0, n_samples, n_samples)
        scores[i] = metric_fn(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": float(metric_fn(y_true, y_pred)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "bootstrap_mean": float(scores.mean()),
        "n": int(n_samples),
    }


def main():
    with open("results/raw_predictions.json") as f:
        data = json.load(f)

    results = {"seven_class_bootstrap": {}, "ddi_threshold_calibration": {}, "ddi_bootstrap": {}}

    # --- Seven-class accuracy/kappa bootstrap per architecture ---
    for arch in ARCHITECTURES:
        d = data[f"{arch}_seven_class"]
        y_true = np.array(d["y_true"])
        y_pred = np.array(d["y_proba"]).argmax(axis=1)
        results["seven_class_bootstrap"][arch] = {
            "accuracy": bootstrap_ci(y_true, y_pred, accuracy_score),
            "kappa": bootstrap_ci(y_true, y_pred, cohen_kappa_score),
        }

    # --- DDI: threshold calibration for the joint (recommended) models ---
    for arch in ARCHITECTURES:
        d = data[f"{arch}_ddi_joint"]
        y_true = np.array(d["y_true"])
        y_proba = np.array(d["y_proba"])  # P(malignant)

        sweep = []
        for t in np.linspace(0.05, 0.95, 19):
            y_pred = (y_proba >= t).astype(int)
            sweep.append(
                {
                    "threshold": float(t),
                    "kappa": float(cohen_kappa_score(y_true, y_pred)),
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "malignant_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
                }
            )
        best = max(sweep, key=lambda s: s["kappa"])
        default_pred = (y_proba >= 0.5).astype(int)

        results["ddi_threshold_calibration"][arch] = {
            "default_threshold_kappa": float(cohen_kappa_score(y_true, default_pred)),
            "best_threshold": best,
            "sweep": sweep,
        }

    # --- DDI: bootstrap CI for joint vs. zero-shot vs. fine-tuned kappa, per architecture ---
    for arch in ARCHITECTURES:
        results["ddi_bootstrap"][arch] = {}
        for approach in ["joint", "zero_shot", "finetuned"]:
            d = data[f"{arch}_ddi_{approach}"]
            y_true = np.array(d["y_true"])
            y_pred = (np.array(d["y_proba"]) >= 0.5).astype(int)
            results["ddi_bootstrap"][arch][approach] = bootstrap_ci(y_true, y_pred, cohen_kappa_score)

    with open("results/tier1_calibration_bootstrap.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=== Seven-class accuracy/kappa, 95% bootstrap CI ===")
    for arch in ARCHITECTURES:
        a = results["seven_class_bootstrap"][arch]["accuracy"]
        k = results["seven_class_bootstrap"][arch]["kappa"]
        print(
            f"{arch:15s} acc={a['point']:.3f} [{a['ci_low']:.3f},{a['ci_high']:.3f}]  "
            f"kappa={k['point']:.3f} [{k['ci_low']:.3f},{k['ci_high']:.3f}]  n={a['n']}"
        )

    print("\n=== DDI joint-model threshold calibration ===")
    for arch in ARCHITECTURES:
        c = results["ddi_threshold_calibration"][arch]
        b = c["best_threshold"]
        print(
            f"{arch:15s} default(0.5) kappa={c['default_threshold_kappa']:.3f}  "
            f"best threshold={b['threshold']:.2f} kappa={b['kappa']:.3f} "
            f"acc={b['accuracy']:.3f} malignant_f1={b['malignant_f1']:.3f}"
        )

    print("\n=== DDI approach bootstrap CI (kappa), 95% CI ===")
    for arch in ARCHITECTURES:
        row = results["ddi_bootstrap"][arch]
        parts = [f"{ap}={row[ap]['point']:.3f}[{row[ap]['ci_low']:.3f},{row[ap]['ci_high']:.3f}]" for ap in ["joint", "zero_shot", "finetuned"]]
        print(f"{arch:15s} " + "  ".join(parts))

    print("\nWrote results/tier1_calibration_bootstrap.json")


if __name__ == "__main__":
    main()
