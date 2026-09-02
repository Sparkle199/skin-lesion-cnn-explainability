"""Tier 1 experiments: decision-threshold calibration for the binary DDI models,
bootstrap confidence intervals testing whether (a) the seven-class architecture ranking
and (b) the joint-vs-zero-shot-vs-finetuned DDI ranking are statistically robust or
within noise, and (2026-08-28 continuation) a paired McNemar's test between the three
DDI approaches plus threshold calibration extended to zero-shot and fine-tuned (not
just joint) -- to check whether calibrating the alternatives closes the gap to joint
rather than only confirming joint wins at the default 0.5 threshold.

Pure numpy/sklearn/scipy -- reads src.export_predictions's raw per-example output only,
no TensorFlow dependency, but run via the project venv on the pod for consistency with
the rest of this project's execution history.

Usage: python scripts/tier1_analysis.py
"""

import json

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

RNG = np.random.default_rng(42)
N_BOOTSTRAP = 2000
ARCHITECTURES = ["resnet50", "efficientnetb4", "vgg16"]
DDI_APPROACHES = ["joint", "zero_shot", "finetuned"]


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


def threshold_sweep(y_true, y_proba):
    """0.05-0.95 threshold sweep (kappa-maximising), returning the full sweep and the
    default-0.5 and best-threshold results -- factored out so it applies identically to
    all three DDI approaches, not just the joint model as in the original Tier 1 pass."""
    y_true, y_proba = np.array(y_true), np.array(y_proba)
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
    return {
        "default_threshold_kappa": float(cohen_kappa_score(y_true, default_pred)),
        "default_threshold_accuracy": float(accuracy_score(y_true, default_pred)),
        "best_threshold": best,
        "sweep": sweep,
    }


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """Exact (binomial) McNemar's test comparing two classifiers' correctness on the
    same paired examples -- appropriate here since the DDI held-out set is small (n=99)
    and all three approaches are scored on the identical rows, making this a paired
    comparison a plain independent-samples test (like bootstrap CI) is not designed for.
    Returns discordant-pair counts and a two-sided p-value for whether A's error rate
    differs significantly from B's."""
    y_true, y_pred_a, y_pred_b = np.array(y_true), np.array(y_pred_a), np.array(y_pred_b)
    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true
    a_right_b_wrong = int(np.sum(correct_a & ~correct_b))
    a_wrong_b_right = int(np.sum(~correct_a & correct_b))
    n_discordant = a_right_b_wrong + a_wrong_b_right
    p_value = 1.0 if n_discordant == 0 else binomtest(
        min(a_right_b_wrong, a_wrong_b_right), n_discordant, p=0.5, alternative="two-sided"
    ).pvalue
    return {
        "a_right_b_wrong": a_right_b_wrong,
        "a_wrong_b_right": a_wrong_b_right,
        "n_discordant": n_discordant,
        "p_value": float(p_value),
    }


def main():
    with open("results/raw_predictions.json") as f:
        data = json.load(f)

    results = {
        "seven_class_bootstrap": {},
        "ddi_threshold_calibration": {},
        "ddi_bootstrap": {},
        "ddi_mcnemar": {},
    }

    # --- Seven-class accuracy/kappa bootstrap per architecture ---
    for arch in ARCHITECTURES:
        d = data[f"{arch}_seven_class"]
        y_true = np.array(d["y_true"])
        y_pred = np.array(d["y_proba"]).argmax(axis=1)
        results["seven_class_bootstrap"][arch] = {
            "accuracy": bootstrap_ci(y_true, y_pred, accuracy_score),
            "kappa": bootstrap_ci(y_true, y_pred, cohen_kappa_score),
        }

    # --- DDI: threshold calibration for ALL THREE approaches (was joint-only before) ---
    for arch in ARCHITECTURES:
        results["ddi_threshold_calibration"][arch] = {}
        for approach in DDI_APPROACHES:
            d = data[f"{arch}_ddi_{approach}"]
            results["ddi_threshold_calibration"][arch][approach] = threshold_sweep(d["y_true"], d["y_proba"])

    # --- DDI: bootstrap CI for joint vs. zero-shot vs. fine-tuned kappa, per architecture ---
    for arch in ARCHITECTURES:
        results["ddi_bootstrap"][arch] = {}
        for approach in DDI_APPROACHES:
            d = data[f"{arch}_ddi_{approach}"]
            y_true = np.array(d["y_true"])
            y_pred = (np.array(d["y_proba"]) >= 0.5).astype(int)
            results["ddi_bootstrap"][arch][approach] = bootstrap_ci(y_true, y_pred, cohen_kappa_score)

    # --- DDI: paired McNemar's test between every pair of approaches, per architecture ---
    for arch in ARCHITECTURES:
        results["ddi_mcnemar"][arch] = {}
        preds = {}
        y_true_ref = None
        for approach in DDI_APPROACHES:
            d = data[f"{arch}_ddi_{approach}"]
            y_true_ref = d["y_true"]
            preds[approach] = (np.array(d["y_proba"]) >= 0.5).astype(int)
        for a, b in [("joint", "zero_shot"), ("joint", "finetuned"), ("zero_shot", "finetuned")]:
            results["ddi_mcnemar"][arch][f"{a}_vs_{b}"] = mcnemar_test(y_true_ref, preds[a], preds[b])

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

    print("\n=== DDI threshold calibration, all three approaches ===")
    for arch in ARCHITECTURES:
        for approach in DDI_APPROACHES:
            c = results["ddi_threshold_calibration"][arch][approach]
            b = c["best_threshold"]
            print(
                f"{arch:15s} {approach:11s} default(0.5) kappa={c['default_threshold_kappa']:.3f}  "
                f"best t={b['threshold']:.2f} kappa={b['kappa']:.3f} acc={b['accuracy']:.3f}"
            )

    print("\n=== DDI approach bootstrap CI (kappa), 95% CI ===")
    for arch in ARCHITECTURES:
        row = results["ddi_bootstrap"][arch]
        parts = [f"{ap}={row[ap]['point']:.3f}[{row[ap]['ci_low']:.3f},{row[ap]['ci_high']:.3f}]" for ap in DDI_APPROACHES]
        print(f"{arch:15s} " + "  ".join(parts))

    print("\n=== DDI paired McNemar's test (exact binomial), p-values ===")
    for arch in ARCHITECTURES:
        row = results["ddi_mcnemar"][arch]
        for pair, r in row.items():
            sig = "*" if r["p_value"] < 0.05 else " "
            print(
                f"{arch:15s} {pair:22s} discordant={r['n_discordant']:3d} "
                f"({r['a_right_b_wrong']} vs {r['a_wrong_b_right']})  p={r['p_value']:.4f} {sig}"
            )

    print("\nWrote results/tier1_calibration_bootstrap.json")


if __name__ == "__main__":
    main()
