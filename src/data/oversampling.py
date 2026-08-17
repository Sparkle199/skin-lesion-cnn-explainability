"""Pure arithmetic for source-aware oversampling of the binary task's DDI-vs-HAM10000
imbalance (DDI is ~6% of the combined corpus by volume). Kept separate from
src/data/pipeline.py's tf.data logic so it can be tested without TensorFlow, per the
lesson recorded in the agent journal after src/trade_off.py.
"""

import math

# Cap on how much of each training batch may come from DDI. A full 50/50 balance risks
# the model overfitting on DDI's ~656 underlying images even with augmentation
# smoothing out repeats (see agent journal, 2026-08-10 entry, and spec's RISKS section).
MAX_DDI_FRACTION = 0.4


def validate_ddi_fraction(ddi_fraction: float) -> float:
    """Reject an oversampling target outside (0, MAX_DDI_FRACTION] rather than silently
    clamping it -- a caller requesting e.g. 60% DDI per batch should be told why that's
    discouraged, not have their request silently rewritten to something else."""
    if not 0 < ddi_fraction <= MAX_DDI_FRACTION:
        raise ValueError(
            f"ddi_fraction must be in (0, {MAX_DDI_FRACTION}], got {ddi_fraction}. "
            f"A higher fraction risks overfitting on DDI's ~656 underlying images even "
            f"with augmentation."
        )
    return ddi_fraction


def compute_steps_per_epoch(n_ham10000: int, batch_size: int, ddi_fraction: float) -> int:
    """Steps per epoch such that, in expectation, every HAM10000 image is seen roughly
    once per epoch.

    HAM10000 is treated as the epoch-defining dataset since it's the larger, primary
    source; DDI is oversampled (via repeat + weighted sampling in
    src.data.pipeline.make_oversampled_binary_dataset) to fill its target share of each
    batch regardless of how many steps that takes to cycle through HAM10000 once.
    """
    validate_ddi_fraction(ddi_fraction)
    ham_per_batch = batch_size * (1 - ddi_fraction)
    return max(1, math.ceil(n_ham10000 / ham_per_batch))
