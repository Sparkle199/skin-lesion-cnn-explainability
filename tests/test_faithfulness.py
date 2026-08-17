from src.xai.faithfulness import compute_overlap, mean_overlap


def test_compute_overlap_perfect_match_scores_one(synthetic_lesion_mask):
    attention = synthetic_lesion_mask.astype(float)

    result = compute_overlap(attention, synthetic_lesion_mask, threshold=0.5)

    assert result["iou"] == 1.0
    assert result["dice"] == 1.0


def test_compute_overlap_disjoint_scores_zero(synthetic_lesion_mask):
    attention = (~synthetic_lesion_mask).astype(float)

    result = compute_overlap(attention, synthetic_lesion_mask, threshold=0.5)

    assert result["iou"] == 0.0
    assert result["dice"] == 0.0


def test_compute_overlap_partial_matches_hand_calculation():
    import numpy as np

    truth = np.array([[1, 1, 0, 0]], dtype=bool)
    attention = np.array([[0.9, 0.1, 0.9, 0.1]])  # thresholds to [True, False, True, False]

    result = compute_overlap(attention, truth, threshold=0.5)

    # predicted={0,2}, truth={0,1}; intersection=1, union=3 -> IoU=1/3; dice=2*1/(2+2)=0.5
    assert abs(result["iou"] - 1 / 3) < 1e-9
    assert abs(result["dice"] - 0.5) < 1e-9


def test_mean_overlap_averages_across_samples(synthetic_lesion_mask):
    perfect = synthetic_lesion_mask.astype(float)
    disjoint = (~synthetic_lesion_mask).astype(float)

    result = mean_overlap([perfect, disjoint], [synthetic_lesion_mask, synthetic_lesion_mask])

    assert result["n"] == 2
    assert abs(result["mean_iou"] - 0.5) < 1e-9
    assert abs(result["mean_dice"] - 0.5) < 1e-9


def test_mean_overlap_rejects_mismatched_lengths(synthetic_lesion_mask):
    import pytest

    with pytest.raises(ValueError):
        mean_overlap([synthetic_lesion_mask.astype(float)], [synthetic_lesion_mask, synthetic_lesion_mask])
