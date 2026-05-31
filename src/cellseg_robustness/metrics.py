"""Instance segmentation metrics for proof-of-work experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InstanceMetrics:
    true_instances: int
    pred_instances: int
    matched_instances: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    object_f1: float
    mean_matched_iou: float
    mean_matched_dice: float
    count_error: int
    absolute_count_error: int


def instance_ids(mask: np.ndarray) -> np.ndarray:
    """Return sorted positive instance ids."""
    ids = np.unique(mask)
    return ids[ids > 0]


def relabel_sequential(mask: np.ndarray) -> np.ndarray:
    """Relabel positive instance ids to 1..N."""
    output = np.zeros(mask.shape, dtype=np.int32)
    for new_id, old_id in enumerate(instance_ids(mask), start=1):
        output[mask == old_id] = new_id
    return output


def match_instances(
    truth: np.ndarray,
    prediction: np.ndarray,
    iou_threshold: float = 0.5,
) -> list[tuple[int, int, float, float]]:
    """Greedily match predicted instances to truth instances by IoU."""
    truth_ids = instance_ids(truth)
    pred_ids = instance_ids(prediction)
    candidates: list[tuple[float, int, int, float]] = []

    for truth_id in truth_ids:
        truth_mask = truth == truth_id
        truth_area = int(truth_mask.sum())
        if truth_area == 0:
            continue
        overlapping_pred_ids = np.unique(prediction[truth_mask])
        for pred_id in overlapping_pred_ids:
            if pred_id <= 0:
                continue
            pred_mask = prediction == pred_id
            intersection = int(np.logical_and(truth_mask, pred_mask).sum())
            union = truth_area + int(pred_mask.sum()) - intersection
            if union == 0:
                continue
            iou = intersection / union
            if iou < iou_threshold:
                continue
            dice = (2 * intersection) / (truth_area + int(pred_mask.sum()))
            candidates.append((iou, int(truth_id), int(pred_id), dice))

    candidates.sort(reverse=True, key=lambda item: item[0])
    matched_truth: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[int, int, float, float]] = []

    for iou, truth_id, pred_id, dice in candidates:
        if truth_id in matched_truth or pred_id in matched_pred:
            continue
        matched_truth.add(truth_id)
        matched_pred.add(pred_id)
        matches.append((truth_id, pred_id, iou, dice))

    return matches


def compute_instance_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    iou_threshold: float = 0.5,
) -> InstanceMetrics:
    """Compute object-level detection and overlap metrics."""
    truth_count = int(len(instance_ids(truth)))
    pred_count = int(len(instance_ids(prediction)))
    matches = match_instances(truth, prediction, iou_threshold=iou_threshold)
    matched_count = len(matches)
    false_positives = pred_count - matched_count
    false_negatives = truth_count - matched_count
    precision = matched_count / pred_count if pred_count else 0.0
    recall = matched_count / truth_count if truth_count else 0.0
    object_f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    mean_iou = float(np.mean([match[2] for match in matches])) if matches else 0.0
    mean_dice = float(np.mean([match[3] for match in matches])) if matches else 0.0
    count_error = pred_count - truth_count

    return InstanceMetrics(
        true_instances=truth_count,
        pred_instances=pred_count,
        matched_instances=matched_count,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        object_f1=object_f1,
        mean_matched_iou=mean_iou,
        mean_matched_dice=mean_dice,
        count_error=count_error,
        absolute_count_error=abs(count_error),
    )
