"""
auxiliary metrics for the task of qrs detection

References
----------
[1] http://2019.icbeb.org/Challenge.html
"""

import multiprocessing as mp
from numbers import Real
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).absolute().parents[2]))

from torch_ecg.models.loss import MaskedBCEWithLogitsLoss
from torch_ecg.utils.utils_data import mask_to_intervals

__all__ = [
    "compute_main_task_metric",
]


_MBCE = MaskedBCEWithLogitsLoss()


def compute_main_task_metric(
    mask_truths: Sequence[Union[np.ndarray, Sequence[int]]],
    mask_preds: Sequence[Union[np.ndarray, Sequence[int]]],
    fs: Real,
    reduction: int,
    weight_masks: Optional[Sequence[Union[np.ndarray, Sequence[int]]]] = None,
    rpeaks: Optional[Sequence[Sequence[int]]] = None,
    verbose: int = 0,
) -> Dict[str, float]:
    """

    this metric for evaluating the main task model (seq_lab or unet),
    which imitates the metric provided by the organizers of CPSC2021

    Parameters
    ----------
    mask_truths: array_like,
        sequences of AF labels on rr intervals, of shape (n_samples, seq_len)
    mask_preds: array_like,
        sequences of AF predictions on rr intervals, of shape (n_samples, seq_len)
    fs: Real,
        sampling frequency of the model input ECGs,
        used when (indices of) `rpeaks` not privided
    reduction: int,
        reduction ratio of the main task model
    rpeaks: array_like, optional,
        indices of rpeaks in the model input ECGs,
        if set, more precise scores can be computed

    Returns
    -------
    main_score: float,
        the score computed from predicts from the main task model,
        similar to CPSC2021 challenge metric
    neg_masked_bce: float,
        negative masked BCE loss
    """
    default_rr = int(fs * 0.8 / reduction)
    if rpeaks is not None:
        assert len(rpeaks) == len(mask_truths)
    with mp.Pool(processes=max(1, mp.cpu_count())) as pool:
        af_episode_truths = pool.starmap(
            func=mask_to_intervals, iterable=[(row, 1, True) for row in mask_truths]
        )
    with mp.Pool(processes=max(1, mp.cpu_count())) as pool:
        af_episode_preds = pool.starmap(
            func=mask_to_intervals, iterable=[(row, 1, True) for row in mask_preds]
        )
    af_episode_truths = [
        [[itv[0] * reduction, itv[1] * reduction] for itv in sample]
        for sample in af_episode_truths
    ]
    af_episode_preds = [
        [[itv[0] * reduction, itv[1] * reduction] for itv in sample]
        for sample in af_episode_preds
    ]
    n_samples, seq_len = np.array(mask_truths).shape
    print(f'Number of samples: {n_samples}')
    scoring_mask = np.zeros((n_samples, seq_len * reduction))
    for idx, sample in enumerate(af_episode_truths):
        for itv in sample:
            if rpeaks is not None:
                itv_rpeaks = [
                    i for i, r in enumerate(rpeaks[idx]) if itv[0] <= r < itv[1]
                ]
                start = rpeaks[idx][max(0, itv_rpeaks[0] - 2)]
                end = rpeaks[idx][min(len(rpeaks[idx]) - 1, itv_rpeaks[0] + 2)] + 1
                scoring_mask[idx][start:end] = 0.5
                start = rpeaks[idx][max(0, itv_rpeaks[-1] - 2)]
                end = rpeaks[idx][min(len(rpeaks[idx]) - 1, itv_rpeaks[-1] + 2)] + 1
                scoring_mask[idx][start:end] = 0.5
                start = rpeaks[idx][max(0, itv_rpeaks[0] - 1)]
                end = rpeaks[idx][min(len(rpeaks[idx]) - 1, itv_rpeaks[0] + 1)] + 1
                scoring_mask[idx][start:end] = 1
                start = rpeaks[idx][max(0, itv_rpeaks[-1] - 1)]
                end = rpeaks[idx][min(len(rpeaks[idx]) - 1, itv_rpeaks[-1] + 1)] + 1
                scoring_mask[idx][start:end] = 1
            else:
                scoring_mask[idx][
                    max(0, itv[0] - 2 * default_rr) : min(
                        seq_len, itv[0] + 2 * default_rr + 1
                    )
                ] = 0.5
                scoring_mask[idx][
                    max(0, itv[1] - 2 * default_rr) : min(
                        seq_len, itv[1] + 2 * default_rr + 1
                    )
                ] = 0.5
                scoring_mask[idx][
                    max(0, itv[0] - 1 * default_rr) : min(
                        seq_len, itv[0] + 1 * default_rr + 1
                    )
                ] = 1
                scoring_mask[idx][
                    max(0, itv[1] - 1 * default_rr) : min(
                        seq_len, itv[1] + 1 * default_rr + 1
                    )
                ] = 1
    main_score = sum(
        [
            scoring_mask[idx][itv].sum() / max(1, len(af_episode_truths[idx]))
            for idx in range(n_samples)
            for itv in af_episode_preds[idx]
        ]
    )
    main_score += sum(
        [0 == len(t) == len(p) for t, p in zip(af_episode_truths, af_episode_preds)]
    )
    neg_masked_bce = -_MBCE(
        torch.as_tensor(mask_preds, dtype=torch.float32, device=torch.device("cpu")),
        torch.as_tensor(mask_truths, dtype=torch.float32, device=torch.device("cpu")),
        torch.as_tensor(weight_masks, dtype=torch.float32, device=torch.device("cpu")),
    ).item()
    acc, uar, class_acc = compute_uar(af_episode_truths, af_episode_preds, n_samples)
    metrics = {
        "main_score": main_score,
        "main_score_avg": main_score / n_samples,
        "neg_masked_bce": neg_masked_bce,
        "accuracy": acc,
        "uar": uar,
        "class accuracy N": class_acc[0],
        "class accuracy peAF": class_acc[1],
        "class accuracy pAF": class_acc[2],
    }
    return metrics


def compute_uar(af_episode_truths, af_episode_preds, n_samples):
    """
    Calculates accuracy, class recalls, unweighted average recall (uar)
    """
    acc = sum([len(t) == len(p) for t, p in zip(af_episode_truths, af_episode_preds)]) / n_samples
    print(f"Accuracy: {acc}")
    n_classes = 3
    class_acc = np.zeros(n_classes)
    class_nsamples = np.zeros(n_classes)
    for t, p in zip(af_episode_truths, af_episode_preds):
        if len(t) == 0:  # "N"
            class_nsamples[0] += 1
            if len(p) == 0:
                class_acc[0] += 1
        elif len(t) == 1:  # "persistent AF"
            class_nsamples[1] += 1
            if len(p) == 1:
                class_acc[1] += 1
        else:  # len(t) >= 2:  # "pAF"
            class_nsamples[2] += 1
            if len(p) >= 2:
                class_acc[2] += 1
    class_acc = class_acc / class_nsamples
    uar = np.nanmean(class_acc)
    return acc, uar, class_acc
