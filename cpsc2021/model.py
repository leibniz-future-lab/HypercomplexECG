"""

Possible Solutions
------------------
1. segmentation (AF, non-AF) -> postprocess (merge too-close intervals, etc) -> onsets & offsets
2. sequence labelling (AF, non-AF) -> postprocess (merge too-close intervals, etc) -> onsets & offsets
3. per-beat (R peak detection first) classification (CNN, etc. + RR LSTM) -> postprocess (merge too-close intervals, etc) -> onsets & offsets
4. object detection (? onsets and offsets)

"""

from itertools import repeat
from numbers import Real
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).absolute().parents[2]))

from torch_ecg.cfg import CFG
from torch_ecg.components.outputs import RPeaksDetectionOutput, SequenceTaggingOutput

# models from torch_ecg
from torch_ecg.models.ecg_seq_lab_net import ECG_SEQ_LAB_NET  # noqa: F401
from torch_ecg.utils.misc import add_docstring
from torch_ecg.utils.utils_data import mask_to_intervals
from torch_ecg.utils.utils_interval import intervals_union

__all__ = [
    "ECG_SEQ_LAB_NET_CPSC2021",
]


class ECG_SEQ_LAB_NET_CPSC2021(ECG_SEQ_LAB_NET):
    """ """

    __DEBUG__ = True
    __name__ = "ECG_SEQ_LAB_NET_CPSC2021"

    def __init__(self, config: CFG, **kwargs: Any) -> None:
        """

        Parameters
        ----------
        config: dict,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file

        Usage
        -----
        >>> from cfg import ModelCfg
        >>> task = "main"
        >>> model_cfg = deepcopy(ModelCfg[task])
        >>> model_cfg.model_name = "seq_lab"
        >>> model = ECG_SEQ_LAB_NET_CPSC2021(model_cfg)

        """
        if config[config.model_name].reduction == 1:
            config[config.model_name].recover_length = True
        super().__init__(config.classes, config.n_leads, config[config.model_name])
        self.task = config.task

    def extract_features(self, input: Tensor) -> Tensor:
        """
        extract feature map before the dense (linear) classifying layer(s)

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, channels, seq_len)

        Returns
        -------
        features: Tensor,
            of shape (batch_size, seq_len, channels)

        """
        # cnn
        cnn_output = self.cnn(input)  # (batch_size, channels, seq_len)

        # rnn or none
        if self.rnn:
            rnn_output = cnn_output.permute(2, 0, 1)  # (seq_len, batch_size, channels)
            rnn_output = self.rnn(rnn_output)  # (seq_len, batch_size, channels)
            rnn_output = rnn_output.permute(1, 2, 0)  # (batch_size, channels, seq_len)
        else:
            rnn_output = cnn_output

        # attention
        if self.attn:
            rnn_output = self.attn(rnn_output)  # (batch_size, channels, seq_len)
        features = rnn_output.permute(0, 2, 1)  # (batch_size, seq_len, channels)
        return features

    @torch.no_grad()
    def inference(
        self,
        input: Union[Sequence[float], np.ndarray, Tensor],
        bin_pred_thr: float = 0.5,
        **kwargs: Any,
    ) -> Union[SequenceTaggingOutput, RPeaksDetectionOutput]:
        """

        Parameters
        ----------
        input: array_like,
            input tensor, of shape (..., channels, seq_len)
        bin_pred_thr: float, default 0.5,
            the threshold for making binary predictions from scalar predictions
        kwargs: task specific key word arguments

        Returns
        -------
        output: SequenceTaggingOutput or RPeaksDetectionOutput,
            the output of the model
            for main task, the output is a SequenceTaggingOutput instance, with items:
                - classes: list,
                    the list of classes
                - prob: array_like,
                    the probability array of the input sequence of signals
                - pred: array_like,
                    the binary prediction array of the input sequence of signals
                - af_episodes: list of list of intervals,
                    af episodes, in the form of intervals of [start, end], right inclusive
                - af_mask: alias of pred

        """
        return self._inference_main_task(input, bin_pred_thr, **kwargs)

    @add_docstring(inference.__doc__)
    def inference_CPSC2021(
        self,
        input: Union[Sequence[float], np.ndarray, Tensor],
        bin_pred_thr: float = 0.5,
        **kwargs: Any,
    ) -> Union[SequenceTaggingOutput, RPeaksDetectionOutput]:
        """
        alias for `self.inference`
        """
        return self.inference(input, bin_pred_thr, **kwargs)

    @torch.no_grad()
    def _inference_main_task(
        self,
        input: Union[Sequence[float], np.ndarray, Tensor],
        bin_pred_thr: float = 0.5,
        rpeaks: Optional[Union[Sequence[int], Sequence[Sequence[int]]]] = None,
        episode_len_thr: int = 5,
    ) -> SequenceTaggingOutput:
        """

        Parameters
        ----------
        input: array_like,
            input tensor, of shape (..., channels, seq_len)
        bin_pred_thr: float, default 0.5,
            the threshold for making binary predictions from scalar predictions
        rpeaks: sequence of sequence of int, optional,
            sequences of r peak indices
        episode_len_thr: int, default 5,
            minimal length of (both af and normal) episodes,
            with units in number of beats (rpeaks)

        Returns
        -------
        output: SequenceTaggingOutput, with items:
            - classes: list,
                the list of classes
            - prob: array_like,
                the probability array of the input sequence of signals
            - pred: array_like,
                the binary prediction array of the input sequence of signals
            - af_episodes: list of list of intervals,
                af episodes, in the form of intervals of [start, end], right inclusive
            - af_mask: alias of pred

        """
        self.eval()
        _input = torch.as_tensor(input, dtype=self.dtype, device=self.device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        batch_size, n_leads, seq_len = _input.shape
        prob = self.sigmoid(self.forward(_input))
        prob = prob.cpu().detach().numpy().squeeze(-1)

        af_episodes, af_mask = _main_task_post_process(
            prob=prob,
            fs=self.config.fs,
            reduction=self.config.reduction,
            bin_pred_thr=bin_pred_thr,
            rpeaks=rpeaks,
            siglens=list(repeat(seq_len, batch_size)),
            episode_len_thr=episode_len_thr,
        )
        return SequenceTaggingOutput(
            #classes=self.class_names,
            classes=self.classes,
            prob=prob,
            pred=af_mask,
            af_episodes=af_episodes,
            af_mask=af_mask,  # alias of pred
        )


def _main_task_post_process(
    prob: np.ndarray,
    fs: Real,
    reduction: int,
    bin_pred_thr: float = 0.5,
    rpeaks: Sequence[Sequence[int]] = None,
    siglens: Optional[Sequence[int]] = None,
    episode_len_thr: int = 5,
) -> Tuple[List[List[List[int]]], np.ndarray]:
    """

    post processing of the main task,
    converting mask into list of af episodes,
    and doing filtration, eliminating (both af and normal) episodes that are too short

    Parameters
    ----------
    prob: ndarray,
        predicted af mask, of shape (batch_size, seq_len)
    fs: real number,
        sampling frequency of the signal
    reduction: int,
        reduction ratio of the predicted af mask w.r.t. the signal
    bin_pred_thr: float, default 0.5,
        the threshold for making binary predictions from scalar predictions
    rpeaks: sequence of sequence of int, optional,
        sequences of r peak indices
    siglens: sequence of int, optional,
        original signal lengths,
        used to do padding for af intervals
    episode_len_thr: int, default 5,
        minimal length of (both af and normal) episodes,
        with units in number of beats (rpeaks)

    Returns
    -------
    af_episodes: list of list of intervals,
        af episodes, in the form of intervals of [start, end], right inclusive
    af_mask: ndarray,
        array (mask) of binary prediction of af, of shape (batch_size, seq_len)

    """
    batch_size, prob_arr_len = prob.shape
    model_spacing = 1000 / fs  # units in ms
    input_len = reduction * prob_arr_len
    default_rr = int(fs * 0.8)

    af_mask = (prob >= bin_pred_thr).astype(int)

    af_episodes = []
    for b_idx in range(batch_size):
        b_mask = af_mask[b_idx]
        intervals = mask_to_intervals(b_mask, [0, 1])
        b_af_episodes = [
            [itv[0] * reduction, itv[1] * reduction] for itv in intervals[1]
        ]
        b_n_episodes = [
            [itv[0] * reduction, itv[1] * reduction] for itv in intervals[0]
        ]
        if siglens is not None and siglens[b_idx] % reduction > 0:
            b_n_episodes.append(
                [siglens[b_idx] // reduction * reduction, siglens[b_idx]]
            )
        if rpeaks is not None:
            b_rpeaks = rpeaks[b_idx]
            # merge non-af episodes shorter than `episode_len_thr`
            b_af_episodes.extend(
                [
                    itv
                    for itv in b_n_episodes
                    if len([r for r in b_rpeaks if itv[0] <= r < itv[1]])
                    < episode_len_thr
                ]
            )
            b_af_episodes = intervals_union(b_af_episodes)
            # eliminate af episodes shorter than `episode_len_thr`
            # and make right inclusive
            b_af_episodes = [
                [itv[0], itv[1] - 1]
                for itv in b_af_episodes
                if len([r for r in b_rpeaks if itv[0] <= r < itv[1]]) >= episode_len_thr
            ]
        else:
            # merge non-af episodes shorter than `episode_len_thr`
            b_af_episodes.extend(
                [
                    itv
                    for itv in b_n_episodes
                    if itv[1] - itv[0] < default_rr * episode_len_thr
                ]
            )
            b_af_episodes = intervals_union(b_af_episodes)
            # eliminate af episodes shorter than `episode_len_thr`
            # and make right inclusive
            b_af_episodes = [
                [itv[0], itv[1] - 1]
                for itv in b_af_episodes
                if itv[1] - itv[0] >= default_rr * episode_len_thr
            ]
        af_episodes.append(b_af_episodes)
    return af_episodes, af_mask
