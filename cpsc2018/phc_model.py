"""
"""

from copy import deepcopy
from typing import Any, Optional, Sequence, Union

# import time
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
from einops import rearrange

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).absolute().parents[2]))

from torch_ecg.components.outputs import MultiLabelClassificationOutput

from torch_ecg.utils.misc import dict_to_str
from torch_ecg.utils.utils_nn import CkptMixin, SizeMixin

from torch_ecg.cfg import CFG
from torch_ecg.model_configs.ecg_crnn import ECG_CRNN_CONFIG

from .cfg import ModelCfg
from .model import ECG_CNN_CPSC2018
from models.multi_scopic_phc import MultiScopicPHCNN
from models.densenet_phc import DenseNet_PHC
from models.resnet_phc import ResNet_PHC
from models.phc_nets import PHM_MLP, PHM_SEBlock

__all__ = [
    "ECG_CNN_CPSC2018_PHC",
]


class ECG_CNN_CPSC2018_PHC(ECG_CNN_CPSC2018):
    """ """

    __DEBUG__ = False
    __name__ = "ECG_CNN_CPSC2018_PHC"

    def __init__(
        self,
        classes: Sequence[str],
        n_leads: int,
        config: Optional[CFG] = None,
        **kwargs: Any
    ) -> None:
        """
        Parameters
        ----------
        classes: list,
            list of the classes for classification
        n_leads: int,
            number of leads (number of input channels)
        config: dict, optional,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file

        """
        # model_config = deepcopy(ModelCfg)
        # model_config.update(deepcopy(config) or {})
        # assert n_leads == 12, "CinC2020 only supports 12-lead models"
        # super().__init__(classes, n_leads, model_config, **kwargs)
        ###########
        super().__init__(classes=classes, n_leads=n_leads, config=config)
        # self.classes = list(classes)
        # self.n_classes = len(classes)
        # self.n_leads = n_leads
        # self.config = deepcopy(ECG_CRNN_CONFIG)
        # self.config.update(deepcopy(model_config) or {})
        # # print(f"Model configs: {self.config}")
        if self.__DEBUG__:
            print(f"classes (totally {self.n_classes}) for prediction:{self.classes}")
            print(
                f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}"
            )
            debug_input_len = 4000

        cnn_choice = self.config.cnn.name.lower()
        cnn_config = self.config.cnn[self.config.cnn.name]
        # if "resnet" in cnn_choice:
        #     self.cnn = ResNet(self.n_leads, **cnn_config)
        # el
        if "multi_scopic" in cnn_choice:
            self.cnn = MultiScopicPHCNN(n=self.n_leads, in_channels=self.n_leads, **cnn_config)
        elif "resnet" in cnn_choice:
            self.cnn = ResNet_PHC(n=self.n_leads, in_channels=self.n_leads, **cnn_config)
        elif "densenet" in cnn_choice or "dense_net" in cnn_choice:
            self.cnn = DenseNet_PHC(n=self.n_leads, in_channels=self.n_leads, **cnn_config)
        else:
            raise NotImplementedError(
                f"the CNN \042{cnn_choice}\042 not implemented yet"
            )
        attn_input_size = self.cnn.compute_output_shape(None, None)[1]

        if self.__DEBUG__:
            cnn_output_shape = self.cnn.compute_output_shape(debug_input_len, None)
            print(
                f"cnn output shape (batch_size, features, seq_len) = {cnn_output_shape}, "
                f"given input_len = {debug_input_len}"
            )

        # attention
        if self.config.attn.name.lower() == "none":
            self.attn = None
            clf_input_size = attn_input_size
        elif self.config.attn.name.lower() == "se":  # squeeze_exitation
            self.attn = PHM_SEBlock(
                n=self.n_leads,
                in_channels=attn_input_size,
                reduction=self.config.attn.se.reduction,
                activation=self.config.attn.se.activation,
                kw_activation=self.config.attn.se.kw_activation,
                bias=self.config.attn.se.bias,
            )
            clf_input_size = self.attn.compute_output_shape(None, None)[1]

        if self.__DEBUG__:
            print(f"clf_input_size = {clf_input_size}")

        if self.config.rnn.name.lower() == "lstm" and not self.config.rnn.lstm.retseq:
            self.pool = None
            if self.config.global_pool.lower() != "none":
                print(
                    f"since `retseq` of rnn is False, hence global pooling `{self.config.global_pool}` is ignored"
                )
        elif self.config.global_pool.lower() == "max":
            self.pool = nn.AdaptiveMaxPool1d(
                (self.config.global_pool_size,), return_indices=False
            )
            clf_input_size *= self.config.global_pool_size
        elif self.config.global_pool.lower() == "avg":
            self.pool = nn.AdaptiveAvgPool1d((self.config.global_pool_size,))
            clf_input_size *= self.config.global_pool_size
        elif self.config.global_pool.lower() == "attn":
            raise NotImplementedError("Attentive pooling not implemented yet!")
        elif self.config.global_pool.lower() == "none":
            self.pool = None
        else:
            raise NotImplementedError(
                f"pooling method {self.config.global_pool} not implemented yet!"
            )

        # input of `self.clf` has shape: batch_size, channels
        self.clf = PHM_MLP(
            n=self.n_leads,
            in_channels=clf_input_size,
            out_channels=self.config.clf.out_channels + [self.n_classes],
            activation=self.config.clf.activation,
            bias=self.config.clf.bias,
            dropouts=self.config.clf.dropouts,
            skip_last_activation=True,
        )

        # for inference
        # classification: if single-label, use softmax; otherwise (multi-label) use sigmoid
        # sequence tagging: if background counted in `classes`, use softmax; otherwise use sigmoid
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)

    # def forward(self, input: Tensor) -> Tensor:
    #     """
    #
    #     Parameters
    #     ----------
    #     input: Tensor,
    #         of shape (batch_size, channels, seq_len)
    #
    #     Returns
    #     -------
    #     pred: Tensor,
    #         of shape (batch_size, n_classes)
    #
    #     """
    #
    #     # print(f"Input size: {input.size()}")
    #     # features = self.extract_features(input)
    #     # tic = time.perf_counter()
    #     features = self.cnn(input)
    #     # toc = time.perf_counter()
    #     # print(f"Extract features runtime: {toc - tic:0.4f} seconds")
    #     # print(f"Features size: {features.size()}")
    #
    #     if self.pool:
    #         features = self.pool(features)  # (batch_size, channels, pool_size)
    #         # features = features.squeeze(dim=-1)
    #         features = rearrange(
    #             features,
    #             "batch_size channels pool_size -> batch_size (channels pool_size)",
    #         )
    #     else:
    #         # features of shape (batch_size, channels) or (batch_size, seq_len, channels)
    #         pass
    #
    #     # print(f"Pooling size: {features.size()}")
    #     # print(f"clf in shape = {x.shape}")
    #     # tic = time.perf_counter()
    #     pred = self.clf(features)  # batch_size, n_classes
    #     # toc = time.perf_counter()
    #     # print(f"Classifier runtime: {toc - tic:0.4f} seconds")
    #
    #     # print(f"Pred size: {pred.size()}")
    #     return pred
