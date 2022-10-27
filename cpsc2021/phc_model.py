
from typing import Any, NoReturn
from copy import deepcopy

import torch
from torch import nn

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).absolute().parents[2]))

from torch_ecg.cfg import CFG, DEFAULTS
from torch_ecg.utils.misc import dict_to_str
from torch_ecg.models._nets import MLP, StackedLSTM, SEBlock
from torch_ecg.model_configs.ecg_seq_lab_net import ECG_SEQ_LAB_NET_CONFIG

from .model import ECG_SEQ_LAB_NET_CPSC2021
from models.phc_nets import (
    PHM_MLP,
    PHM_SEBlock,
)
from models.multi_scopic_phc import MultiScopicPHCNN
from models.densenet_phc import DenseNet_PHC
from models.resnet_phc import ResNet_PHC

if DEFAULTS.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)

__all__ = [
    "ECG_SEQ_LAB_NET_CPSC2021_PHC",
]


class ECG_SEQ_LAB_NET_CPSC2021_PHC(ECG_SEQ_LAB_NET_CPSC2021):
    """ """

    __DEBUG__ = True
    __name__ = "ECG_SEQ_LAB_NET_CPSC2021_PHC"
    __DEFAULT_CONFIG__ = {"recover_length": False}
    __DEFAULT_CONFIG__.update(deepcopy(ECG_SEQ_LAB_NET_CONFIG))

    def __init__(self, config: CFG, **kwargs: Any) -> NoReturn:
        """

        Parameters
        ----------
        config: dict,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file
        """

        # ------------------  ECG_CRNN
        super().__init__(config)
        # self.classes = list(config.classes)
        # self.n_classes = len(config.classes)
        # self.n_leads = config.n_leads
        # self.config = deepcopy(config.seq_lab_phc)
        # self.config.update(deepcopy(config) or {})

        # ECG_SEQ_LAB_NET
        # _config = CFG(deepcopy(self.__DEFAULT_CONFIG__))
        # _config.update(deepcopy(config[config.model_name]) or {})
        # _config.global_pool = "none"
        # _config.recover_length = False
        # _config.global_pool = "none"
        # # ECG_SEQ_LAB_NET_CPSC2021
        # self.task = config.task
        # if _config.reduction == 1:
        #     _config.recover_length = True
        #
        # self.config = deepcopy(ECG_CRNN_CONFIG)
        # self.config.update(deepcopy(_config) or {})

        if self.__DEBUG__:
            print(f"classes (totally {self.n_classes}) for prediction:{self.classes}")
            print(
                f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}"
            )
            debug_input_len = 4000

        cnn_choice = self.config.cnn.name.lower()
        cnn_config = self.config.cnn[self.config.cnn.name]
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
        rnn_input_size = self.cnn.compute_output_shape(None, None)[1]

        if self.__DEBUG__:
            cnn_output_shape = self.cnn.compute_output_shape(debug_input_len, None)
            print(
                f"cnn output shape (batch_size, features, seq_len) = {cnn_output_shape}, "
                f"given input_len = {debug_input_len}"
            )

        if self.config.rnn.name.lower() == "none":
            self.rnn = None
            attn_input_size = rnn_input_size
        elif self.config.rnn.name.lower() == "lstm":
            self.rnn = StackedLSTM(
                input_size=rnn_input_size,
                hidden_sizes=self.config.rnn.lstm.hidden_sizes,
                bias=self.config.rnn.lstm.bias,
                dropouts=self.config.rnn.lstm.dropouts,
                bidirectional=self.config.rnn.lstm.bidirectional,
                return_sequences=self.config.rnn.lstm.retseq,
            )
            attn_input_size = self.rnn.compute_output_shape(None, None)[-1]
        else:
            raise NotImplementedError

        print(f"Attention module: {self.config.attn.name.lower()}")
        # attention
        if self.config.rnn.name.lower() == "lstm" and not self.config.rnn.lstm.retseq:
            self.attn = None
            clf_input_size = attn_input_size
            if self.config.attn.name.lower() != "none":
                print(
                    f"since `retseq` of rnn is False, hence attention `{self.config.attn.name}` is ignored"
                )
        elif self.config.attn.name.lower() == "none":
            self.attn = None
            clf_input_size = attn_input_size
        elif self.config.attn.name.lower() == "se":  # squeeze_excitation
            self.attn = PHM_SEBlock(
                n=self.n_leads,
                in_channels=attn_input_size,
                reduction=self.config.attn.se.reduction,
                activation=self.config.attn.se.activation,
                kw_activation=self.config.attn.se.kw_activation,
                bias=self.config.attn.se.bias,
            )
            # self.attn = SEBlock(
            #     in_channels=attn_input_size,
            #     reduction=self.config.attn.se.reduction,
            #     activation=self.config.attn.se.activation,
            #     kw_activation=self.config.attn.se.kw_activation,
            #     bias=self.config.attn.se.bias,
            # )
            clf_input_size = self.attn.compute_output_shape(None, None)[1]
        else:
            raise NotImplementedError

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
        if self.config.clf.name.lower() == "mlp":
            self.clf = PHM_MLP(
                n=self.n_leads,
                in_channels=clf_input_size,
                out_channels=self.config.clf.out_channels + [self.n_classes],
                activation=self.config.clf.activation,
                bias=self.config.clf.bias,
                dropouts=self.config.clf.dropouts,
                skip_last_activation=True,
            )
            # self.clf = MLP(
            #     in_channels=clf_input_size,
            #     out_channels=self.config.clf.out_channels + [self.n_classes],
            #     activation=self.config.clf.activation,
            #     bias=self.config.clf.bias,
            #     dropouts=self.config.clf.dropouts,
            #     skip_last_activation=True,
            # )

        # for inference
        # classification: if single-label, use softmax; otherwise (multi-label) use sigmoid
        # sequence tagging: if background counted in `classes`, use softmax; otherwise use sigmoid
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)
