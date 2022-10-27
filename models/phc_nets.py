"""
Adapted from: https://github.com/DeepPSP/torch_ecg/blob/master/torch_ecg/models/_nets.py

basic building blocks, for 1d signal (time series)
"""

from copy import deepcopy
from itertools import repeat
from inspect import isclass
from numbers import Real
from typing import Any, NoReturn, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from deprecate_kwargs import deprecate_kwargs
from torch import Tensor, nn
from torch.nn.utils.rnn import PackedSequence

from torch_ecg.cfg import CFG
from torch_ecg.utils.misc import list_sum
from torch_ecg.utils.utils_nn import SizeMixin  # compute_output_shape,
from torch_ecg.utils.utils_nn import (
    compute_avgpool_output_shape,
    compute_conv_output_shape,
    compute_maxpool_output_shape,
)
from torch_ecg.models._nets import SeparableConv
from torch_ecg.models._nets import Initializers, get_activation
from .phc_layers import PHConv1D, PHMLinear


_DEFAULT_CONV_CONFIGS = CFG(
    norm=True,
    activation="relu",
    kw_activation={"inplace": True},
    kernel_initializer="he_normal",
    kw_initializer={},
    ordering="cba",
    conv_type=None,
    width_multiplier=1.0,
)


# ---------------------------------------------
# PHM

class PHM_SeqLin(nn.Sequential, SizeMixin):
    """
    Sequential linear,
    might be useful in learning non-linear classifying hyper-surfaces

    """

    __DEBUG__ = False
    __name__ = "PHM_SeqLin"

    def __init__(
        self,
        n: int,
        in_channels: int,
        out_channels: Sequence[int],
        activation: Union[str, nn.Module] = "relu",
        kernel_initializer: Optional[str] = None,
        bias: bool = True,
        dropouts: Union[float, Sequence[float]] = 0.0,
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        out_channels: sequence of int,
            number of ouput channels for each linear layer
        activation: str or nn.Module, default "relu",
            name of activation after each linear layer
        kernel_initializer: str, optional,
            name of kernel initializer for `weight` of each linear layer
        bias: bool, default True,
            if True, each linear layer will have a learnable bias vector
        dropouts: float or sequence of float, default 0,
            dropout ratio(s) (if > 0) after each (activation after each) linear layer
        kwargs: dict, optional,
            extra parameters

        """
        super().__init__()
        self.__n = n
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__num_layers = len(self.__out_channels)
        kw_activation = kwargs.get("kw_activation", {})
        kw_initializer = kwargs.get("kw_initializer", {})
        act_layer = get_activation(activation)
        if not isclass(act_layer):
            raise TypeError("`activation` must be a class or str, not an instance")
        self.__activation = act_layer.__name__
        if kernel_initializer:
            if kernel_initializer.lower() in Initializers.keys():
                self.__kernel_initializer = Initializers[kernel_initializer.lower()]
            else:
                raise ValueError(f"initializer `{kernel_initializer}` not supported")
        else:
            self.__kernel_initializer = None
        self.__bias = bias
        if isinstance(dropouts, Real):
            if self.__num_layers > 1:
                self.__dropouts = list(repeat(dropouts, self.__num_layers - 1)) + [0.0]
            else:
                self.__dropouts = [dropouts]
        else:
            self.__dropouts = dropouts
            assert (
                len(self.__dropouts) == self.__num_layers
            ), f"`out_channels` indicates {self.__num_layers} linear layers, while `dropouts` indicates {len(self.__dropouts)}"
        self.__skip_last_activation = kwargs.get("skip_last_activation", False)

        # print(f"Linear in: {self.__in_channels}, out: {self.__out_channels}")
        lin_in_channels = self.__in_channels
        for idx in range(self.__num_layers):
            if idx == self.__num_layers-1:
                lin_layer = PHMLinear(
                    n=1,
                    in_features=lin_in_channels,
                    out_features=self.__out_channels[idx],
                )
            else:
                lin_layer = PHMLinear(
                    n=self.__n,
                    in_features=lin_in_channels,
                    out_features=self.__out_channels[idx],
                )
            if self.__kernel_initializer:
                self.__kernel_initializer(lin_layer.weight, **kw_initializer)
            self.add_module(
                f"lin_{idx}",
                lin_layer,
            )
            if idx < self.__num_layers - 1 or not self.__skip_last_activation:
                self.add_module(
                    f"act_{idx}",
                    act_layer(**kw_activation),
                )
            if self.__dropouts[idx] > 0:
                self.add_module(
                    f"dropout_{idx}",
                    nn.Dropout(self.__dropouts[idx]),
                )
            lin_in_channels = self.__out_channels[idx]

    def forward(self, input: Tensor) -> Tensor:
        """
        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels) or (batch_size, seq_len, n_channels)

        Returns
        -------
        output: Tensor,
            of shape (batch_size, n_channels) or (batch_size, seq_len, n_channels),
            ndim in accordance with `input`

        """
        # print(f"SeqLin in_channels: {self.__in_channels}")
        # print(f"SeqLin out_channels: {self.__out_channels}")
        output = super().forward(input)
        return output

    def compute_output_shape(
        self,
        seq_len: Optional[int] = None,
        batch_size: Optional[int] = None,
        input_seq: bool = True,
    ) -> Sequence[Union[int, None]]:
        """

        Parameters
        ----------
        seq_len: int, optional,
            length of the 1d sequence,
        batch_size: int, optional,
            the batch size, can be None
        input_seq: bool, default True,
            if True, the input is a sequence (Tensor of dim 3) of vectors of features,
            otherwise a vector of features (Tensor of dim 2)

        Returns
        -------
        output_shape: sequence,
            the output shape, given `seq_len` and `batch_size`

        """
        if input_seq:
            output_shape = (batch_size, seq_len, self.__out_channels[-1])
        else:
            output_shape = (batch_size, self.__out_channels[-1])
        return output_shape

    @property
    def in_channels(self) -> int:
        return self.__in_channels


class PHM_MLP(PHM_SeqLin):
    """
    multi-layer perceptron,
    alias for sequential linear block

    """

    __DEBUG__ = False
    __name__ = "PHM_MLP"

    def __init__(
        self,
        n : int,
        in_channels: int,
        out_channels: Sequence[int],
        activation: str = "relu",
        kernel_initializer: Optional[str] = None,
        bias: bool = True,
        dropouts: Union[float, Sequence[float]] = 0.0,
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        out_channels: sequence of int,
            number of ouput channels for each linear layer
        activation: str, default "relu",
            name of activation after each linear layer
        kernel_initializer: str, optional,
            name of kernel initializer for `weight` of each linear layer
        bias: bool, default True,
            if True, each linear layer will have a learnable bias vector
        dropouts: float or sequence of float, default 0,
            dropout ratio(s) (if > 0) after each (activation after each) linear layer
        kwargs: dict, optional,
            extra parameters

        """
        super().__init__(
            n,
            in_channels,
            out_channels,
            activation,
            kernel_initializer,
            bias,
            dropouts,
            **kwargs,
        )


# ---------------------------------------------
# attention mechanism: PHM SEBlock

class PHM_SEBlock(nn.Module, SizeMixin):
    """

    Squeeze-and-Excitation Block

    References
    ----------
    [1] J. Hu, L. Shen, S. Albanie, G. Sun and E. Wu, "Squeeze-and-Excitation Networks," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 42, no. 8, pp. 2011-2023, 1 Aug. 2020, doi: 10.1109/TPAMI.2019.2913372.
    [2] J. Hu, L. Shen and G. Sun, "Squeeze-and-Excitation Networks," 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT, 2018, pp. 7132-7141, doi: 10.1109/CVPR.2018.00745.
    [3] https://github.com/hujie-frank/SENet
    [4] https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py

    """

    __DEBUG__ = False
    __name__ = "PHM_SEBlock"
    __DEFAULT_CONFIG__ = CFG(
        bias=False, activation="relu", kw_activation={"inplace": True}, dropouts=0.0
    )

    def __init__(self, n : int, in_channels: int, reduction: int = 16, **config) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        reduction: int, default 16,
            reduction ratio of mid-channels to `in_channels`
        config: dict,
            other parameters, including
            activation choices, weight initializer, dropouts, etc.
            for the linear layers
        """
        super().__init__()
        self.__n = n
        self.__in_channels = in_channels
        self.__mid_channels = in_channels // reduction
        self.__out_channels = in_channels
        self.config = CFG(deepcopy(self.__DEFAULT_CONFIG__))
        self.config.update(deepcopy(config))

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            PHM_SeqLin(
                n=self.__n,
                in_channels=self.__in_channels,
                out_channels=[self.__mid_channels, self.__out_channels],
                activation=self.config.activation,
                kw_activation=self.config.kw_activation,
                bias=self.config.bias,
                dropouts=self.config.dropouts,
                skip_last_activation=True,
            ),
            nn.Sigmoid(),
        )

    def forward(self, input: Tensor) -> Tensor:
        """

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)

        """
        batch_size, n_channels, seq_len = input.shape
        y = self.avg_pool(input).squeeze(-1)  # --> batch_size, n_channels
        y = self.fc(y).unsqueeze(-1)  # --> batch_size, n_channels, 1
        # output = input * y.expand_as(input)  # equiv. to the following
        # (batch_size, n_channels, seq_len) x (batch_size, n_channels, 1)
        output = input * y  # --> (batch_size, n_channels, seq_len)
        return output

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """

        Parameters
        ----------
        seq_len: int, optional,
            length of the 1d sequence,
            if is None, then the input is composed of single feature vectors for each batch
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
        output_shape: sequence,
            the output shape, given `seq_len` and `batch_size`

        """
        return (batch_size, self.__in_channels, seq_len)

    @property
    def in_channels(self) -> int:
        return self.__in_channels


# ---------------------------------------------
# PHC layers

class PHConv_Bn_Activation(nn.Sequential, SizeMixin):
    """

    1d convolution --> batch normalization (optional) -- > activation (optional),
    orderings can be adjusted,
    with "same" padding as default padding

    """

    __name__ = "PHConv_Bn_Activation"

    def __init__(
        self,
        n: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        batch_norm: Union[bool, str, nn.Module] = True,
        activation: Optional[Union[str, nn.Module]] = None,
        kernel_initializer: Optional[Union[str, callable]] = None,
        bias: bool = True,
        ordering: str = "cba",
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input signal
        out_channels: int,
            number of channels produced by the convolution
        kernel_size: int,
            size (length) of the convolution kernel
        stride: int,
            stride (subsample length) of the convolution
        padding: int, optional,
            zero-padding added to both sides of the input
        dilation: int, default 1,
            spacing between the kernel points
        groups: int, default 1,
            connection pattern (of channels) of the inputs and outputs
        batch_norm: bool or str or Module, default True,
            (batch) normalization, or other normalizations, e.g. group normalization
            (the name of) the Module itself or (if is bool) whether or not to use `nn.BatchNorm1d`
        activation: str or Module, optional,
            name or Module of the activation,
            if is str, can be one of
            "mish", "swish", "relu", "leaky", "leaky_relu", "linear", "hardswish", "relu6"
            "linear" is equivalent to `activation=None`
        kernel_initializer: str or callable (function), optional,
            a function to initialize kernel weights of the convolution,
            or name or the initialzer, can be one of the keys of `Initializers`
        bias: bool, default True,
            if True, adds a learnable bias to the output
        ordering: str, default "cba",
            ordering of the layers, case insensitive
        kwargs: dict, optional,
            other key word arguments, including
            `conv_type`, `kw_activation`, `kw_initializer`, `kw_bn`,
            `alpha` (alias `width_multiplier`), etc.

        NOTE that if `padding` is not specified (default None),
        then the actual padding used for the convolutional layer is automatically computed
        to fit the "same" padding (not actually "same" for even kernel sizes)

        """
        super().__init__()
        self.__n = n
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__kernel_size = kernel_size
        self.__stride = stride
        self.__dilation = dilation
        if padding is None:
            # "same" padding
            self.__padding = (self.__dilation * (self.__kernel_size - 1)) // 2
        elif isinstance(padding, int):
            self.__padding = padding
        self.__groups = groups
        self.__bias = bias
        self.__ordering = ordering.lower()
        assert "c" in self.__ordering

        kw_activation = kwargs.get("kw_activation", {})
        kw_initializer = kwargs.get("kw_initializer", {})
        kw_bn = kwargs.get("kw_bn", {})
        self.__conv_type = kwargs.get("conv_type", None)
        if isinstance(self.__conv_type, str):
            self.__conv_type = self.__conv_type.lower()
        self.__width_multiplier = (
            kwargs.get("width_multiplier", None) or kwargs.get("alpha", None) or 1.0
        )
        self.__out_channels = int(self.__width_multiplier * self.__out_channels)
        assert self.__out_channels % self.__groups == 0, (
            f"width_multiplier (input is {self.__width_multiplier}) makes "
            f"`out_channels` (= {self.__out_channels}) "
            f"not divisible by `groups` (= {self.__groups})"
        )

        if self.__conv_type is None:
            conv_layer = PHConv1D(
                n=self.__n,
                in_features=self.__in_channels,
                out_features=self.__out_channels,
                kernel_size=self.__kernel_size,
                padding=self.__padding,
                stride=self.__stride,
                dilation=self.__dilation,
                cuda=True,
                groups=self.__groups,
            )
            if kernel_initializer:
                if callable(kernel_initializer):
                    kernel_initializer(conv_layer.weight)
                elif (
                    isinstance(kernel_initializer, str)
                    and kernel_initializer.lower() in Initializers.keys()
                ):
                    Initializers[kernel_initializer.lower()](
                        conv_layer.weight, **kw_initializer
                    )
                else:
                    raise ValueError(
                        f"initializer `{kernel_initializer}` not supported"
                    )
        elif self.__conv_type == "separable":
            conv_layer = PH_SeparableConv(
                n=self.__n,
                in_channels=self.__in_channels,
                # out_channels=self.__out_channels,
                out_channels=out_channels,  # note the existence of `width_multiplier` in `kwargs`
                kernel_size=self.__kernel_size,
                stride=self.__stride,
                padding=self.__padding,
                dilation=self.__dilation,
                groups=self.__groups,
                kernel_initializer=kernel_initializer,
                bias=self.__bias,
                **kwargs,
            )
            # conv_layer = SeparableConv(
            #     in_channels=self.__in_channels,
            #     # out_channels=self.__out_channels,
            #     out_channels=out_channels,  # note the existence of `width_multiplier` in `kwargs`
            #     kernel_size=self.__kernel_size,
            #     stride=self.__stride,
            #     padding=self.__padding,
            #     dilation=self.__dilation,
            #     groups=self.__groups,
            #     kernel_initializer=kernel_initializer,
            #     bias=self.__bias,
            #     **kwargs,
            # )
        else:
            raise NotImplementedError(
                f"convolution of type {self.__conv_type} not implemented yet!"
            )

        if "b" in self.__ordering and self.__ordering.index(
            "c"
        ) < self.__ordering.index("b"):
            bn_in_channels = self.__out_channels
        else:
            bn_in_channels = self.__in_channels
        if batch_norm:
            if isinstance(batch_norm, bool):
                bn_layer = nn.BatchNorm1d(bn_in_channels, **kw_bn)
            elif isinstance(batch_norm, str):
                if batch_norm.lower() in [
                    "batch_norm",
                    "batch_normalization",
                ]:
                    bn_layer = nn.BatchNorm1d(bn_in_channels, **kw_bn)
                elif batch_norm.lower() in [
                    "instance_norm",
                    "instance_normalization",
                ]:
                    bn_layer = nn.InstanceNorm1d(bn_in_channels, **kw_bn)
                elif batch_norm.lower() in [
                    "group_norm",
                    "group_normalization",
                ]:
                    bn_layer = nn.GroupNorm(self.__groups, bn_in_channels, **kw_bn)
                elif batch_norm.lower() in [
                    "layer_norm",
                    "layer_normalization",
                ]:
                    bn_layer = nn.LayerNorm(**kw_bn)
                else:
                    raise ValueError(
                        f"normalization method {batch_norm} not supported yet!"
                    )
            else:
                bn_layer = batch_norm
        else:
            bn_layer = None

        act_layer = get_activation(activation, kw_activation)
        if act_layer is not None:
            act_name = f"activation_{type(act_layer).__name__}"

        if self.__ordering in ["cba", "cb", "ca"]:
            self.add_module("conv1d", conv_layer)
            if bn_layer:
                self.add_module("batch_norm", bn_layer)
            if act_layer:
                self.add_module(act_name, act_layer)
        elif self.__ordering in ["cab"]:
            self.add_module("conv1d", conv_layer)
            self.add_module(act_name, act_layer)
            self.add_module("batch_norm", bn_layer)
        elif self.__ordering in ["bac", "bc"]:
            if bn_layer:
                self.add_module("batch_norm", bn_layer)
            if act_layer:
                self.add_module(act_name, act_layer)
            self.add_module("conv1d", conv_layer)
        elif self.__ordering in ["acb", "ac"]:
            if act_layer:
                self.add_module(act_name, act_layer)
            self.add_module("conv1d", conv_layer)
            if bn_layer:
                self.add_module("batch_norm", bn_layer)
        elif self.__ordering in ["bca"]:
            if bn_layer:
                self.add_module("batch_norm", bn_layer)
            self.add_module("conv1d", conv_layer)
            if act_layer:
                self.add_module(act_name, act_layer)
        else:
            raise ValueError(f"ordering \042{self.__ordering}\042 not supported!")

        # print(vars(self))
        # print('Done!')

    def _assign_weights_lead_wise(
        self, other: "PHConv_Bn_Activation", indices: Sequence[int]
    ) -> NoReturn:
        """

        assign weights of `self` to `other` according to `indices` in the `lead-wise` manner

        Parameters
        ----------
        other: `PHConv_Bn_Activation`,
            the target instance of `PHConv_Bn_Activation`
        indices: sequence of int,
            the indices of weights (weight and bias (if not None))
            to be assigned to `other`
        """
        assert (
            self.conv_type is None and other.conv_type is None
        ), "only normal convolution supported!"
        assert (
            self.in_channels * other.groups == other.in_channels * self.groups
        ), "in_channels should be in proportion to groups"
        assert (
            self.out_channels * other.groups == other.out_channels * self.groups
        ), "out_channels should be in proportion to groups"
        assert (
            len(indices) == other.groups
        ), "`indices` should have length equal to `groups` of `other`"
        assert len(set(indices)) == len(
            indices
        ), "`indices` should not contain duplicates"
        assert not any([isinstance(m, nn.LayerNorm) for m in self]) and not any(
            [isinstance(m, nn.LayerNorm) for m in other]
        ), "Lead-wise assignment of weights is not supported for the existence of `LayerNorm` layers"
        for field in [
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "bias",
            "ordering",
        ]:
            if getattr(self, field) != getattr(other, field):
                raise ValueError(
                    f"{field} of self and other should be the same, "
                    f"but got {getattr(self, field)} and {getattr(other, field)}"
                )
        units = self.out_channels // self.groups
        out_indices = list_sum([[i * units + j for j in range(units)] for i in indices])
        for m, om in zip(self, other):
            if isinstance(
                m, (PHConv1D, nn.Conv1d, nn.BatchNorm1d, nn.GroupNorm, nn.InstanceNorm1d)
            ):
                om.weight.data = m.weight.data[out_indices].clone()
                if m.bias is not None:
                    om.bias.data = m.bias.data[out_indices].clone()

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """

        Parameters
        ----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
        output_shape: sequence,
            the output shape, given `seq_len` and `batch_size`

        """
        if self.__conv_type is None:
            input_shape = [batch_size, self.__in_channels, seq_len]
            output_shape = compute_conv_output_shape(
                input_shape=input_shape,
                num_filters=self.__out_channels,
                kernel_size=self.__kernel_size,
                stride=self.__stride,
                dilation=self.__dilation,
                padding=self.__padding,
                channel_last=False,
            )
        elif self.__conv_type in [
            "separable",
            "anti_alias",
            "aa",
        ]:
            output_shape = self.conv1d.compute_output_shape(seq_len, batch_size)
        return output_shape

    @property
    def in_channels(self) -> int:
        return self.__in_channels

    @property
    def out_channels(self) -> int:
        return self.__out_channels

    @property
    def kernel_size(self) -> int:
        return self.__kernel_size

    @property
    def stride(self) -> int:
        return self.__stride

    @property
    def padding(self) -> int:
        return self.__padding

    @property
    def dilation(self) -> int:
        return self.__dilation

    @property
    def groups(self) -> int:
        return self.__groups

    @property
    def bias(self) -> bool:
        return self.__bias

    @property
    def ordering(self) -> str:
        return self.__ordering

    @property
    def conv_type(self) -> Optional[str]:
        return self.__conv_type


class PHC_DownSample(nn.Sequential, SizeMixin):
    """

    NOTE: this down sampling module allows changement of number of channels,
    via additional convolution, with some abuse of terminology

    the "conv" mode is not simply down "sampling" if `group` != `in_channels`

    """

    __name__ = "PHC_DownSample"
    __MODES__ = [
        "max",
        "avg",
        "lp",
        "lse",
        "conv",
        "nearest",
        "area",
        "linear",
        "blur",
    ]

    @deprecate_kwargs([["norm", "batch_norm"]])
    def __init__(
        self,
        n: int,
        down_scale: int,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: Optional[int] = None,
        groups: Optional[int] = None,
        padding: int = 0,
        batch_norm: Union[bool, nn.Module] = False,
        mode: str = "max",
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters
        ----------
        down_scale: int,
            scale (in terms of stride) of down sampling
        in_channels: int,
            number of channels of the input
        out_channels: int, optional,
            number of channels of the output
        kernel_size: int, optional,
            kernel size of down sampling,
            if not specified, defaults to `down_scale`,
        groups: int, optional,
            connection pattern (of channels) of the inputs and outputs
        padding: int, default 0,
            zero-padding added to both sides of the input
        batch_norm: bool or Module, default False,
            (batch) normalization,
            the Module itself or (if is bool) whether or not to use `nn.BatchNorm1d`
        mode: str, default "max",
            can be one of `self.__MODES__`

        """
        super().__init__()
        self.__n = n
        self.__mode = mode.lower()
        assert self.__mode in self.__MODES__
        self.__down_scale = down_scale
        self.__kernel_size = kernel_size or down_scale
        self.__in_channels = in_channels
        self.__out_channels = out_channels or in_channels
        self.__groups = groups or self.__in_channels
        self.__padding = padding

        if self.__mode == "max":
            if self.__in_channels == self.__out_channels:
                down_layer = nn.MaxPool1d(
                    kernel_size=self.__kernel_size,
                    stride=self.__down_scale,
                    padding=self.__padding,
                )
            else:
                down_layer = nn.Sequential(
                    nn.MaxPool1d(
                        kernel_size=self.__kernel_size,
                        stride=self.__down_scale,
                        padding=self.__padding,
                    ),
                    # nn.Conv1d(
                    #     self.__in_channels,
                    #     self.__out_channels,
                    #     kernel_size=1,
                    #     groups=self.__groups,
                    #     bias=False,
                    # ),
                    PHConv1D(n=self.__n,
                             in_features=self.__in_channels,
                             out_features=self.__out_channels,
                             kernel_size=1,
                             groups=self.__groups,
                             cuda=True,
                    ),
                )
        elif self.__mode == "avg":
            if self.__in_channels == self.__out_channels:
                down_layer = nn.AvgPool1d(
                    kernel_size=self.__kernel_size,
                    stride=self.__down_scale,
                    padding=self.__padding,
                )
            else:
                down_layer = nn.Sequential(
                    (
                        nn.AvgPool1d(
                            kernel_size=self.__kernel_size,
                            stride=self.__down_scale,
                            padding=self.__padding,
                        ),
                        # nn.Conv1d(
                        #     self.__in_channels,
                        #     self.__out_channels,
                        #     kernel_size=1,
                        #     groups=self.__groups,
                        #     bias=False,
                        # ),
                        PHConv1D(n=self.__n,
                                 in_features=self.__in_channels,
                                 out_features=self.__out_channels,
                                 kernel_size=1,
                                 cuda=True
                        ),
                    )
                )
        elif self.__mode == "conv":
            # down_layer = nn.Conv1d(
            #     in_channels=self.__in_channels,
            #     out_channels=self.__out_channels,
            #     kernel_size=1,
            #     groups=self.__groups,
            #     bias=False,
            #     stride=self.__down_scale,
            # )
            down_layer = PHConv1D(
                n=self.__n,
                in_features=self.__in_channels,
                out_features=self.__out_channels,
                kernel_size=1,
                stride=self.__down_scale,
                cuda=True
            )
        elif self.__mode == "nearest":
            raise NotImplementedError
        elif self.__mode == "area":
            raise NotImplementedError
        elif self.__mode == "linear":
            raise NotImplementedError
        elif self.__mode == "blur":
            raise NotImplementedError  # available in torch_ecg
        else:
            down_layer = None
        if down_layer:
            self.add_module(
                "down_sample",
                down_layer,
            )

        if batch_norm:
            bn_layer = (
                nn.BatchNorm1d(self.__out_channels)
                if isinstance(batch_norm, bool)
                else batch_norm(self.__out_channels)
            )
            self.add_module(
                "batch_normalization",
                bn_layer,
            )

    def forward(self, input: Tensor) -> Tensor:
        """
        use the forward method of `nn.Sequential`

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        """
        if self.__mode in [
            "max",
            "avg",
            "conv",
            "blur",
        ]:
            output = super().forward(input)
        else:
            # align_corners = False if mode in ["nearest", "area"] else True
            output = F.interpolate(
                input=input,
                scale_factor=1 / self.__down_scale,
                mode=self.__mode,
                # align_corners=align_corners,
            )

        # print(f"Downsample input size: {input.size()}, output size: {output.size()}")
        return output

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """

        Parameters
        ----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
        output_shape: sequence,
            the output shape, given `seq_len` and `batch_size`

        """
        if self.__mode == "conv":
            out_seq_len = compute_conv_output_shape(
                input_shape=(batch_size, self.__in_channels, seq_len),
                stride=self.__down_scale,
                padding=self.__padding,
            )[-1]
        elif self.__mode == "max":
            out_seq_len = compute_maxpool_output_shape(
                input_shape=(batch_size, self.__in_channels, seq_len),
                kernel_size=self.__kernel_size,
                stride=self.__down_scale,
                padding=self.__padding,
            )[-1]
        elif self.__mode == "blur":
            if self.__in_channels == self.__out_channels:
                out_seq_len = self.down_sample.compute_output_shape(
                    seq_len, batch_size
                )[-1]
            else:
                out_seq_len = self.down_sample[0].compute_output_shape(
                    seq_len, batch_size
                )[-1]
        elif self.__mode in [
            "avg",
            "nearest",
            "area",
            "linear",
        ]:
            out_seq_len = compute_avgpool_output_shape(
                input_shape=(batch_size, self.__in_channels, seq_len),
                kernel_size=self.__kernel_size,
                stride=self.__down_scale,
                padding=self.__padding,
            )[-1]
        output_shape = (batch_size, self.__out_channels, out_seq_len)
        return output_shape

    @property
    def in_channels(self) -> int:
        return self.__in_channels


class PH_SeparableConv(nn.Sequential, SizeMixin):
    """

    (Super-)Separable Convolution

    References
    ----------
    [1] Kaiser, Lukasz, Aidan N. Gomez, and Francois Chollet. "Depthwise separable convolutions for neural machine translation." arXiv preprint arXiv:1706.03059 (2017).
    [2] https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py

    """

    __DEBUG__ = False
    __name__ = "PH_SeparableConv"

    def __init__(
        self,
        n : int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        kernel_initializer: Optional[Union[str, callable]] = None,
        bias: bool = True,
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input signal
        out_channels: int,
            number of channels produced by the convolution
        kernel_size: int,
            size (length) of the convolution kernel
        stride: int,
            stride (subsample length) of the convolution
        padding: int, optional,
            zero-padding added to both sides of the input
        dilation: int, default 1,
            spacing between the kernel points
        groups: int, default 1,
            connection pattern (of channels) of the inputs and outputs
        kernel_initializer: str or callable (function), optional,
            a function to initialize kernel weights of the convolution,
            or name or the initialzer, can be one of the keys of `Initializers`
        bias: bool, default True,
            if True, adds a learnable bias to the output
        kwargs: dict, optional,
            extra parameters, including `depth_multiplier`, `width_multiplier` (alias `alpha`), etc.

        """
        super().__init__()
        self.__n = n
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__kernel_size = kernel_size
        self.__stride = stride
        self.__dilation = dilation
        if padding is None:
            # "same" padding
            self.__padding = (self.__dilation * (self.__kernel_size - 1)) // 2
        elif isinstance(padding, int):
            self.__padding = padding
        self.__groups = groups
        self.__bias = bias
        kw_initializer = kwargs.get("kw_initializer", {})
        self.__depth_multiplier = kwargs.get("depth_multiplier", 1)
        dc_out_channels = int(self.__in_channels * self.__depth_multiplier)
        assert (
            dc_out_channels % self.__in_channels == 0
        ), f"depth_multiplier (input is {self.__depth_multiplier}) should be positive integers"
        self.__width_multiplier = (
            kwargs.get("width_multiplier", None) or kwargs.get("alpha", None) or 1
        )
        self.__out_channels = int(self.__width_multiplier * self.__out_channels)
        assert self.__out_channels % self.__groups == 0, (
            f"width_multiplier (input is {self.__width_multiplier}) "
            f"makes `out_channels` not divisible by `groups` (= {self.__groups})"
        )

        self.add_module(
            "depthwise_conv",
            nn.Conv1d(
                in_channels=self.__in_channels,
                out_channels=dc_out_channels,
                kernel_size=self.__kernel_size,
                stride=self.__stride,
                padding=self.__padding,
                dilation=self.__dilation,
                groups=self.__in_channels,
                bias=self.__bias,
            ),
            # PHConv1D(
            #     n=self.__n,
            #     in_features=self.__in_channels,
            #     out_features=dc_out_channels,
            #     kernel_size=self.__kernel_size,
            #     stride=self.__stride,
            #     padding=self.__padding,
            #     dilation=self.__dilation,
            #     groups=self.__in_channels,
            # ),
        )
        self.add_module(
            "pointwise_conv",
            # nn.Conv1d(
            #     in_channels=dc_out_channels,
            #     out_channels=self.__out_channels,
            #     groups=self.__groups,
            #     bias=self.__bias,
            #     kernel_size=1,
            #     stride=1,
            #     padding=0,
            #     dilation=1,
            # ),
            PHConv1D(
                n=self.__n,
                in_features=dc_out_channels,
                out_features=self.__out_channels,
                groups=self.__groups,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
            ),
        )

        if kernel_initializer:
            if callable(kernel_initializer):
                for module in self:
                    kernel_initializer(module.weight)
            elif (
                isinstance(kernel_initializer, str)
                and kernel_initializer.lower() in Initializers.keys()
            ):
                for module in self:
                    Initializers[kernel_initializer.lower()](
                        module.weight, **kw_initializer
                    )
            else:  # TODO: add more initializers
                raise ValueError(f"initializer `{kernel_initializer}` not supported")

    def forward(self, input: Tensor) -> Tensor:
        """

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channles, seq_len)

        Returns
        -------
        output: Tensor,
            of shape (batch_size, n_channles, seq_len)

        """
        output = super().forward(input)
        return output

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """

        Parameters
        ----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
        output_shape: sequence,
            the output shape, given `seq_len` and `batch_size`

        """
        # depthwise_conv
        output_shape = compute_conv_output_shape(
            input_shape=(batch_size, self.__in_channels, seq_len),
            num_filters=self.__in_channels,
            kernel_size=self.__kernel_size,
            stride=self.__stride,
            padding=self.__padding,
            dilation=self.__dilation,
        )
        # pointwise_conv
        output_shape = compute_conv_output_shape(
            input_shape=output_shape,
            num_filters=self.__out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
        )
        return output_shape

    @property
    def in_channels(self) -> int:
        return self.__in_channels


# ---------------------------------------------
class GRU(nn.Sequential, SizeMixin):
    """
    GRU
    """

    __DEBUG__ = False
    __name__ = "GRU"

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: Union[Sequence[bool], bool] = True,
        dropouts: Union[float, Sequence[float]] = 0.0,
        bidirectional: bool = True,
        return_sequences: bool = True,
        batch_first: bool = False,
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters
        ----------
        input_size: int,
            the number of features in the input
        hidden_size: int,
            the number of features in the hidden state of the GRU layer
        bias: bool, or sequence of bool, default True,
            use bias weights or not
        dropouts: float or sequence of float, default 0.0,
            if non-zero, introduces a `Dropout` layer on the outputs of each
            GRU layer EXCEPT the last layer, with dropout probability equal to this value
            or corresponding value in the sequence (except for the last GRU layer)
        bidirectional: bool, default True,
            if True, each GRU layer becomes bidirectional
        return_sequences: bool, default True,
            if True, returns the the full output sequence,
            otherwise the last output in the output sequence
        kwargs: dict, optional,
            extra parameters

        """
        super().__init__()
        self.__hidden_size = hidden_size
        self.__dropouts = dropouts
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.bias = bias

        module_name_prefix = "bidirectional_gru" if bidirectional else "gru"
        self.__module_names = []
        _input_size = input_size
        self.add_module(
            name=f"{module_name_prefix}",
            module=nn.GRU(input_size=_input_size,
                    hidden_size=self.__hidden_size,
                    num_layers=self.num_layers,
                    batch_first=self.batch_first,
                    bidirectional=self.bidirectional,
                    dropout=self.__dropouts,
                    bias = self.bias,
                   )
        )
        self.__module_names.append("gru")
        if self.__dropouts > 0 :
            self.add_module(
                name=f"dropout",
                module=nn.Dropout(self.__dropouts),
            )
            self.__module_names.append("dp")

    def forward(
        self,
        input: Union[Tensor, PackedSequence],
        hx: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        """
        keep up with `nn.LSTM.forward`, parameters ref. `nn.LSTM.forward`

        Parameters
        ----------
        input: Tensor,
            of shape (seq_len, batch_size, n_channels)
        hx: 2-tuple of Tensor, optional,

        Returns
        -------
        final_output: Tensor,
            of shape (seq_len, batch_size, n_channels) if `return_sequences` is True,
            otherwise of shape (batch_size, n_channels)

        """
        output, _hx = input, hx
        for idx, (name, module) in enumerate(zip(self.__module_names, self)):
            if name == "dp":
                output = module(output)
            elif name == "gru":
                module.flatten_parameters()
                output, _hx = module(output, _hx)
        if self.return_sequences:
            final_output = output  # seq_len, batch_size, n_direction*hidden_size
        else:
            final_output = output[-1, ...]  # batch_size, n_direction*hidden_size
        return final_output

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """

        Parameters
        ----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
        output_shape: sequence,
            the output shape, given `seq_len` and `batch_size`

        """
        output_size = self.__hidden_size
        if self.bidirectional:
            output_size *= 2
        if self.return_sequences:
            output_shape = (seq_len, batch_size, output_size)
        else:
            output_shape = (batch_size, output_size)
        return output_shape
