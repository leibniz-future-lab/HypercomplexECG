"""
PH layers from https://github.com/eleGAN23/HyperNets

PHC:
E. Grassucci, A. Zhang, and D. Comminiello, “PHNNs: Lightweight neural networks via parameterized hypercomplex
convolutions,” arXiv preprint arXiv:2110.04176v2, 2021.

PHM:
A. Zhang, Y. Tay, S. Zhang, A. Chan, A. T. Luu, S. C. Hui, and J. Fu, “Beyond fully-connected layers with quaternions:
Parameterization of hypercomplex multiplications with 1/n parameters,” in ICLR, 2021.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import NoReturn, Optional, Sequence, Union

from torch_ecg.utils.utils_nn import (
    compute_conv_output_shape,
)

################################
## PHC LAYER: 1D convolutions ##
################################

class PHConv1D(nn.Module):

    def __init__(self, n, in_features, out_features, kernel_size, padding=0, stride=1, dilation=1, cuda=torch.cuda.is_available(), groups=1):
        super(PHConv1D, self).__init__()
        self.n = n
        self.in_features = in_features
        self.out_features = out_features
        self.padding = padding #'same'
        # print(f"padding: {self.padding}")
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.cuda = torch.cuda.is_available() #cuda

        self.bias = nn.Parameter(torch.Tensor(out_features))

        self.A = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))
        self.F = nn.Parameter(torch.nn.init.xavier_uniform_(
            torch.zeros((n, self.out_features // n, self.in_features // n, kernel_size))))
        # self.F = nn.Parameter(torch.nn.init.xavier_uniform_(
        #     torch.zeros((n, self.out_features // n, self.in_features // self.groups // n, kernel_size))))
        self.weight = torch.zeros((self.out_features, self.in_features))
        self.kernel_size = kernel_size

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def kronecker_product1(self, A, F):
        siz1 = torch.Size(torch.tensor(A.shape[-2:]) * torch.tensor(F.shape[-3:-1]))
        siz2 = torch.Size(torch.tensor(F.shape[-1:]))
        res = A.unsqueeze(-1).unsqueeze(-3).unsqueeze(-1) * F.unsqueeze(-3).unsqueeze(-5)
        siz0 = res.shape[:1]
        out = res.reshape(siz0 + siz1 + siz2)
        return out

    def kronecker_product2(self):
        H = torch.zeros((self.out_features, self.in_features, self.kernel_size, self.kernel_size))
        if self.cuda:
            H = H.cuda()
        for i in range(self.n):
            kron_prod = torch.kron(self.A[i], self.F[i]).view(self.out_features, self.in_features, self.kernel_size,
                                                              self.kernel_size)
            H = H + kron_prod
        return H

    def forward(self, input):
        self.weight = torch.sum(self.kronecker_product1(self.A, self.F), dim=0)
        if self.cuda:
            self.weight = self.weight.cuda()

        input = input.type(dtype=self.weight.type())
        out = F.conv1d(input, weight=self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        # print(f"in: {self.in_features}, out: {self.out_features}")
        # print(f"weight size: {self.weight.size()}")
        # print(f"A size: {self.A.size()}, F size: {self.F.size()}")
        # print(f"kronecker size: {self.kronecker_product1(self.A, self.F).size()}")
        # print(f"weight size: {self.weight.size()}")
        # print(f"Input size: {input.size()}, output size: {out.size()}")

        return out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.A, a=math.sqrt(5))
        init.kaiming_uniform_(self.F, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.placeholder)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        out_seq_len = compute_conv_output_shape(
            input_shape=(batch_size, self.in_features, seq_len),
            num_filters=self.out_features,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            channel_last=False,
        )[-1]
        output_shape = (batch_size, self.out_features, out_seq_len)
        return output_shape


########################
## STANDARD PHM LAYER ##
########################

class PHMLinear(nn.Module):

    def __init__(self, n, in_features, out_features, cuda=True):
        super(PHMLinear, self).__init__()
        self.n = n
        self.in_features = in_features
        self.out_features = out_features
        # print(f"in: {in_features}, out: {out_features}")
        self.cuda = cuda

        self.bias = nn.Parameter(torch.Tensor(out_features))

        self.A = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))

        self.S = nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.zeros((n, self.out_features // n, self.in_features // n))))

        self.weight = torch.zeros((self.out_features, self.in_features))
        # print(f"weight size: {self.weight.size()}")
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def kronecker_product1(self, a, b):  # adapted from Bayer Research's implementation
        siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
        res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
        siz0 = res.shape[:-4]
        out = res.reshape(siz0 + siz1)
        return out

    def kronecker_product2(self):
        H = torch.zeros((self.out_features, self.in_features))
        for i in range(self.n):
            H = H + torch.kron(self.A[i], self.S[i])
        return H

    def forward(self, input):
        self.weight = torch.sum(self.kronecker_product1(self.A, self.S), dim=0)
        #     self.weight = self.kronecker_product2() <- SLOWER
        input = input.type(dtype=self.weight.type())

        # print(f"A size: {self.A.size()}, S size: {self.S.size()}")
        # print(f"kronecker size: {self.kronecker_product1(self.A, self.S).size()}")
        # print(f"input size: {input.size()}")
        # print(f"weight size: {self.weight.size()}")
        # print(f"bias size: {self.bias.size()}")
        out = F.linear(input, weight=self.weight, bias=self.bias)
        # print(f"PHMLinear in shape = {input.size()}, out shape = {out.size()}, weight shape = {self.weight.size()}")
        return out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.A, a=math.sqrt(5))
        init.kaiming_uniform_(self.S, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.placeholder)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)
