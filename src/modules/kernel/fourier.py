#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Implement the positive orthogonal random features from the paper
"Rethinking Attention with Performers" https://arxiv.org/pdf/2009.14794.pdf
and the traditional random Fourier features that approximate the RBF kernel.
"""

from math import sqrt, log
from typing import Optional
from einops import repeat

import torch

from .base import Kernel


def orthogonal_random_matrix_(
        num_rows: int,
        num_columns: int,
        scaling: float = 0,
        device: Optional[torch.device] = None
):
    num_full_blocks = int(num_rows / num_columns)
    block_list = []

    for _ in range(num_full_blocks):
        q = orthogonal_matrix_chunk(num_columns, device)
        block_list.append(q)


    remaining_rows = num_rows - (num_full_blocks * num_columns)
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(num_columns, device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((num_rows, num_columns), device=device)\
            .norm(dim=1)
    elif scaling == 1:
        multiplier = sqrt((float(num_columns))) * torch.ones((num_rows,), device=device)
    else:
        raise ValueError(f"Invalid scaling {scaling}")

    return torch.diag(multiplier) @ final_matrix


def orthogonal_matrix_chunk(
        cols: int,
        device: torch.device = None
) -> torch.Tensor:
    unstructured_block = torch.randn((cols, cols), device=device)
    q, _ = torch.qr(unstructured_block.cpu(), some=True)
    q = q.to(device)
    return q.t()


class RandomFourierFeatures(Kernel):
    """Random Fourier Features for the RBF kernel according to [1].
    [1]: "Weighted Sums of Random Kitchen Sinks: Replacing minimization with
         randomization in learning" by A. Rahimi and Benjamin Recht.
    Arguments
    ---------
        hidden_size: int, The input query dimensions in order to sample
                          the noise matrix
        softmax_temp: float, The temerature for the Gaussian kernel
                      approximation exp(-t * |x-y|^2)
                      (default: 1/sqrt(query_dimensions))
        orthogonal: bool, When True the random matrix is initialized for
                    orthogonal random features to reduce the approximation
                    variance (default: False)
    """
    def __init__(
            self,
            head_size: int,
            kernel_size: Optional[int] = None,
            softmax_temp: Optional[float] = None,
            orthogonal: bool = False
    ):
        super(RandomFourierFeatures, self).__init__(head_size)
        assert kernel_size % 2 == 0, "kernel size not divisible by 2"
        self.kernel_size = kernel_size
        self.orthogonal = orthogonal
        self.softmax_temp = (
            1/sqrt(head_size) if softmax_temp is None
            else softmax_temp
        )

        # Make a buffer for storing the sampled omega
        self.register_buffer(
            "omega",
            torch.zeros(head_size, self.kernel_size//2)
        )

    def new_kernel(self):
        if self.orthogonal:
            orthogonal_random_matrix_(self.omega)
        else:
            self.omega.normal_()

    def forward(
            self,
            x: torch.Tensor,
            **kwargs
    ) -> torch.Tensor:
        x = x * sqrt(self.softmax_temp)
        u = x.unsqueeze(-2).matmul(self.omega).squeeze(-2)
        phi = torch.cat([torch.cos(u), torch.sin(u)], dim=-1)
        return phi * sqrt(2/self.kernel_size)


class SmoothedRandomFourierFeatures(RandomFourierFeatures):
    """Simply add a constant value to the dot product in order to avoid
    possible numerical instabilities when the feature map is slightly
    negative.
    Implements K(x, y) = exp(-|x-y|^2) + s.
    Arguments
    ---------
        query_dimensions: int, The input query dimensions in order to sample
                          the noise matrix
        n_dims: int, The size of the feature map (should be divisible by 2)
                (default: query_dimensions)
        softmax_temp: float, The temerature for the Gaussian kernel
                      approximation exp(-t * |x-y|^2)
                      (default: 1/sqrt(query_dimensions))
        orthogonal: bool, When True the random matrix is initialized for
                    orthogonal random features to reduce the approximation
                    variance (default: False)
        smoothing: float, The smoothing parameter to add to the dot product.
    """
    def __init__(self, query_dimensions, n_dims=None, softmax_temp=None,
                 orthogonal=False, smoothing=1.0):
        super(SmoothedRandomFourierFeatures, self).__init__(
            query_dimensions,
            n_dims=query_dimensions-1 if n_dims is None else n_dims-1,
            softmax_temp=softmax_temp,
            orthogonal=orthogonal,
        )
        self.smoothing = smoothing

    def forward(self, x):
        y = super().forward(x)
        smoothing = torch.full(
            y.shape[:-1] + (1,),
            self.smoothing,
            dtype=y.dtype,
            device=y.device
        )
        return torch.cat([y, smoothing], dim=-1)


class SoftmaxKernel(Kernel):
    """Positive orthogonal random features that approximate the softmax kernel.
    Basically implementation of Lemma 1 from "Rethinking Attention with
    Performers".
    Arguments
    ---------
        head_size: int, The input query dimensions in order to sample
                          the noise matrix
        kernel_size: int, The size of the feature map (should be divisible by 2)
                (default: query_dimensions)
        softmax_temp: float, The temerature for the softmax approximation
                     (default: 1/sqrt(query_dimensions))
        orthogonal: bool, If set to true then the random matrix should be
                    orthogonal which results in lower approximation variance
                    (default: True)
        stabilize: bool, If set to True subtract the max norm from the
                   exponentials to make sure that there are no infinities. It
                   is equivalent to a robust implementation of softmax where
                   the max is subtracted before the exponentiation.
                   (default: False)
    """
    def __init__(
            self,
            head_size: int,
            kernel_size: Optional[int] = None,
            ortho_scaling: Optional[float] = 0,
            causal: bool = False,
            orthogonal: bool = True,
            eps: float = 1e-4
    ):
        super(SoftmaxKernel, self).__init__(head_size)
        kernel_size = int(self.head_size * log(self.head_size)) if kernel_size is None else kernel_size
        self.kernel_size = kernel_size
        self.ortho_scaling = ortho_scaling
        self.causal = causal
        self.orthogonal = orthogonal
        self.eps = eps

        self.register_buffer(
            "omegas",
            self.new_kernel()
        )

        if self.causal:
            raise NotImplementedError("linear causal attention not yet implemented")

    def new_kernel(
            self,
            device: Optional[torch.device] = "cpu"
    ):
        return orthogonal_random_matrix_(
            self.kernel_size,
            self.head_size,
            scaling=self.ortho_scaling,
            device=device
        )



    def forward(
            self,
            x: torch.Tensor,
            is_query: bool,
            normalize_data: bool = True,
    ) -> torch.Tensor:
        b, h, *_ = x.shape

        if normalize_data:
            x_norm = 1. / (x.shape[-1] ** 0.25)
        else:
            x_norm = 1.

        ratio = 1. / (self.omegas.shape[0] ** 0.5)

        projection_matrix = repeat(self.omegas, 'j d -> b h j d', b=b, h=h)

        data_dash = torch.einsum('...id,...jd->...ij', (x_norm * x), projection_matrix)

        diag_x = torch.sum(x ** 2, dim=-1)
        diag_x = ((diag_x / 2.0) * (x_norm ** 2)).unsqueeze(dim=-1)

        if is_query:
            data_dash = ratio * (
                torch.exp(data_dash - diag_x - torch.max(data_dash, dim=-1, keepdim=True).values) + self.eps
            )
        else:
            data_dash = ratio * (
                    torch.exp(data_dash - diag_x - torch.max(data_dash)) + self.eps
            )

        return data_dash


class GeneralizedRandomFeatures(RandomFourierFeatures):
    """Implements the generalized random Fourier features from Performers.
    It computes φ(χ) = [f(ω_1 χ), f(ω_2 χ), ..., f(ω_n χ)] where f(.) is the
    passed in `kernel_fn`.
    Arguments
    ---------
        query_dimensions: int, The input query dimensions in order to sample
                          the noise matrix
        n_dims: int, The size of the feature map (default: query_dimensions)
        softmax_temp: float, A normalizer for the dot products that is
                     multiplied to the input features before the feature map
                     application (default: 1.0)
        orthogonal: bool, If set to true then the random matrix should be
                    orthogonal which results in lower approximation variance
                    (default: True)
        kernel_fn: callable, defines the f used for the feature map.
                   (default: relu)
    """
    def __init__(self, query_dimensions, n_dims=None, softmax_temp=1.0,
                 orthogonal=True, kernel_fn=torch.relu):
        super(GeneralizedRandomFeatures, self).__init__(
            query_dimensions,
            n_dims=2*query_dimensions if n_dims is None else 2*n_dims,
            softmax_temp=softmax_temp,
            orthogonal=orthogonal
        )
        self.kernel_fn = kernel_fn

    def forward(self, x):
        if self.softmax_temp != 1.0:
            x = x * sqrt(self.softmax_temp)
        u = x.unsqueeze(-2).matmul(self.omega).squeeze(-2)
        return self.kernel_fn(u)