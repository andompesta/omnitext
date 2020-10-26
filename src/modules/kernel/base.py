
#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Create the feature map interface and some commonly used feature maps.
All attention implementations that expect a feature map shall receive a factory
function that returns a feature map instance when called with the query
dimensions.
"""

from functools import partial

import torch
from torch.nn import Module


class Kernel(Module):
    """Define the FeatureMap interface."""
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size

    def new_kernel(self):
        """Create a new instance of this feature map. In particular, if it is a
        random feature map sample new parameters."""
        raise NotImplementedError()

    def forward_queries(self, x):
        """Encode the queries `x` using this feature map."""
        return self(x)

    def forward_keys(self, x):
        """Encode the keys `x` using this feature map."""
        return self(x)

    def forward(self, x):
        """Encode x using this feature map. For symmetric feature maps it
        suffices to define this function, but for asymmetric feature maps one
        needs to define the `forward_queries` and `forward_keys` functions."""
        raise NotImplementedError()

    @classmethod
    def factory(cls, *args, **kwargs):
        """Return a function that when called with the query dimensions returns
        an instance of this feature map.
        It is inherited by the subclasses so it is available in all feature
        maps.
        """
        def inner(hidden_size):
            return cls(hidden_size, *args, **kwargs)
        return inner


class ActivationFunctionKernel(Kernel):
    """Define a feature map that is simply an element-wise activation
    function."""
    def __init__(self, head_size, activation_function):
        super().__init__(head_size)
        self.activation_function = activation_function

    def new_kernel(self):
        return

    def forward(self, x):
        return self.activation_function(x)


ELUkernel = ActivationFunctionKernel.factory(
    lambda x: torch.nn.functional.elu(x) + 1
)