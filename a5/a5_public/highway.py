#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


import random

### YOUR CODE HERE for part 1h


class Highway(nn.Module):
    # y = H(x,WH)· T(x,WT) + x · (1 − T(x,WT))

    def __init__(self, size: int, activation_fn=F.relu):
        """
        Initialize a highway network.
        """
        super(Highway, self).__init__()
        self.nonlinear = nn.Linear(size, size)
        self.linear = nn.Linear(size, size)
        self.gate = nn.Linear(size, size)
        self.activation_fn = activation_fn

    def forward(self, x_conv: torch.Tensor) -> torch.Tensor:
        gate_output = torch.sigmoid(self.gate(x_conv))
        nonlinear = self.activation_fn(self.nonlinear(x_conv))
        linear = self.linear(x_conv)
        x_highway = (gate_output * nonlinear) + (1 - gate_output) * linear
        return x_highway


### END YOUR CODE
