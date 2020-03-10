#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


import random
### YOUR CODE HERE for part 1i

class CNN(nn.Module):
    """
    Implementation of CNN
    """
    def __init__(self, in_channels, out_channels, kernel_size: int=5):
        super(CNN, self).__init__()
        self.conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x_reshaped: torch.Tensor):
        output = self.conv_layer(x_reshaped)
        return output
        pass
### END YOUR CODE

