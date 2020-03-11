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

    def __init__(
        self,
        embed_size: int = 50,
        m_word: int = 21,
        kernel_size: int = 5,
        filter: int = None,
    ):
        super(CNN, self).__init__()
        self.conv_layer = nn.Conv1d(
            in_channels=embed_size, out_channels=filter, kernel_size=kernel_size
        )
        self.maxpool = nn.MaxPool1d(kernel_size=m_word - kernel_size + 1)

    def forward(self, x_reshaped: torch.Tensor):
        """
        @param x_reshaped (Tensor): Tensor of char-level embedding with shape (max_sentence_length,
                                    batch_size, e_char, m_word), where e_char = embed_size of char,
                                    m_word = max_word_length.
        @return x_conv_out (Tensor): Tensor of word-level embedding with shape (max_sentence_length,
                                    batch_size)
        """
        x_conv = self.conv_layer(x_reshaped)
        x_conv_out = self.maxpool(F.relu(x_conv))

        return torch.squeeze(x_conv_out, -1)


### END YOUR CODE
