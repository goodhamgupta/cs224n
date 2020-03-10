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
    def __init__(self, embed_size, m_word,kernel_size: int=5, filter: int = None):
        super(CNN, self).__init__()
        self.conv_layer = nn.Conv1d(embed_size, m_word, kernel_size=kernel_size)
        self.maxpool = nn.MaxPool1d(kernel_size=m_word - k + 1)

    def forward(self, x_reshaped: torch.Tensor):
        X_word_emb_list = []
        # divide input into sentence_length batchs
        for X_padded in x_reshaped:
            X_emb = self.char_embedding(X_padded)
            X_reshaped = torch.transpose(X_emb, dim0=-1, dim1=-2)
            # conv1d can only take 3-dim mat as input
            # so it needs to concat/stack all the embeddings of word
            # after going through the network
            X_conv_out = self.convNN(X_reshaped)
            X_highway = self.highway(X_conv_out)
            X_word_emb = self.dropout(X_highway)
            X_word_emb_list.append(X_word_emb)

        X_word_emb = torch.stack(X_word_emb_list)
        return X_word_emb

### END YOUR CODE

