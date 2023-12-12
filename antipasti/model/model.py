# -*- coding: utf-8 -*-

r"""This module contains the model class. 

:Authors:   Kevin Michalewicz <k.michalewicz22@imperial.ac.uk>

"""
import numpy as np
import torch
from torch.nn import Linear, ReLU, Conv2d, MaxPool2d, Module

class ANTIPASTI(Module):
    r"""Predicting the binding affinity of an antibody from its normal mode correlation map.

    Parameters
    ----------
    n_filters: int
        Number of filters in the convolutional layer.
    filter_size: int
        Size of filters in the convolutional layer.
    pooling_size: int
        Size of the max pooling operation.
    input_shape: int
        Shape of the normal mode correlation maps.
    l1_lambda: float
        Weight of L1 regularisation.
    mode: str
        To use the full model, provide ``full``. Otherwise, ANTIPASTI corresponds to a linear map.

    """
    def __init__(
            self,
            n_filters=2,
            filter_size=4,
            pooling_size=1,
            input_shape=281,
            l1_lambda=0.002,
            mode='full',
    ):
        super(ANTIPASTI, self).__init__()
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.pooling_size = pooling_size
        self.input_shape = input_shape
        self.mode = mode
        if self.mode == 'full':
            self.fully_connected_input = n_filters * ((input_shape-filter_size+1)//pooling_size) ** 2
            self.conv1 = Conv2d(1, n_filters, filter_size)
            self.pool = MaxPool2d(pooling_size, pooling_size)
            self.relu = ReLU()
        else:
            self.fully_connected_input = self.input_shape ** 2
        self.fc1 = Linear(self.fully_connected_input, 1, bias=False)
        self.l1_lambda = l1_lambda

    def forward(self, x):
        r"""Model's forward pass.

        Returns
        -------
        output: torch.Tensor
            Predicted binding affinity.
        inter_filter: torch.Tensor
            Filters before the fully-connected layer.
            
        """
        inter = x
        if self.mode == 'full':
            x = self.conv1(x) + torch.transpose(self.conv1(x), 2, 3)
            x = self.relu(x)
            inter = x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x.float(), inter

    def l1_regularization_loss(self):
        l1_loss = torch.tensor(0.0)
        for param in self.parameters():
            l1_loss += torch.norm(param, p=1)
        return self.l1_lambda * l1_loss