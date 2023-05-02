# -*- coding: utf-8 -*-

r"""This module contains the model class. 

:Authors:   Kevin Michalewicz <k.michalewicz22@imperial.ac.uk>

"""

import torch
from torch.nn import Linear, ReLU, Conv2d, MaxPool2d, Module

class NormalModeAnalysisCNN(Module):
    """
    NMA-CNN class. It predicts the binding affinity of an antibody from its normal mode correlation map.
    
    """
    def __init__(
            self,
            n_filters=2,
            filter_size=5,
            pooling_size=1,
            input_shape=215,
    ):
        """
        :param n_filters: number of filters in the convolutional layer
        :type n_filters: int
        :param filter_size: size of filters in the convolutional layer
        :type filter_size: int
        :param pooling_size: size of the max pooling operation
        :type pooling_size: int
        :param input_shape: shape of the normal mode correlation maps
        :type input_shape: int

        """
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.pooling_size = pooling_size
        self.input_shape = input_shape
        self.fully_connected_input = n_filters * ((input_shape-filter_size+1)//pooling_size) ** 2
        self.conv1 = Conv2d(1, n_filters, filter_size)
        self.pool = MaxPool2d(pooling_size, pooling_size)
        self.relu = ReLU()
        self.fc1 = Linear(self.fully_connected_input, 1, bias=False)

    def forward(self, x):
        """
        Model's forward pass.

        :return: binding affinity and filters before the fully-connected layer
            
        """
        x = self.conv1(x) + torch.transpose(self.conv1(x), 2, 3)
        x = self.pool(x)
        inter = x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x.float(), inter
