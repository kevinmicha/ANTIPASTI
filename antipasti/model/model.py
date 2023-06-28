# -*- coding: utf-8 -*-

r"""This module contains the model class. 

:Authors:   Kevin Michalewicz <k.michalewicz22@imperial.ac.uk>

"""
import numpy as np
import torch
from torch.nn import Linear, ReLU, Conv2d, MaxPool2d, Module, Dropout, Parameter

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

    """
    def __init__(
            self,
            n_filters=2,
            filter_size=4,
            pooling_size=1,
            input_shape=281,
            l1_lambda=0.002,
    ):
        super(ANTIPASTI, self).__init__()
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.pooling_size = pooling_size
        self.input_shape = input_shape
        self.fully_connected_input = n_filters * ((input_shape-filter_size+1)//pooling_size) ** 2
        self.conv1 = Conv2d(1, n_filters, filter_size)
        #self.conv1_bn = torch.nn.BatchNorm2d(n_filters)
        #torch.nn.init.normal_(self.conv1.weight, mean=0.0, std=1/input_shape)
        #torch.nn.init.constant_(self.conv1.bias, 0)
        self.pool = MaxPool2d(pooling_size, pooling_size)
        self.dropit = Dropout(p=0.05)
        self.relu = ReLU()
        self.fc1 = Linear(self.fully_connected_input, 1, bias=False)
        #self.fc1_bn = torch.nn.BatchNorm1d(self.fully_connected_input)
        #torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=1/np.sqrt(self.fully_connected_input))
        #self.fc2 = Linear(self.fully_connected_input, 1, bias=False)
        self.l1_lambda = l1_lambda

    def forward(self, input):
        r"""Model's forward pass.

        Returns
        -------
        output: torch.Tensor
            Predicted binding affinity.
        inter_filter: torch.Tensor
            Filters before the fully-connected layer.
            
        """
        #if torch.numel(torch.nonzero(input[0,0,-80:,-80:])) == 0:
        #    x = self.conv2(input) + torch.transpose(self.conv2(input), 2, 3)
        #else:
        #    x = self.conv1(input) + torch.transpose(self.conv1(input), 2, 3)
        x = self.conv1(input) + torch.transpose(self.conv1(input), 2, 3)
        #x = self.conv1_bn(x)
        x = self.relu(x)
        x = self.pool(x)
        inter = x 
        x = x.view(x.size(0), -1)
        x = self.dropit(x)
        #if torch.numel(torch.nonzero(input[0,0,-80:,-80:])) == 0:
        #    x = self.fc2(x)
        #    print('nano')
        #else:
        #    x = self.fc1(x)
        #    print('paired')
        x = self.fc1(x)

        return x.float(), inter

    def l1_regularization_loss(self):
        l1_loss = torch.tensor(0.0)
        for param in self.parameters():
            l1_loss += torch.norm(param, p=1)
        return self.l1_lambda * l1_loss