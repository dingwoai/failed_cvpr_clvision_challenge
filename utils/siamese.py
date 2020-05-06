#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from .mnasnet_mod import *


class SiameseNetwork(nn.Module):
    def __init__(self, n_class):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = mnasnet1_0(pretrained=True)
        self.cnn1.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(1280, n_class))
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, 128),
            )

    def forward_once(self, x):
        output = self.cnn1.layers(x)
        output = output.mean([2, 3])
        # output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward_orig(self, x):
        output = self.cnn1(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_orig(input1)
        output4 = self.forward_orig(input2)
        return output1, output2, output3, output4


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class ContrastiveAndCELoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveAndCELoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, output3, output4, label_s, label_o1, label_o2):
        '''
        output1, output2: learned representation of input1 & 2
        output3, output4: original classification result of input 1 & 2
        '''
        euclidean_distance = nn.functional.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label_s) * torch.pow(euclidean_distance, 2) +
                                      (label_s) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        loss = loss_contrastive + nn.functional.cross_entropy(output3, label_o1) + nn.functional.cross_entropy(output4, label_o2)
        return loss