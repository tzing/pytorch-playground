"""
Original code by Ke Ding
https://github.com/DingKe/pytorch_workplace/blob/master/focalloss/loss.py

Porting to torch 0.4.0 by tzing

MIT License Copyright (c) 2017-2018 Ke Ding, tzing
"""
import torch
import torch.nn as nn


def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.zeros(*size)
    index = index.view(*view)
    ones = 1.

    if index.device != mask.device:
        mask = mask.to(index.device)

    return mask.scatter_(1, index, ones)


class FocalLoss(nn.Module):

    def __init__(self, alpha=.25, gamma=2.):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input_, target):
        p = input_.exp()
        y = one_hot(target, input_.size(-1))

        loss = -1 * y * input_  # cross entropy
        loss = loss * self.alpha * (1 - p) ** self.gamma  # focal loss

        return loss.sum()
