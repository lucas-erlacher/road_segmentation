import torch
import torch.nn as nn


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, model):
        super(UnetSkipConnectionBlock, self).__init__()

        self.model = model

    def forward(self, x):
        return torch.cat([x, self.model(x)], 1)
