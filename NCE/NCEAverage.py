import torch
from torch import nn
from .alias_multinomial import AliasMethod
import math


class NCEAverage(nn.Module):

    def __init__(self, inputSize, outputSize):
        super(NCEAverage, self).__init__()
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.W1 = nn.Linear(self.inputSize,self.inputSize, bias=True)
        self.W2 = nn.Linear(self.inputSize,self.inputSize, bias=True)

    def forward(self, l, ab):
        batchSize = l.size(0)
        outputSize = self.outputSize
        inputSize = self.inputSize

        # sample
        l_ = self.W1(l)
        out_ab = torch.mm(l_, torch.t(ab).detach()).contiguous()
        ind_ab = torch.tensor(range(batchSize)).view(-1).cuda().detach()
        # sample
        ab_ = self.W2(ab)
        out_l = torch.mm(ab_, torch.t(l).detach()).contiguous()
        ind_l = ind_ab

        return out_l, ind_l, out_ab, ind_ab
