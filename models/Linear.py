import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.Linear1 = nn.Linear(self.seq_len, self.pred_len * 10)
        self.act = nn.GELU()
        self.Linear2 = nn.Linear(self.pred_len * 10, 2*self.pred_len)
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = self.Linear1(x)
        x = self.act(x)
        x = self.Linear2(x)
        return x.squeeze() # [Batch, Output length, Channel]