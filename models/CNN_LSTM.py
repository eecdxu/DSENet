import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.pred_len = configs.pred_len

        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv1d(64,32,kernel_size=3, stride = 2, padding=1)
        self.batch1 =nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32,32,kernel_size=3, stride = 2, padding=1)
        self.batch2 =nn.BatchNorm1d(32)
        self.LSTM = nn.LSTM(input_size=32, hidden_size=32*2,
                            num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64*240*2, self.pred_len*2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.pred_len*2, self.pred_len)


    def forward(self, x, xx):
        #in_size1 = x.size(0)  # one batch
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.act(self.batch1(x))
        x = self.conv3(x)
        x = self.act(self.batch2(x))
        x = x.permute(0,2,1)
        x, h = self.LSTM(x)

        xx = self.act(self.conv1(xx))
        xx = self.conv2(xx)
        xx = self.act(self.batch1(xx))
        xx = self.conv3(xx)
        xx = self.act(self.batch2(xx))
        xx = xx.permute(0,2,1)
        xx, h = self.LSTM(xx)

        x = torch.reshape(x,(x.shape[0],x.shape[1]*x.shape[2]))
        xx = torch.reshape(xx,(xx.shape[0],xx.shape[1]*xx.shape[2]))
        x = torch.cat([x, xx], dim=1)
        x = self.act(self.fc1(x))
        output = self.fc2(x)
    
        return output.unsqueeze(1)