import torch
import torch.nn as nn
from layers.PatchTST_layers import get_activation_fn

class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv1d, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.pointwise_conv(self.conv1(x))
        # x = self.conv1(x)
        return x


class Agg_0(nn.Module):
    def __init__(self, seg_dim, act='gelu'):
        super().__init__()
        self.conv = nn.Conv1d(3 * seg_dim, seg_dim, 3, 1, 1)
        # self.conv = nn.Conv1d(seg_dim, seg_dim, 1, 1)
        self.norm1 = nn.LayerNorm(seg_dim)
        self.norm2 = nn.BatchNorm1d(seg_dim)
        self.act = get_activation_fn(act)

    def forward(self, x):
        x = self.conv(x)
        b, c, p = x.shape
        # x = self.act(self.norm(x.reshape(b, c, -1).permute(0, 2, 1)))
        x = self.act(self.norm2(x.reshape(b, c, -1)))
        x = x.permute(0,2,1)

        return x


class Aggregator(nn.Module):
    def __init__(self, dim, seg=4, act='gelu'):
        super().__init__()
        self.dim = dim
        self.seg = seg
        self.act = get_activation_fn(act)

        seg_dim = self.dim // self.seg

        self.norm0 = nn.BatchNorm1d(seg_dim)

        self.agg1 = SeparableConv1d(seg_dim, seg_dim, 3, 1, 1)
        self.norm1 = nn.BatchNorm1d(seg_dim)

        self.agg2 = SeparableConv1d(seg_dim, seg_dim, 5, 1, 2)
        self.norm2 = nn.BatchNorm1d(seg_dim)

        self.agg3 = SeparableConv1d(seg_dim, seg_dim, 7, 1, 3)
        self.norm3 = nn.BatchNorm1d(seg_dim)

        self.agg0 = Agg_0(seg_dim, act)


    def forward(self, x, num_head):
        B, N, C = x.shape
        # P = size
        # assert N == P

        x = x.transpose(1, 2).view(B, C, N)
        seg_dim = self.dim // self.seg

        x = x.split([seg_dim]*self.seg, dim=1)

        x_local = x[4].reshape(3, B//3, seg_dim, N).permute(1,0,2,3).reshape(B//3, 3*seg_dim, N)
        # x_local = self.agg0(x_local[:, 2*seg_dim-1:3*seg_dim-1, :])
        x_local = self.agg0(x_local)

        x0 = self.act(self.norm0(x[0]))
        x1 = self.act(self.norm1(self.agg1(x[1])))
        x2 = self.act(self.norm2(self.agg2(x[2])))
        x3 = self.act(self.norm3(self.agg3(x[3])))
        # x0 = x[0]
        # x1 = x[1]
        # x2 = x[2]
        # x3 = x[3]
        # x1 = self.agg1(x[1])
        # x1 = self.act1(self.norm1(x1.permute(0, 2, 1)))
        # x2 = self.agg2(x[2])
        # x2 = self.act2(self.norm2(x2.permute(0, 2, 1)))
        # x3 = self.agg3(x[3])
        # x3 = self.act3(self.norm3(x3.permute(0, 2, 1)))

        x = torch.cat([x0, x1, x2, x3], dim=1)
        # x = x.permute(0, 2, 1)

        C = C // 5 * 4
        # x = x.reshape(3, B//3, num_head, C//num_head, N).permute(0, 1, 4, 2, 3)
        x = x.reshape(3, B//3, C, N).permute(0, 1, 3, 2)
        # x = x.reshape(3, B//3, num_head, C//num_head, N)

        return x, x_local, C