import numpy as np
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def ACC1(pred, true):
    # 将数组展平成一维数组，方便比较
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    equal_count = np.sum(pred_flat == true_flat)
    acc = equal_count / pred.size

    # 统计值为0、1和2的个数
    count_0 = np.sum(true_flat == 0)
    count_1 = np.sum(true_flat == 1)
    count_2 = np.sum(true_flat == 2)

    count_true_0_pred_0 = np.sum((true_flat == 0) & (pred_flat == 0))
    count_true_0_pred_1 = np.sum((true_flat == 0) & (pred_flat == 1))
    count_true_0_pred_2 = np.sum((true_flat == 0) & (pred_flat == 2))

    count_true_1_pred_0 = np.sum((true_flat == 1) & (pred_flat == 0))
    count_true_1_pred_1 = np.sum((true_flat == 1) & (pred_flat == 1))
    count_true_1_pred_2 = np.sum((true_flat == 1) & (pred_flat == 2))

    count_true_2_pred_0 = np.sum((true_flat == 2) & (pred_flat == 0))
    count_true_2_pred_1 = np.sum((true_flat == 2) & (pred_flat == 1))
    count_true_2_pred_2 = np.sum((true_flat == 2) & (pred_flat == 2))

    if count_0 == 0:
        acc_00 = '类别0的真实值个数为0'
        acc_01 = '类别0的真实值个数为0'
        acc_02 = '类别0的真实值个数为0'
    else:
        acc_00 = count_true_0_pred_0/count_0
        acc_01 = count_true_0_pred_1/count_0
        acc_02 = count_true_0_pred_2/count_0
    if count_1 == 0:
        acc_10 = '类别1的真实值个数为0'
        acc_11 = '类别1的真实值个数为0'
        acc_12 = '类别1的真实值个数为0'
    else:
        acc_10 = count_true_1_pred_0/count_1
        acc_11 = count_true_1_pred_1/count_1
        acc_12 = count_true_1_pred_2/count_1
    if count_2 == 0:
        acc_20 = '类别2的真实值个数为0'
        acc_21 = '类别2的真实值个数为0'
        acc_22 = '类别2的真实值个数为0'
    else:
        acc_20 = count_true_2_pred_0/count_2
        acc_21 = count_true_2_pred_1/count_2
        acc_22 = count_true_2_pred_2/count_2

    return acc, acc_00,acc_01,acc_02,acc_10,acc_11,acc_12,acc_20,acc_21,acc_22


def metric1(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    acc, acc_00,acc_01,acc_02,acc_10,acc_11,acc_12,acc_20,acc_21,acc_22 = ACC1(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr, acc, acc_00,acc_01,acc_02,acc_10,acc_11,acc_12,acc_20,acc_21,acc_22

def ACC2(pred, true):
    # 将数组展平成一维数组，方便比较
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    equal_count = np.sum(pred_flat == true_flat)
    acc = equal_count / pred.size

    # 统计值为0、1和2的个数
    count_0 = np.sum(true_flat == 0)
    count_1 = np.sum(true_flat == 1)

    count_true_0_pred_0 = np.sum((true_flat == 0) & (pred_flat == 0))
    count_true_0_pred_1 = np.sum((true_flat == 0) & (pred_flat == 1))

    count_true_1_pred_0 = np.sum((true_flat == 1) & (pred_flat == 0))
    count_true_1_pred_1 = np.sum((true_flat == 1) & (pred_flat == 1))

    if count_0 == 0:
        acc_00 = '类别0的真实值个数为0'
        acc_01 = '类别0的真实值个数为0'
    else:
        acc_00 = count_true_0_pred_0/count_0
        acc_01 = count_true_0_pred_1/count_0
    if count_1 == 0:
        acc_10 = '类别1的真实值个数为0'
        acc_11 = '类别1的真实值个数为0'
    else:
        acc_10 = count_true_1_pred_0/count_1
        acc_11 = count_true_1_pred_1/count_1

    return acc, acc_00, acc_01, acc_10, acc_11, count_0, count_1, count_true_0_pred_0, count_true_0_pred_1, count_true_1_pred_0, count_true_1_pred_1


def metric2(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    acc, acc_00, acc_01, acc_10, acc_11, count_0, count_1, count_true_0_pred_0, count_true_0_pred_1, count_true_1_pred_0, count_true_1_pred_1 = ACC2(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr, acc, acc_00, acc_01, acc_10, acc_11, count_0, count_1, count_true_0_pred_0, count_true_0_pred_1, count_true_1_pred_0, count_true_1_pred_1

class PearsonLoss(_Loss):
    def __init__(self, ):
        super(PearsonLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    def forward(self, pred, true): 
        pearson = self.cos(pred - torch.mean(pred, dim=2, keepdim=True), true - torch.mean(true, dim=2, keepdim=True)) 
        return pearson.mean()
