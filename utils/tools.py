import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import torch.nn as nn

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 2 else args.learning_rate * (0.9 ** ((epoch - 2) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(flow, true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    time1=np.linspace(1,240-1/8, 1920)
    time2=np.linspace(1,240, 240)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(time1, flow, label='FLow', linewidth=2)
    # plt.xlabel('Time (sec)')
    plt.ylabel('Voltage (uV)')
    plt.subplot(2,1,2)
    plt.plot(time2, true, label='GroundTruth', linewidth=2)
    y_labels = ['normal', 'hypopnea', 'apnea']
    if preds is not None:
        plt.plot(time2, preds, label='Prediction', linewidth=2)
    # plt.ylim(80, 100)
    plt.legend()
    plt.xlabel('Time (sec)')
    plt.yticks([0, 1, 2], y_labels)
    plt.savefig(name, bbox_inches='tight')

def visual2(flow, spo2, true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    time1=np.linspace(1,480, 480)
    time2=np.linspace(1,60, 60)
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(time1, flow, label='Respiration', linewidth=2)
    # plt.xlabel('Time (sec)')
    plt.ylabel('Voltage (uV)')
    plt.subplot(3,1,2)
    plt.plot(time1, spo2, label='Respiration', linewidth=2)
    # plt.xlabel('Time (sec)')
    plt.ylabel('SpO2 (%)')
    plt.subplot(3,1,3)
    plt.plot(time2, true, label='GroundTruth', linewidth=2)
    y_labels = ['normal', 'hypopnea', 'apnea']
    if preds is not None:
        plt.plot(time2, preds, label='Prediction', linewidth=2)
    # plt.ylim(80, 100)
    plt.legend()
    plt.xlabel('Time (sec)')
    plt.yticks([0, 1, 2], y_labels)
    plt.savefig(name, bbox_inches='tight')

def visual3(abdo, thor, spo2, true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    time1=np.linspace(1,240-1/8, 1920)
    time2=np.linspace(1,240, 240)
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(time1, abdo, label='Abdo', linewidth=2)
    plt.plot(time1, thor, label='Flow', linewidth=2)
    plt.ylabel('Voltage (uV)')
    plt.subplot(3,1,2)
    plt.plot(time1, spo2, label='SpO2', linewidth=2)
    plt.ylabel('SpO2 (%)')
    plt.subplot(3,1,3)
    plt.plot(time2, true, label='GroundTruth', linewidth=2)
    y_labels = ['normal', 'hypopnea', 'apnea']
    if preds is not None:
        plt.plot(time2, preds, label='Prediction', linewidth=2)
    # plt.ylim(80, 100)
    plt.legend()
    plt.xlabel('Time (sec)')
    plt.yticks([0, 1, 2], y_labels)
    plt.savefig(name, bbox_inches='tight')

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        # front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        # end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        # x = torch.cat([front, x, end], dim=1)
        # x = self.avg(x.permute(0, 2, 1))
        # x = x.permute(0, 2, 1)
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=2)
        x = self.avg(x)
        return x