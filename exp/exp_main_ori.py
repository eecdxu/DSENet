from models import PatchTSTv0,PatchTSTv0_ori,PatchTSTv0_tnt,PatchTSTv0_MGA,PatchTSTv0_subpatch
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop, moving_avg, visual2
from utils.metrics import metric1, metric2, PearsonLoss

import numpy as np
import torch, scipy
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os, h5py
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.scaler = StandardScaler()
        self.moving_avg = moving_avg(4, stride=4)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTSTv0_ori,
            'PatchTST_subpatch': PatchTSTv0_subpatch,
            'PatchTST_tnt': PatchTSTv0_tnt,
            'PatchTST_MGA': PatchTSTv0_MGA,
            'PatchTST_swin': PatchTSTv0,
            'PatchTST_swin_hiera': PatchTSTv0,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion1 = nn.CrossEntropyLoss()
        criterion2 = PearsonLoss()
        return criterion1, criterion2

    def vali(self, vali_data, vali_loader, criterion1, criterion2):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                # batch_x = batch_x.unsqueeze(1)
                # batch_x  = self.moving_avg(batch_x)
                # batch_x = batch_x.squeeze(1)
                self.scaler.fit(batch_x.T)
                batch_x = self.scaler.transform(batch_x.T).T
                batch_x = torch.from_numpy(batch_x).float().to(self.device)
                batch_x = batch_x.float().to(self.device)
                batch_x = batch_x.unsqueeze(1)

                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs, backbone, atten1, atten2, atten3 = self.model(batch_x)
                        
                # f_dim = -1 if self.args.features == 'MS' else 0
                # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                # loss = criterion(pred, true)
                loss = criterion1(pred, true.long())
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        # vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion1, criterion2 = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                # batch_x = batch_x.unsqueeze(1)
                # batch_x  = self.moving_avg(batch_x)
                # batch_x = batch_x.squeeze(1)
                self.scaler.fit(batch_x.T)
                batch_x = self.scaler.transform(batch_x.T).T
                batch_x = torch.from_numpy(batch_x).float().to(self.device)
                batch_x = batch_x.unsqueeze(1)

                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs, backbone, atten1, atten2, atten3 = self.model(batch_x)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion1(outputs, batch_y) - criterion2(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs, backbone, atten1, atten2, atten3 = self.model(batch_x)
                    # print(outputs.shape,batch_y.shape)
                    # f_dim = -1 if self.args.features == 'MS' else 0
                    # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    # loss1=criterion1(outputs, batch_y)
                    # loss2=criterion2(outputs, batch_y)
                    # loss = loss1-loss2
                    loss = criterion1(outputs, batch_y.long())
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # vali_loss = self.vali(vali_data, vali_loader, criterion1, criterion2)
            test_loss = self.vali(test_data, test_loader, criterion1, criterion2)

            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            # early_stopping(vali_loss, self.model, path)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, test_loss))
            early_stopping(test_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location='cuda'))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda'))
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth'), map_location='cuda'))

        preds = []
        trues = []
        backbones = []
        atten1s, atten2s, atten3s = [], [], []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                # batch_x = batch_x.unsqueeze(1)
                # batch_x  = self.moving_avg(batch_x)
                # batch_x = batch_x.squeeze(1)
                self.scaler.fit(batch_x.T)
                batch_x = self.scaler.transform(batch_x.T).T
                batch_x = torch.from_numpy(batch_x).float().to(self.device)
                batch_x = batch_x.unsqueeze(1)

                batch_y = batch_y.float().to(self.device)
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs, backbone, atten1, atten2, atten3 = self.model(batch_x)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                backbone = backbone.detach().cpu().numpy()
                atten1, atten2, atten3 = atten1.detach().cpu().numpy(), atten2.detach().cpu().numpy(), atten3.detach().cpu().numpy()

                pred = np.argmax(outputs, axis=1)
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.extend(pred)
                trues.extend(true)
                backbones.extend(backbone.reshape(backbone.shape[0], -1))
                atten1s.extend(atten1)
                atten2s.extend(atten2)
                atten3s.extend(atten3)

                # if i % 1 == 0:
                #     input1 = batch_x.detach().cpu().numpy()
                #     flow = input1[0, 0, :].reshape(1, -1).squeeze()
                #     input2 = batch_xxx_ori
                #     spo2 = input2[0, :].reshape(1, -1).squeeze()
                #     gt = true[0, 0, :].reshape(1, -1).squeeze()
                #     pd = pred[0, 0, :].reshape(1, -1).squeeze()
                #     visual2(flow, spo2, gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # target_path = '/home/wsco/xcd/Pytorch/PatchTST_Flow_SpO2/sleep_events'
        # output_file_path = os.path.join(target_path, 'Wake_event_test_pred_Flow_SpO2.h5')
        # with h5py.File(output_file_path, 'w') as hf:
        #     for i, element in enumerate(preds):
        #         dataset_name = f'wake_event_{i}'
        #         hf.create_dataset(dataset_name, data=element)

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.vstack(preds)
        trues = np.vstack(trues)
        backbones = np.vstack(backbones)

        preds = preds.reshape(-1, preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-1])
        backbones = backbones.reshape(-1, backbones.shape[-1])
        # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        result_path = self.args.result_path
        mae, mse, rmse, mape, mspe, rse, corr, acc, acc_00,acc_01,acc_10,acc_11, c0, c1, c00, c01, c10, c11 = metric2(preds, trues)
        print('mse:{}, mae:{}, rse:{}, acc:{}, acc_00:{},acc_01:{},acc_10:{},acc_11:{}, c0:{}, c1:{}, c00:{}, c01:{}, c10:{}, c11:{}'
              .format(mse, mae, rse, acc, acc_00,acc_01,acc_10,acc_11, c0, c1, c00, c01, c10, c11))
        f = open(result_path, 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, acc:{}, acc_00:{},acc_01:{},acc_10:{},acc_11:{}, c0:{}, c1:{}, c00:{}, c01:{}, c10:{}, c11:{}'
                .format(mse, mae, rse, acc, acc_00,acc_01,acc_10,acc_11, c0, c1, c00, c01, c10, c11))
        f.write('\n')
        f.write('\n')
        f.close()

        # 将backbones保存为h5文件，只创建一个数据集
        target_path = '/media/wsco/XuCd/Pytorch/MESA_Cls/backbones_PatchTST'
        output_file_path = os.path.join(target_path, 'backbones.h5')
        with h5py.File(output_file_path, 'w') as hf:
            hf.create_dataset('backbones', data=backbones)
        output_file_path = os.path.join(target_path, 'preds.h5')
        with h5py.File(output_file_path, 'w') as hf:
            hf.create_dataset('preds', data=preds)

        # 将atten1, atten2, atten3保存成h5文件
        target_path = '/media/wsco/XuCd/Pytorch/MESA_Cls/attentions_PatchTST'
        output_file_path = os.path.join(target_path, 'atten1.h5')
        with h5py.File(output_file_path, 'w') as hf:
            hf.create_dataset('atten1', data=atten1s)
        output_file_path = os.path.join(target_path, 'atten2.h5')
        with h5py.File(output_file_path, 'w') as hf:
            hf.create_dataset('atten2', data=atten2s)
        output_file_path = os.path.join(target_path, 'atten3.h5')
        with h5py.File(output_file_path, 'w') as hf:
            hf.create_dataset('atten3', data=atten3s)

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(pred_loader):
                # length = min(x.size(2)//10,y.size(2))
                # x=x.squeeze(1)[:,0:length*10]
                # y=y.squeeze(1)[:,0:length]
                # batch_x = x.squeeze(1).unfold(dimension=-1, size=2400, step=2400)
                # batch_x = batch_x[:, :, 0::2]
                # batch_y = y.squeeze(1).unfold(dimension=-1, size=240, step=240)
                self.scaler.fit(batch_x.T)
                batch_x = self.scaler.transform(batch_x.T).T
                batch_x = torch.from_numpy(batch_x).float().to(self.device)
                # batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
