import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from utils.timefeatures import time_features
import warnings
import h5py, scipy

warnings.filterwarnings('ignore')
    
class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

# class Dataset_Res_SaO2(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path1='ETTh1.csv', data_path2='ETTh1.csv',StudyNumber=1,
#                  target='OT', scale=True, timeenc=0, freq='h'):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]

#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq

#         self.root_path = root_path
#         self.data_path1 = data_path1
#         self.data_path2 = data_path2
#         self.idx = StudyNumber
#         self.__read_data__()

#     def __read_data__(self):

#         filepath_Res = os.path.join(self.root_path, self.data_path1)
#         with h5py.File(filepath_Res, 'r') as file:
#             dataset_names = list(file.keys())
#             Res = []
#             Res3 = []
#             for idx, dataset_name in enumerate(dataset_names):
#                 if idx == self.idx:
#                     dataset = file[dataset_name]
#                     data = dataset[:]
#                     Res3 = data
#                 else:
#                     dataset = file[dataset_name]
#                     data = dataset[:]
#                     Res.extend(data)
        
#         filepath_Sleep_Events = os.path.join(self.root_path, self.data_path2)
#         with h5py.File(filepath_Sleep_Events, 'r') as file:
#             dataset_names = list(file.keys())
#             Sleep_Events = []
#             for idx, dataset_name in enumerate(dataset_names):
#                 if idx == self.idx:
#                     dataset = file[dataset_name]
#                     data = dataset[:]
#                     Sleep_Events3 = data
#                 else:
#                     dataset = file[dataset_name]
#                     data = dataset[:]
#                     Sleep_Events.extend(data)
        
#         # 将训练数据转换为numpy数组
#         Sleep_Events = np.array(Sleep_Events)
#         Res = np.array(Res)
#         # 找到类别为0的样本索引
#         zero_class_indices = np.where(Sleep_Events == 0)[0]
#         one_class_indices = np.where(Sleep_Events == 1)[0]
#         two_class_indices = np.where(Sleep_Events == 2)[0]
#         print(len(zero_class_indices), len(one_class_indices), len(two_class_indices))
#         delete_ratio = 0.6  # 比如删除一半的0类别样本
#         num_to_delete = int(len(zero_class_indices) * delete_ratio)
#         # 随机选择要删除的0类别样本索引
#         np.random.seed(25)  # 设置随机种子以保证结果可重复
#         delete_indices = np.random.choice(zero_class_indices, num_to_delete, replace=False)
#         # 删除选定的0类别样本
#         Sleep_Events = np.delete(Sleep_Events, delete_indices)
#         Res = np.delete(Res, delete_indices, axis=0)

#         # one_class_indices = np.where(Sleep_Events == 1)[0]
#         # delete_ratio = 0.8  # 比如删除一半的1类别样本
#         # num_to_delete = int(len(one_class_indices) * delete_ratio)
#         # # 随机选择要删除的1类别样本索引
#         # np.random.seed(26)  # 设置随机种子以保证结果可重复
#         # delete_indices = np.random.choice(one_class_indices, num_to_delete, replace=False)
#         # # 删除选定的1类别样本
#         # Sleep_Events = np.delete(Sleep_Events, delete_indices)
#         # Res = np.delete(Res, delete_indices, axis=0)

#         zero_class_indices = np.where(Sleep_Events == 0)[0]
#         one_class_indices = np.where(Sleep_Events == 1)[0]
#         two_class_indices = np.where(Sleep_Events == 2)[0]
#         print(len(zero_class_indices), len(one_class_indices), len(two_class_indices))

#         if self.set_type==0:
#             self.data_x = Res
#             self.data_y = Sleep_Events
#         elif self.set_type==1:
#             self.data_x = Res
#             self.data_y = Sleep_Events
#         elif self.set_type==2:
#             self.data_x = Res3
#             self.data_y = Sleep_Events3

#     def __getitem__(self, index):

#         return self.data_x[index], self.data_y[index]
    
#     def __len__(self):
#         return len(self.data_x)

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)

## 按照人分，有验证集合
class Dataset_Res_SaO2(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path1='ETTh1.csv',data_path2='ETTh1.csv', StudyNumber=1,
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val':1, 'test': 1}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path1 = data_path1
        self.data_path2 = data_path2
        self.idx = StudyNumber
        self.rng =  np.random.default_rng(self.idx)
        self.__read_data__()

    def __read_data__(self):

        filepath_Res = os.path.join(self.root_path, self.data_path1)
        with h5py.File(filepath_Res, 'r') as file:
            dataset_names = list(file.keys())
            Res = []
            for idx, dataset_name in enumerate(dataset_names):
                dataset = file[dataset_name]
                data = dataset[:]
                Res.append(data)
        
        filepath_Sleep_Events = os.path.join(self.root_path, self.data_path2)
        with h5py.File(filepath_Sleep_Events, 'r') as file:
            dataset_names = list(file.keys())
            Sleep_Events = []
            for idx, dataset_name in enumerate(dataset_names):
                dataset = file[dataset_name]
                data = dataset[:]
                Sleep_Events.append(data)

        indices = self.rng.permutation(len(Res))
        Res = [Res[i] for i in indices]
        Sleep_Events = [Sleep_Events[i] for i in indices]
        # num_train = int(len(Sleep_Events) * 0.7)
        # num_test = int(len(Sleep_Events) * 0.1)
        # num_vali = len(Sleep_Events) - num_train - num_test
        
        # border1s = [0, num_train, num_train + num_vali]
        # border2s = [num_train, num_train + num_vali, len(Sleep_Events)]
        # border1 = border1s[self.set_type]
        # border2 = border2s[self.set_type]
        num_train = int(len(Sleep_Events) * 0.8)        
        border1s = [0, num_train]
        border2s = [num_train, len(Sleep_Events)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # 将训练数据转换为numpy数组
        Sleep_Events_ = Sleep_Events[border1:border2]
        Res_ = Res[border1:border2]
        Sleep_Events_ = np.hstack(Sleep_Events_)
        Res_ = np.vstack(Res_)
        
        # 找到类别为0的样本索引
        zero_class_indices = np.where(Sleep_Events_ == 0)[0]
        one_class_indices = np.where(Sleep_Events_ == 1)[0]
        two_class_indices = np.where(Sleep_Events_ == 2)[0]
        print("删除前",len(zero_class_indices), len(one_class_indices), len(two_class_indices))

        delete_ratio = 0.7  # 比如删除一半的0类别样本
        num_to_delete = int(len(zero_class_indices) * delete_ratio)
        # 随机选择要删除的0类别样本索引
        np.random.seed(self.idx)  # 设置随机种子以保证结果可重复
        delete_indices = np.random.choice(zero_class_indices, num_to_delete, replace=False)
        Sleep_Events_ = np.delete(Sleep_Events_, delete_indices, axis=0)
        Res_ = np.delete(Res_, delete_indices, axis=0)
        
        zero_class_indices = np.where(Sleep_Events_ == 0)[0]
        one_class_indices = np.where(Sleep_Events_ == 1)[0]
        two_class_indices = np.where(Sleep_Events_ == 2)[0]

        # 随机选择要删除的1类别样本索引
        delete_ratio = 0.0  # 比如删除一半的0类别样本
        num_to_delete = int(len(one_class_indices) * delete_ratio)
        np.random.seed(self.idx)  # 设置随机种子以保证结果可重复
        delete_indices = np.random.choice(one_class_indices, num_to_delete, replace=False)
        Sleep_Events_ = np.delete(Sleep_Events_, delete_indices, axis=0)
        Res_ = np.delete(Res_, delete_indices, axis=0)
        zero_class_indices = np.where(Sleep_Events_ == 0)[0]
        one_class_indices = np.where(Sleep_Events_ == 1)[0]
        two_class_indices = np.where(Sleep_Events_ == 2)[0]
        print("删除后",len(zero_class_indices), len(one_class_indices), len(two_class_indices))

        self.data_x = Res_
        self.data_y = Sleep_Events_

    def __getitem__(self, index):

        return self.data_x[index], self.data_y[index]
    
    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
# 5折交叉验证
class Dataset_Res_SaO2_KFold(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path1='ETTh1.csv',data_path2='ETTh1.csv', StudyNumber=1,fold=1,
                 target='OT', scale=True, timeenc=0, freq='h'):
        # init
        assert flag in ['train', 'test']
        type_map = {'train': 0, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path1 = data_path1
        self.data_path2 = data_path2
        self.fold = fold
        self.idx = StudyNumber
        self.rng =  np.random.default_rng(self.idx)
        self.__read_data__()

    def __read_data__(self):

        filepath_Res = os.path.join(self.root_path, self.data_path1)
        with h5py.File(filepath_Res, 'r') as file:
            dataset_names = list(file.keys())
            Res = []
            for idx, dataset_name in enumerate(dataset_names):
                dataset = file[dataset_name]
                data = dataset[:]
                Res.append(data)
        
        filepath_Sleep_Events = os.path.join(self.root_path, self.data_path2)
        with h5py.File(filepath_Sleep_Events, 'r') as file:
            dataset_names = list(file.keys())
            Sleep_Events = []
            for idx, dataset_name in enumerate(dataset_names):
                dataset = file[dataset_name]
                data = dataset[:]
                Sleep_Events.append(data)

        # indices = self.rng.permutation(len(Res))
        # Res = [Res[i] for i in indices]
        # Sleep_Events = [Sleep_Events[i] for i in indices]

        # 初始化 KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=self.idx)
        fold_indices = list(kf.split(Res))
        train_val_index, test_index = fold_indices[self.fold]
        # overlap = np.intersect1d(train_val_index, test_index)

        X_train_val = [Res[i] for i in train_val_index]
        X_test = [Res[i] for i in test_index]
        y_train_val = [Sleep_Events[i] for i in train_val_index]
        y_test = [Sleep_Events[i] for i in test_index]
        
        # 从训练和验证集中再次按 4:1 比例划分为训练集和验证集
        # n_train = int(1 * len(X_train_val))  # 80% 用于训练
        
        # X_train = X_train_val[:n_train]
        # y_train = y_train_val[:n_train]
        # X_val = X_train_val[n_train:]
        # y_val = y_train_val[n_train:]
        
        # 将训练数据转换为numpy数组
        if self.set_type ==0:
            Res_ = X_train_val
            Sleep_Events_ = y_train_val
        elif self.set_type ==2:
            Res_ = X_test
            Sleep_Events_ = y_test
        
        Sleep_Events_ = np.hstack(Sleep_Events_)
        Res_ = np.vstack(Res_)
        
        # 找到类别为0的样本索引
        zero_class_indices = np.where(Sleep_Events_ == 0)[0]
        one_class_indices = np.where(Sleep_Events_ == 1)[0]
        two_class_indices = np.where(Sleep_Events_ == 2)[0]
        print("删除前",len(zero_class_indices), len(one_class_indices), len(two_class_indices))

        if self.set_type ==0:
            delete_ratio = 1-len(one_class_indices) / len(zero_class_indices)
        elif self.set_type ==2:
            delete_ratio = 1-(len(one_class_indices) / len(zero_class_indices))

        num_to_delete = int(len(zero_class_indices) * delete_ratio)
        # 随机选择要删除的0类别样本索引
        np.random.seed(self.idx)  # 设置随机种子以保证结果可重复
        delete_indices = np.random.choice(zero_class_indices, num_to_delete, replace=False)
        Sleep_Events_ = np.delete(Sleep_Events_, delete_indices, axis=0)
        Res_ = np.delete(Res_, delete_indices, axis=0)
        
        zero_class_indices = np.where(Sleep_Events_ == 0)[0]
        one_class_indices = np.where(Sleep_Events_ == 1)[0]
        two_class_indices = np.where(Sleep_Events_ == 2)[0]

        # # 随机选择要删除的1类别样本索引
        # delete_ratio = 0.0  # 比如删除一半的0类别样本
        # num_to_delete = int(len(one_class_indices) * delete_ratio)
        # np.random.seed(self.idx)  # 设置随机种子以保证结果可重复
        # delete_indices = np.random.choice(one_class_indices, num_to_delete, replace=False)
        # Sleep_Events_ = np.delete(Sleep_Events_, delete_indices, axis=0)
        # Res_ = np.delete(Res_, delete_indices, axis=0)
        # zero_class_indices = np.where(Sleep_Events_ == 0)[0]
        # one_class_indices = np.where(Sleep_Events_ == 1)[0]
        # two_class_indices = np.where(Sleep_Events_ == 2)[0]
        print("删除后",len(zero_class_indices), len(one_class_indices), len(two_class_indices))

        self.data_x = Res_
        self.data_y = Sleep_Events_

    def __getitem__(self, index):

        return self.data_x[index], self.data_y[index]
    
    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)