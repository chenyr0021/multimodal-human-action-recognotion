import scipy.io as sio
import numpy as np
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
# def mat2np():
#     root = './Inertial'
#     new_root = './Inertial_np'
#
#     file_names = os.listdir(root)
#     for file in file_names:
#         data = sio.loadmat(os.path.join(root, file))['d_iner']
#         data = np.resize(data, (300, 6))
#         np.save(os.path.join(new_root, file.split('.')[0]+'.npy'), data)
#
#
# class IMU_dataset(Dataset):
#     def __init__(self, root='./Inertial_np/train'):
#         super(IMU_dataset, self).__init__()
#         self.path = []
#         self.label = []
#         for file in os.listdir(root):
#             self.path.append(os.path.join(root, file))
#             self.label.append(int(file.split('_')[0][1:])-1)
#
#     def __getitem__(self, item):
#         file_name = self.path[item]
#         data = torch.from_numpy(np.load(file_name)).float()
#         data = data.reshape((300, 3, 2))
#         data = data.transpose(0, 2)
#         label = self.label[item]
#         return data, label
#
#     def __len__(self):
#         return len(self.label)


class IMU_dataset(Dataset):
    '''
    1. read csv file
    2. read imu data in 21-long window
    3. return topk
    '''
    def __init__(self, root, slide_win, side, mode):
        '''

        :param root: root path of dataset
        :param slide_win: length of slide window
        :param side: left: 0 or right: 1
        '''
        self.slide_win = slide_win
        split_file = root + '/split/%s_split_%d.txt'%(mode, side)
        split_ptr = open(split_file, 'r')
        files = split_ptr.read().split('\n')[:-1]
        # files = os.listdir(root+'/imu')
        self.data_files = [os.path.join(root+'/imu', f) for f in files]
        self.label_files_root = os.path.join(root, 'annotation/')
        mapping_file = root+'/mapping.txt'
        self.actions_dict = {}
        file_ptr = open(mapping_file, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        for a in actions:
            self.actions_dict[a.split()[1]] = int(a.split()[0])
        self.inputs = []
        self.labels = []
        for i in range(len(self.data_files)):
            ip, lb = self.get_file_data(i)
            self.inputs.extend(ip)
            self.labels.extend(lb)



    def get_file_data(self, index):
        data_file = self.data_files[index]
        label_file = self.label_files_root + data_file.split('/')[-1].split('_')[0] + '.csv'
        df = pd.read_csv(data_file, usecols=lambda x: x != 'timestamp')
        df2 = pd.read_csv(label_file, usecols=lambda x: x != 'timestamp')
        file_data_dic = {}
        frame_cnt = int(df['video_frame_seq'].iloc[-1])
        for i in range(frame_cnt + 1):
            file_data_dic[i] = df.loc[df['video_frame_seq'] == i][['acc_x', 'acc_y', 'acc_z']].to_numpy(dtype=np.float32)
            # print(data_dic)
            # exit()
        file_labels = [i for i in df2['class']]

        inputs = []
        labels = []
        t = len(file_data_dic)
        for start in range(t - self.slide_win + 1):
            ip = torch.from_numpy(np.concatenate([file_data_dic[j] for j in range(start, start + self.slide_win)]))
            # label 取滑窗中间的值
            lb = file_labels[start + self.slide_win // 2]
            # print(ip.shape)
            if ip.shape[0] < 50:
                ip = torch.cat([ip, torch.zeros((50-ip.shape[0], 3))], dim=0)
            _, ind = torch.topk(torch.norm(ip, p=1, dim=1), 50)
            ind = torch.sort(ind)
            ip = ip[ind[0], :]
            inputs.append(ip.T)
            labels.append(lb)
            # print(ip.shape)
        return inputs, labels
        # inputs = torch.cat(inputs).cuda().requires_grad_()
        # # print(inputs.requires_grad)
        # labels = torch.tensor(labels, dtype=torch.int64).cuda()


    def __getitem__(self, index):
        # df = pd.read_csv(self.data_files[index], usecols=lambda x: x != 'timestamp')
        # df2 = pd.read_csv(self.label_files[index], usecols=lambda x: x != 'timestamp')
        # data_dic = {}
        # frame_cnt = int(df['video_frame_seq'].iloc[-1])
        # for i in range(frame_cnt+1):
        #     data_dic[i] = df.loc[df['video_frame_seq']==i][['acc_x','acc_y','acc_z']].to_numpy(dtype=np.float32)
        #     # print(data_dic)
        #     # exit()
        # labels = [self.actions_dict[i] for i in df2['class']]
        #
        # # print(label_dic[0])
        # return data_dic, labels
        return self.inputs[index], self.labels[index]


    def __len__(self):
        return len(self.labels)


class IMU_dataset_extract(Dataset):
    def __init__(self, root, side):
        files = os.listdir(root)
        self.data_files = [os.path.join(root, f) for f in files if f.split('.')[0][-1] == str(side)]



    def __getitem__(self, index):
        df = pd.read_csv(self.data_files[index], usecols=lambda x: x != 'timestamp')
        name = self.data_files[index].split('/')[-1].split('.')[0]
        data_dic = {}
        frame_cnt = int(df['video_frame_seq'].iloc[-1])
        for i in range(frame_cnt+1):
            data_dic[i] = df.loc[df['video_frame_seq']==i][['acc_x','acc_y','acc_z']].to_numpy(dtype=np.float32)
            # print(data_dic)
            # exit()

        # print(label_dic[0])
        return data_dic, name
    def __len__(self):
        return len(self.data_files)

if __name__ == '__main__':

    dataset = IMU_dataset('../assemble_dataset', 21, 'train', 0)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print('len of dataset: ', len(dataset))
    for data, label in dataloader:
        print(data[0], label[0])
        break