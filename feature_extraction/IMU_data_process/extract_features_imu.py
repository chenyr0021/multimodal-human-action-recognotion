import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-load_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-gpu', type=str)
parser.add_argument('-save_dir', type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms


import numpy as np

from imu_network import IMU_Network

from imu_dataset import IMU_dataset_extract


def run(root, save_dir, slide_win=21):
    # setup dataset
    test_transforms = transforms.Compose([transforms.Resize(224)])
    for side in [1,0]:
        dataset = IMU_dataset_extract(root, side=side)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

        print('load model...')
        imu = IMU_Network(7)
        imu.load_state_dict(torch.load('./assemble_imu_%d.pt'%side))
        imu.cuda()
        imu = nn.DataParallel(imu, device_ids=[0,1,2,3])
        imu.eval()  # Set model to evaluate mode

        # Iterate over data.
        print('processing data...')
        for data_dic,name in dataset:
            # get the inputs
            # if os.path.exists(os.path.join(save_dir, str(side), name+'.npy')):
            #     print(name+'.npy exists.')
            #     continue

            features = []
            t = len(data_dic)
            # print(t, max(data_dic.keys()))
            for start in range(t - slide_win + 1):
                ip = torch.from_numpy(np.concatenate([data_dic[j] for j in range(start, start + slide_win)]))
                if ip.shape[0] < 50:
                    ip = torch.cat([ip, torch.zeros((50-ip.shape[0], 3))], dim=0)
                _, ind = torch.topk(torch.norm(ip, p=1, dim=1), 50)
                ind = torch.sort(ind)
                ip = ip[ind[0], :].T
                ip = ip.unsqueeze(0)
                # print(ip.shape)

                out = imu.module.extract_features(ip.cuda()).cpu()
                # print(out.shape)

                features.append(out.squeeze(0).unsquee
                ze(-1).detach().numpy())
            np_feature = np.concatenate(features, axis=1)
            print(np_feature.shape)
            np.save(os.path.join(save_dir, str(side), name), np_feature)
            print('save %s finished.' % os.path.join(save_dir, str(side), name))


if __name__ == '__main__':
    # need to add argparse
    run(root='../assemble_dataset/imu', save_dir='/home/backup/data_cyr/assemble/features_imu_lr')
