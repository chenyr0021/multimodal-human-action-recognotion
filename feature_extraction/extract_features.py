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
import videotransforms


import numpy as np

from pytorch_i3d import InceptionI3d

from salads_dataset import Salads50_without_label


def run(root, load_model, save_dir, batch_size=1):
    # setup dataset
    test_transforms = transforms.Compose([transforms.RandomCrop((224, 224)), transforms.ToTensor()])

    dataset = Salads50_without_label(root, test_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print('load model...')
    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load(load_model))
    i3d.cuda()
    i3d = nn.DataParallel(i3d, device_ids=[0,1,2,3])
    i3d.eval()  # Set model to evaluate mode

    # Iterate over data.
    print('processing data...')
    for inputs, name in dataloader:
        # get the inputs
        # if os.path.exists(os.path.join(save_dir, name[0]+'.npy')):
        #         #     print(os.path.join(save_dir, name[0]+'.npy'), ' already exist.')
        #         #     continue

        b,c,t,h,w = inputs.shape
        print(name[0], inputs.shape)
        features = []
        for start in range(t-20):
            ip = Variable(torch.from_numpy(inputs.numpy()[:,:,start:start+21]).cuda())
            out = i3d.module.extract_features(ip).cpu()
            features.append(out.squeeze(0).detach().numpy())
        np_feature = np.concatenate(features, axis=1)
        print(np_feature.shape)
        np.save(os.path.join(save_dir, name[0]), np_feature)
        print('save %s finished.' % os.path.join(save_dir, name[0]))


if __name__ == '__main__':
    # need to add argparse
    run(root='/home/backup/data_cyr/assemble_ori', load_model='./models/rgb_imagenet.pt', save_dir='/home/backup/data_cyr/assemble/features_video')
