import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)

args = parser.parse_args()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

# import torchvision
from torchvision import datasets, transforms
import videotransforms


import numpy as np

from pytorch_i3d import InceptionI3d

from charades_dataset import Charades as Dataset


def run(init_lr=0.1, epoch=2000, mode='rgb', root='/home/dataset/Charades_v1_rgb', train_split='./Charades/charades.json', batch_size=16, save_model=''):
    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(train_split, 'training', root, mode, train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # val_dataset = Dataset(train_split, 'testing', root, mode, test_transforms)
    # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(157, in_channels=3)
        i3d.load_state_dict(torch.load('models/rgb_charades.pt'))
    # i3d.replace_logits(157)
    # i3d.load_state_dict(torch.load('./models/charades_ego.pt'))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = init_lr
    optimizer = optim.Adam(i3d.parameters(), lr=lr, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [30, 100])


    num_steps_per_update = 4 # accum gradient
    steps = 0
    # train it
    for ep in range(epoch):# for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(ep, epoch))
        print('-' * 10)

        i3d.train(True)
        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0
        optimizer.zero_grad()

        # Iterate over data.
        for i, data in enumerate(dataloader):
            # get the inputs
            vid, inputs, labels = data

            # wrap them in Variable
            inputs = Variable(inputs.cuda())
            t = inputs.size(2)
            labels = Variable(labels.cuda())

            per_frame_logits = i3d(inputs)
            # upsample to input size
            per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear', align_corners=True)

            # compute localization loss
            loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
            tot_loc_loss += loc_loss.data

            # compute classification loss (with max-pooling along time B x C x T)
            cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
            tot_cls_loss += cls_loss.data

            loss = 0.5*loc_loss + 0.5*cls_loss
            tot_loss += loss.data
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            lr_sched.step()

        print('Train Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(tot_loc_loss/len(dataset),
                                                                                tot_cls_loss/len(dataset),
                                                                                tot_loss/len(dataset)))

        # save model
        if ep % 10 == 0:
            torch.save(i3d.module.state_dict(), save_model+ str(ep).zfill(3) + '.pt')



if __name__ == '__main__':
    # need to add argparse
    run(root='/home/backup/CharadesEgo_v1_rgb', train_split='./CharadesEgo/charades_ego.json', save_model='./models/charades_ego_')
