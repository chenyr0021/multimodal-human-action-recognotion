import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
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

import torchvision
from torchvision import datasets, transforms
import videotransforms

import numpy as np

from pytorch_i3d import InceptionI3d

from charades_dataset import Charades as Dataset

def wirte2txt(file, vids, scores):
    with open(file, 'a') as f:
        for i in range(len(vids)):
            f.write(vids[i] + ' ')
            f.write(' '.join(map(str, scores[i].tolist())))
            f.write('\n\n')


def run(mode='rgb', root='/home/dataset/Charades_v1_rgb',
        train_split='./Charades/charades.json', batch_size=8):
    # create a txt file to save results
    import time
    cur_time = time.strftime('%Y-%m-%d_%H%M%S', time.localtime(time.time()))
    res_file = './test_scores/' + cur_time+'_charades_ego_scores.txt'
    os.mknod(res_file)

    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    val_dataset = Dataset(train_split, 'testing', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                                 pin_memory=True)
    print("Loading model......")
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(157, in_channels=3)
        i3d.load_state_dict(torch.load('models/rgb_charades.pt'))
    # i3d.replace_logits(157)
    # i3d.load_state_dict(torch.load('models/charades_ego.pt'))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    num_iter = 1

    # Each epoch has a training and validation phase
    i3d.eval()

    tot_loss = 0.0
    tot_loc_loss = 0.0
    tot_cls_loss = 0.0

    print("Start testing......")
    print('-'*20)
    # Iterate over data.
    for data in val_dataloader:
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
        per_video_logits = torch.max(per_frame_logits, dim=2)[0]
        wirte2txt(res_file, vid, per_video_logits)
        # print('{:.5f}\t{}'.format(F.sigmoid(),torch.max(labels, dim=2)[0][i][j]) )
        # print(per_frame_logits.size(), torch.max(per_frame_logits, dim=2)[0].size())
        cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0],
                                                      torch.max(labels, dim=2)[0])
        tot_cls_loss += cls_loss.data

        loss = (0.5 * loc_loss + 0.5 * cls_loss)
        tot_loss += loss.data


        if num_iter % 10 == 0:
            print('Test {}: Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(num_iter, tot_loc_loss / 10,
                                                                                    tot_cls_loss / 10,
                                                                                 tot_loss / 10))
            # save model
            # torch.save(i3d.module.state_dict(), save_model+str(steps).zfill(6)+'.pt')
            tot_loss = tot_loc_loss = tot_cls_loss = 0.
        num_iter += 1
    print('Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(tot_loc_loss / (num_iter%10),
                                                                      tot_cls_loss / (num_iter%10),
                                                                      tot_loss / (num_iter%10)))


if __name__ == '__main__':
    # need to add argparse
    run(root='/home/backup/CharadesEgo_v1_rgb', train_split='./CharadesEgo/charades_ego.json')
