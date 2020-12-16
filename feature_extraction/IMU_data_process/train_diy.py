import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='4,5'
from imu_network import IMU_Network
from imu_dataset import IMU_dataset


def train(batch_size=64, side = 0):
    slide_win = 21
    dataset = IMU_dataset('../assemble_dataset', slide_win, side, mode='train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    imu_net = IMU_Network(num_class=7)
    imu_net.cuda()
    imu_net.train()
    # imu_net.load_state_dict(torch.load('utd_imu.pt'))
    optimizer = optim.SGD(params=imu_net.parameters(), lr=0.008, momentum=0.9)
    criterion = nn.CrossEntropyLoss()


    for i in range(30):
        loss = 0
        acc = 0
        for inputs, labels in dataloader:
            # data_dic = data_dic.cuda()
            # labels_dic = labels_dic.cuda()

            inputs = inputs.cuda().requires_grad_()
            # print(inputs.requires_grad)
            labels = labels.cuda()
            # exit()
            output = imu_net(inputs)
            pred = torch.argmax(output, dim=1)
            running_loss = criterion(output, labels)
            optimizer.zero_grad()
            running_loss.backward()
            optimizer.step()

            # print(torch.sum(pred==labels), torch.numel(pred))
            running_acc = torch.sum(pred==labels).item()/torch.numel(pred)
            acc += torch.sum(pred==labels)
            loss += running_loss.item()

            # print('epoch: {}, loss: {:.4f}, acc: {:.4f}'.format(i+1, running_loss.item(), running_acc))
        print('epoch: {}, loss: {:.4f}, acc: {:.4f}'.format(i+1, loss / len(dataset), acc.cpu().numpy()/len(dataset)))
    torch.save(imu_net.state_dict(), 'assemble_imu_%d.pt'%side)

def validation(batch_size=64, side=0):
    slide_win = 21
    dataset = IMU_dataset('../assemble_dataset', slide_win, side, mode='test')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    imu_net = IMU_Network(num_class=7)
    imu_net.cuda()
    imu_net.eval()
    imu_net.load_state_dict(torch.load('assemble_imu_%d.pt'%side))
    # optimizer = optim.SGD(params=imu_net.parameters(), lr=0.008, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    loss = 0
    acc = 0
    for inputs, labels in dataloader:
        # data_dic = data_dic.cuda()
        # labels_dic = labels_dic.cuda()

        inputs = inputs.cuda()
        # print(inputs.requires_grad)
        labels = labels.cuda()
        # exit()
        output = imu_net(inputs)
        pred = torch.argmax(output, dim=1)
        running_loss = criterion(output, labels)
        # optimizer.zero_grad()
        # running_loss.backward()
        # optimizer.step()

        # print(torch.sum(pred==labels), torch.numel(pred))
        running_acc = torch.sum(pred == labels).item() / torch.numel(pred)
        acc += torch.sum(pred == labels)
        loss += running_loss.item()

        # print('epoch: {}, loss: {:.4f}, acc: {:.4f}'.format(i+1, running_loss.item(), running_acc))
    print('loss: {:.4f}, acc: {:.4f}'.format(loss / len(dataset), acc.cpu().numpy() / len(dataset)))
    # torch.save(imu_net.state_dict(), 'assemble_imu_%d.pt' % side)

if __name__ == '__main__':
    train(side=0)
    validation(side=0)
    train(side=1)
    validation(side=1)