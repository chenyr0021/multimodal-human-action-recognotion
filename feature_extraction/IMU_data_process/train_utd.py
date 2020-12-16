import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# from IMU_data_process.imu_network import IMU_Network
# from IMU_data_process.imu_dataset import IMU_dataset

class IMU_dataset(Dataset):
    def __init__(self, root='./Inertial_np/train'):
        super(IMU_dataset, self).__init__()
        self.path = []
        self.label = []
        for file in os.listdir(root):
            self.path.append(os.path.join(root, file))
            self.label.append(int(file.split('_')[0][1:])-1)

    def __getitem__(self, item):
        file_name = self.path[item]
        data = torch.from_numpy(np.load(file_name)).float()
        data = data.reshape((300, 3, 2))
        data = data.transpose(0, 2)
        label = self.label[item]
        return data, label

    def __len__(self):
        return len(self.label)


class IMU_Network(nn.Module):
    def __init__(self):
        super(IMU_Network, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=128, kernel_size=(3, 10), stride=10),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(True),
                                  nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,3), stride=3),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))
        # self.conv1 = nn.Conv2d(in_channels=2, out_channels=128, kernel_size=(3, 10), stride=10)
        # self.bn = nn.BatchNorm2d(128)
        # self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,3), stride=3)
        self.fc1 = nn.Linear(in_features=2560, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=27)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        return x

def visualize(input, label):
    # acc
    plt.subplot(211)
    frame = np.linspace(1, 300 ,300)
    # [acc/gyro, x/y/z, frame]
    x = input[0,0,:]
    y = input[0,1,:]
    z = input[0,2,:]
    # frame = np.linspace(1,3,3)
    # x = [1,2,3]
    # y = [4,5,6]
    # z = [7,8,9]
    # print(x)
    # print(y)
    # print(z)
    plt.plot(frame, x, frame, y, frame, z)
    plt.show()


def train(batch_size=8):
    dataset = IMU_dataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    imu_net = IMU_Network()
    imu_net.cuda()
    imu_net.train()
    # imu_net.load_state_dict(torch.load('utd_imu.pt'))
    optimizer = optim.SGD(params=imu_net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for i in range(50):
        loss = 0
        acc = 0
        for inputs, labels in dataloader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            output = imu_net(inputs)
            running_loss = criterion(output, labels)
            optimizer.zero_grad()
            running_loss.backward()
            optimizer.step()

            _, pred = torch.max(output, dim=1)
            running_acc = (pred==labels).float().mean()
            acc += (pred==labels).sum().item()
            loss += running_loss.item()

            # print('epoch: {}, loss: {:.4f}, acc: {:.4f}'.format(i+1, running_loss.item(), running_acc.item()))
        print('epoch: {}, loss: {:.4f}, acc: {:.4f}'.format(i+1, loss, acc/len(dataset)))
    torch.save(imu_net.state_dict(), 'utd_imu.pt')

def test(batch_size=8):
    dataset = IMU_dataset(root='./Inertial_np/test')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    imu_net = IMU_Network()
    imu_net.cuda()
    imu_net.load_state_dict(torch.load('utd_imu.pt'))
    imu_net.eval()

    true_nums = 0
    total_nums = 0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        out = imu_net(inputs)
        _, pred = torch.max(out, 1)
        true_nums += (pred==labels).sum().float()
        total_nums += len(labels)

        if total_nums > len(dataset)/4:
            # print(pred, '\n', labels)
            # for i in range(len(labels)):
            # visualize(inputs[0].cpu().numpy(), labels[0].cpu().numpy())
            break

    print('Test acc: {:.4f}'.format(true_nums/total_nums))

if __name__ == '__main__':
    train(128)
    test()

    # # 划分数据集
    # import os
    # import shutil
    # from random import shuffle
    # file_list = os.listdir('./Inertial_np')
    # shuffle(file_list)
    # os.mkdir('./Inertial_np/test')
    # for i in range(len(file_list)//4):
    #     shutil.move(os.path.join('./Inertial_np', file_list[i]), os.path.join('./Inertial_np/test', file_list[i]))