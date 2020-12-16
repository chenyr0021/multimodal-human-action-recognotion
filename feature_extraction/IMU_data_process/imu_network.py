import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class Inception1D(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(Inception1D, self).__init__()
        self.b0 = nn.Conv1d(in_channels, out_channels[0], kernel_size=1, stride=2)

        self.b1a = nn.Conv1d(in_channels, out_channels[1], kernel_size=3, padding=1, stride=2)
        self.b1b = nn.Conv1d(out_channels[1], out_channels[2], kernel_size=1)
        self.bn1 = nn.BatchNorm1d(out_channels[2])

        self.b2a = nn.Conv1d(in_channels, out_channels[3], kernel_size=5, padding=2, stride=2)
        self.b2b = nn.Conv1d(out_channels[3], out_channels[4], kernel_size=1)
        self.bn2 = nn.BatchNorm1d(out_channels[4])

        self.b3a = nn.MaxPool1d(kernel_size=3, padding=1, stride=2)
        self.b3b = nn.Conv1d(in_channels, out_channels[5], kernel_size=1)
        self.name = name

    def forward(self, x):
        b0 = F.relu(self.b0(x))
        b1 = self.bn1(F.relu(self.b1b(F.relu(self.b1a(x)))))
        b2 = self.bn2(F.relu(self.b2b(F.relu(self.b2a(x)))))
        b3 = F.relu(self.b3b(self.b3a(x)))
        return torch.cat([b0, b1, b2, b3], dim=1)



class IMU_Network(nn.Module):
    def __init__(self, num_class):
        super(IMU_Network, self).__init__()
        self.nodes = []
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1)
        self.nodes.append(self.conv1)

        self.block1 = Inception1D(in_channels=64, out_channels=[64, 32, 64, 32, 64, 64], name='block1')
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.nodes.extend([self.block1, self.maxpool1])

        self.block2 = Inception1D(in_channels=256, out_channels=[64, 32, 64, 32, 64, 64], name='block2')
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.nodes.extend([self.block2, self.maxpool2])

        self.block3 = Inception1D(in_channels=256, out_channels=[128, 64, 128, 64, 128, 128], name='block3')
        self.nodes.extend([self.block3])

        # self.block4 = Inception1D(in_channels=256, out_channels=[64, 32, 64, 32, 64, 64])
        self.avgpool = nn.AvgPool1d(kernel_size=2)

        self.logits = nn.Conv1d(in_channels=512, out_channels=num_class, kernel_size=1)
        # self.conv = nn.Sequential(nn.Conv1d(in_channels=3, out_channels=64, kernel_size=7, stride=2),
        #         #                           nn.ReLU(True),
        #         #                           nn.Conv1d(in_channels=64, out_channels=256, kernel_size=5, stride=2),
        #         #                           nn.ReLU(True),
        #         #                           nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3),
        #         #                           nn.ReLU(True),
        #         #                           nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3),
        #         #                           nn.ReLU(True),
        #         #                           nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3),
        #         #                           nn.ReLU(True)
        #         #                           )
        #
        #         # self.fc1 = nn.Linear(in_features=768, out_features=256)
        #         # self.fc2 = nn.Linear(in_features=256, out_features=7)
        #         # self.fc3 = nn.Linear(in_features=64, out_features=7)

    def forward(self, x):
        for node in self.nodes:
            # print(x.shape)
            x = node(x)
        x = self.avgpool(x)
        logits = self.logits(x).squeeze(2)
        # print('end: ',logits.shape)

        # x = self.conv(x)
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))
        # # x = F.relu(self.fc2(x))
        # x = F.softmax(self.fc2(x), dim=1)
        return logits

    def extract_features(self, x):
        for node in self.nodes:
            # print(x.shape)
            x = node(x)
        x = self.avgpool(x)
        return x.squeeze(2)








