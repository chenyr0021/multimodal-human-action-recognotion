from torch import nn
# from non_local.non_local_concatenation import NONLocalBlock2D
# from non_local.non_local_gaussian import NONLocalBlock2D
from non_local.non_local_embedded_gaussian import NONLocalBlock2D
# from non_local.non_local_dot_product import NONLocalBlock2D
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.name = 'Network'

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv_3 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=256 * 2 * 2, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(in_features=256, out_features=10)
        )

    def forward(self, x):
        batch_size = x.size(0)

        feature_1 = self.conv_1(x)

        feature_2 = self.conv_2(feature_1)

        feature_3 = self.conv_3(feature_2)

        output = self.conv_4(feature_3).view(batch_size, -1)
        output = self.fc(output)

        return output

class Network_NL(nn.Module):
    def __init__(self):
        super(Network_NL, self).__init__()
        self.name = 'Network_NL'

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.nl_1 = NONLocalBlock2D(in_channels=32)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.nl_2 = NONLocalBlock2D(in_channels=64)
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.nl_3 = NONLocalBlock2D(in_channels=128)
        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.nl_4 = NONLocalBlock2D(in_channels=256)

        self.fc = nn.Sequential(
            nn.Linear(in_features=256 * 2 * 2, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(in_features=256, out_features=10)
        )

    def forward(self, x):
        batch_size = x.size(0)

        feature_1 = self.conv_1(x)
        nl_feature_1 = self.nl_1(feature_1)

        feature_2 = self.conv_2(nl_feature_1)
        nl_feature_2 = self.nl_2(feature_2)

        feature_3 = self.conv_3(nl_feature_2)
        nl_feature_3 = self.nl_3(feature_3)

        output = self.conv_4(nl_feature_3)
        output = self.nl_4(output).view(batch_size, -1)
        output = self.fc(output)

        return output

    def forward_with_nl_map(self, x):
        batch_size = x.size(0)

        feature_1 = self.conv_1(x)
        nl_feature_1, nl_map_1 = self.nl_1(feature_1, return_nl_map=True)

        feature_2 = self.conv_2(nl_feature_1)
        nl_feature_2, nl_map_2 = self.nl_2(feature_2, return_nl_map=True)

        output = self.conv_3(nl_feature_2).view(batch_size, -1)
        output = self.fc(output)

        return output, [nl_map_1, nl_map_2]


if __name__ == '__main__':
    import torch

    img = torch.randn(3, 1, 28, 28)
    net = Network()
    out = net(img)
    print(out.size())

