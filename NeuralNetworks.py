import torch
import torch.nn as nn
import torch.nn.functional as F

from MyLittleHelpers import prod


# TODO Write NN class with variable overall structure: Arbitrary number of layers - different compositions of layers etc.

# TODO Write NN classes with variable layer architecture


class CNNsmall(nn.Module):
    """
    Small convolutional neural network for minor tests.
        - 2 * (conv + maxpool)
        - 3 * MLP
        - Relu everywhere

    Args:
        inp_shape (list): Shape of input data: [#Channel, Height, Width]
    """

    def __init__(self, inp_shape):
        super(CNNsmall, self).__init__()
        self.data_shape = inp_shape

        self.conv1 = nn.Conv2d(in_channels=inp_shape[0], out_channels=10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.num_flat_features = self._get_num_flat_features(inp_shape)

        self.fc1 = nn.Linear(in_features=self.num_flat_features, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = x.view(-1, self.num_flat_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_part(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = x.view(-1, self.num_flat_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def _get_num_flat_features(self, shape):
        x = torch.randn(1, *shape)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        size = x.size()
        num_features = prod(size)
        # num_features = 1
        # for s in size:
        #     num_features *= s
        return num_features


class FFsmall(nn.Module):

    def __init__(self, inp_shape):
        super(FFsmall, self).__init__()
        self.data_shape = inp_shape
        self.num_features = prod(inp_shape)

        self.fc1 = nn.Linear(in_features=self.num_features, out_features=512)
        self.bn1 = nn.BatchNorm1d(num_features=512)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=64)
        self.fc5 = nn.Linear(in_features=64, out_features=10)

        # TODO Make this a ResNet (?)

    def forward(self, x):
        x = x.view(-1, self.num_features)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


if __name__ == '__main__':
    x = torch.randn(1, 1, 28, 28)
    xshape = x.size()

    CNet = CNNsmall(xshape)
    FFNet = FFsmall(xshape)

    CNet.forward(x)
    FFNet.forward(x)
