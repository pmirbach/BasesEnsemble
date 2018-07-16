import torch
import torch.nn as nn
import torch.nn.functional as F



class CNN_small(nn.Module):
    '''
    Small convolutional neural network for minor tests.
        - 2 * (conv + maxpool)
        - 3 * MLP
        - Relu everywhere

    Args:
        inp_shape (list): Shape of input data: [#Channel, Height, Width]
    '''
    def __init__(self, inp_shape):
        super(CNN_small, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
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

    def _get_num_flat_features(self, shape):
        x = torch.randn(1, *shape)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        size = x.size()
        num_features = 1
        for s in size:
            num_features *= s
        return num_features