import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from MyLittleHelpers import prod, sep
import time


# TODO Write NN class with variable overall structure: Arbitrary number of layers - different compositions of layers etc.

# TODO Write NN classes with variable layer architecture

def _get_get_num_flat_features(num_inp_ch, inp_shape, forward_preprocess=None):
    x = torch.randn(1, num_inp_ch, *inp_shape, requires_grad=False)
    if forward_preprocess is not None:
        x = forward_preprocess(x)
    return prod(x.size())


def _get_data_shape(inp_shape):
    return inp_shape[-3], list(inp_shape[-2:])


def train_Net(Net, x):
    sep()
    # print(Net)
    print(type(x))
    # print(type(x.size()))
    # print(x.size())
    # x = Net(x)
    # print(x.size())
    sep()
    pass


#
# def train_Net_standard(Net, train_loader, N_epoch, criterion, optimizer, device):
#     start_time_0 = time.time()
#     time_epoch = np.zeros((N_epoch,))
#
#     print('Start Training')
#     for epoch in range(N_epoch):
#         start_time_epoch = time.time()
#         N_seen = 0
#         running_loss = 0
#         for i, data in enumerate(train_loader, start=0):
#             inputs, labels = data
#             batchsize = inputs.size()[0]
#
#             inputs, labels = inputs.to(device), labels.to(device)
#
#             optimizer.zero_grad()
#             outputs = Net(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             N_seen += flg_batchsize
#
#             # running_loss += loss.item()
#             # if N_seen >= train_set.__len__() // 5:
#             #     print('[%d, %5d] loss: %.3f' %
#             #           (epoch + 1, N_seen, running_loss / N_seen))
#             #     N_seen -= train_set.__len__() // 5
#             #     running_loss = 0
#         time_epoch[epoch] = time.time() - start_time_epoch
#         time_estimate = (N_epoch - (epoch + 1)) * np.mean(time_epoch[time_epoch.nonzero()])
#         print('Estimated time remaining: {0:5.1f} seconds'.format(time_estimate))
#
#     print('Finished Training - Duration: {0:5.1f} seconds'.format(time.time() - start_time_0))


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
        self.num_inp_channels, self.data_shape = _get_data_shape(inp_shape)

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(in_channels=self.num_inp_channels, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)

        self.num_flat_features = _get_get_num_flat_features(self.num_inp_channels, self.data_shape,
                                                            self.forward_preprocess)

        self.fc1 = nn.Linear(in_features=self.num_flat_features, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

        self.num_out_features = self.fc2.out_features

    def forward_preprocess(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

    def forward(self, x):
        x = self.forward_preprocess(x)
        x = x.view(-1, self.num_flat_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def forward_part(self, x):
        x = self.forward_preprocess(x)
        x = x.view(-1, self.num_flat_features)
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

    def train(self, x):
        train_Net(self, x)


class FFsmall(nn.Module):

    def __init__(self, inp_shape):
        super(FFsmall, self).__init__()
        self.num_inp_channels, self.data_shape = _get_data_shape(inp_shape)
        self.num_flat_features = _get_get_num_flat_features(self.num_inp_channels, self.data_shape)

        self.dropout = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(in_features=self.num_flat_features, out_features=512)
        self.bn1 = nn.BatchNorm1d(num_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=64)
        self.fc5 = nn.Linear(in_features=64, out_features=10)

        self.num_out_features = self.fc4.out_features

        # TODO Make this a ResNet (?)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features)
        x = self.dropout(self.bn1(F.relu(self.fc1(x))))
        x = self.dropout(self.bn2(F.relu(self.fc2(x))))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)

    def forward_part(self, x):
        x = x.view(-1, self.num_flat_features)
        x = self.dropout(self.bn1(F.relu(self.fc1(x))))
        x = self.dropout(self.bn2(F.relu(self.fc2(x))))
        x = F.relu(self.fc3(x))
        return F.relu(self.fc4(x))


class EnsembleMLP(nn.Module):

    def __init__(self, Net_list):
        super(EnsembleMLP, self).__init__()

        self.Net_list = Net_list

        self.num_in_features = 0
        for Net_base in self.Net_list:
            Net_base.eval()
            self.num_in_features += Net_base.num_out_features

        self.fc1 = nn.Linear(in_features=self.num_in_features, out_features=64)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x_bases = []
        for Net_base, x_base in zip(self.Net_list, x):
            x_base = x_base.float()
            x_out = Net_base.forward_part(x_base)
            x_bases.append(x_out)
        x = torch.cat(x_bases, 1)

        x = self.dropout(self.bn1(F.relu(self.fc1(x))))
        return self.fc2(x)


if __name__ == '__main__':
    x_real = torch.randn(4, 1, 28, 28)
    x_ft = torch.randn(4, 2, 28, 28)

    CNet = CNNsmall(x_real.size())
    FFNet = FFsmall(x_ft.size())
    EnsNet = EnsembleMLP([CNet, FFNet])

    # sep()
    # print(EnsNet.parameters())
    # sep()
    CNet.train(x_real)

    # _x_real = CNet.forward(x_real)
    # _x_ft = FFNet.forward(x_ft)
    # _x_ens = EnsNet([x_real, x_ft])
