import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import DataLoader

import torchvision

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


def set_grad_false(layer):
    if hasattr(layer, 'weight'):
        layer.weight.requires_grad = False
    if hasattr(layer, 'bias'):
        layer.bias.requires_grad = False

def inp_split_ensemble(data, device):
    inputs = data[0]
    inputs = [inp.to(device) for inp in inputs]
    labels = data[1].to(device)
    return inputs, labels


def train_Net_standard(Net, train_loader, N_epoch, criterion, optimizer, device, inp_split=None):
    Net.to(device)
    time_epoch = np.zeros((N_epoch,))

    print('Start Training...')

    for epoch in range(N_epoch):

        t0_epoch = time.time()
        epoch_loss = 0

        for i, data in enumerate(train_loader, start=0):

            if inp_split is None:
                inputs, labels = data[0].to(device), data[1].to(device)
            else:
                inputs, labels = inp_split(data, device)
            # inputs = data[0]
            # inputs = [inp.to(device) for inp in inputs]
            #
            # labels = data[1].to(device)



            optimizer.zero_grad()
            outputs = Net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        time_epoch[epoch] = time.time() - t0_epoch
        time_left_est = (N_epoch - (epoch + 1)) * np.mean(time_epoch[time_epoch.nonzero()])

        print("===> Epoch [{:2d}] / {}: Loss: {:.4f} - Approx. {:5.1f} seconds left".format(
            epoch + 1, N_epoch, epoch_loss / len(train_loader), time_left_est))

    print('Finished Training - Duration: {0:5.1f} seconds'.format(np.sum(time_epoch)))


def validate_Net_standard(Net, test_loader, device, inp_split=None):
    Net.eval()
    num_images = len(test_loader.dataset)
    batchsize = test_loader.batch_size

    res = np.zeros(num_images)
    with torch.no_grad():
        for i, data in enumerate(test_loader):

            if inp_split is None:
                inputs, labels = data[0].to(device), data[1].to(device)
            else:
                inputs, labels = inp_split(data, device)
            # inputs = data[0]
            # inputs = [inp.to(device) for inp in inputs]
            #
            # labels = data[1].to(device)

            # inputs, labels = data[0].to(device), data[1].to(device)

            outputs = Net(inputs)
            _, predicted = torch.max(outputs.data, 1)

            comp = predicted == labels

            try:
                res[i * batchsize: (i + 1) * batchsize] = comp
            except ValueError:
                res[-comp.shape[0]:] = comp

    print('Accuracy of the network on the {} test images: {} %'.format(num_images, 100 * np.sum(res) / num_images))
    return res


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

    def train_Net(self, train_loader, N_epoch, criterion, optimizer, device):
        train_Net_standard(self, train_loader, N_epoch, criterion, optimizer, device)

    def validate_Net(self, test_loader, device):
        self.predictions = validate_Net_standard(self, test_loader, device)
        return self.predictions


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

    def train_Net(self, train_loader, N_epoch, criterion, optimizer, device):
        train_Net_standard(self, train_loader, N_epoch, criterion, optimizer, device)

    def validate_Net(self, test_loader, device):
        self.predictions = validate_Net_standard(self, test_loader, device)
        return self.predictions


class EnsembleMLP(nn.Module):

    def __init__(self, Net_list):
        super(EnsembleMLP, self).__init__()

        self.Net_list = Net_list

        self.num_in_features = 0
        for Net_base in self.Net_list:
            Net_base.eval()
            for child in Net_base.children():
                set_grad_false(child)
            self.num_in_features += Net_base.num_out_features

        self.dropout = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(in_features=self.num_in_features, out_features=256)
        self.bn1 = nn.BatchNorm1d(num_features=256)

        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.bn2 = nn.BatchNorm1d(num_features=128)

        self.fc3 = nn.Linear(in_features=128, out_features=64)

        self.fc4 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x_bases = []
        for Net_base, x_base in zip(self.Net_list, x):
            x_base = x_base.float()
            x_out = Net_base.forward_part(x_base)
            x_bases.append(x_out)
        x = torch.cat(x_bases, 1)

        x = self.dropout(self.bn1(F.relu(self.fc1(x))))
        x = self.dropout(self.bn2(F.relu(self.fc2(x))))
        x = self.fc3(x)
        return self.fc4(x)

    def train_Net(self, train_loader, N_epoch, criterion, optimizer, device):
        train_Net_standard(self, train_loader, N_epoch, criterion, optimizer, device, inp_split_ensemble)

    def validate_Net(self, test_loader, device):
        self.predictions = validate_Net_standard(self, test_loader, device, inp_split_ensemble)
        return self.predictions


if __name__ == '__main__':
    x_real = torch.randn(4, 1, 28, 28)
    x_ft = torch.randn(4, 2, 28, 28)

    CNet = CNNsmall(x_real.size())
    FFNet = FFsmall(x_real.size())
    EnsNet = EnsembleMLP([CNet, FFNet])


    # for child in CNet.children():
    #     print(type(child.bias()))
    #     print(child)

    def inspec_Net(Net):
        print(Net)
        print()
        for child in Net.children():

            print(child)

            if hasattr(child, 'weight'):
                print('has weight')
                print(child.weight.requires_grad)
            if hasattr(child, 'bias'):
                print('has bias')
                print(child.bias.requires_grad)

            for param in child.parameters():
                print(param.size())
            print()


    inspec_Net(FFNet)

    for child in FFNet.children():
        set_grad_false(child)

    inspec_Net(FFNet)

    # for param in EnsNet.parameters():
    #     print(param.size())

    # # sep()
    # # print(EnsNet.parameters())
    # # sep()
    # # CNet.train_Net(x_real)
    #
    # # _x_real = CNet.forward(x_real)
    # # _x_ft = FFNet.forward(x_ft)
    # # _x_ens = EnsNet([x_real, x_ft])
    #
    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,))
    # ])
    #
    # train_set = torchvision.datasets.MNIST(root="./data/", train=True, transform=transform, download=True)
    # train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True, num_workers=10)
    # test_set = torchvision.datasets.MNIST(root="./data/", train=False, transform=transform, download=True)
    # test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=False, num_workers=10)
    #
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(CNet.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
    # optimizer = optim.SGD(FFNet.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
    #
    #
    # # CNet.train_Net(train_loader, 10, criterion, optimizer, device)
    # FFNet.train_Net(train_loader, 10, criterion, optimizer, device)
    #
    # # pred = CNet.validate_Net(test_loader, device)
    # pred = FFNet.validate_Net(test_loader, device)
