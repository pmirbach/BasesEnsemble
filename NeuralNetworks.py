import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import DataLoader

import torchvision

import numpy as np

from MyLittleHelpers import prod, sep
import time

from torchvision import models

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
        if layer.bias is not None:
            layer.bias.requires_grad = False


def inp_split_ensemble(data, device):
    inputs = data[0]
    inputs = [inp.to(device) for inp in inputs]
    labels = data[1].to(device)
    return inputs, labels


class ResBlockLinear(nn.Module):

    def __init__(self, n_features, n_res_layers=2):
        super(ResBlockLinear, self).__init__()

        a = []
        for i in range(n_res_layers):
            # a.append(nn.Dropout(p=0.2))
            a.append(nn.Linear(in_features=n_features, out_features=n_features, bias=False))
            a.append(nn.BatchNorm1d(n_features))
            a.append(nn.ReLU(inplace=False))

        self.left = nn.Sequential(*a)
        self.right = nn.Sequential()

    def forward(self, x):
        out = self.left(x)
        residual = x
        out += residual
        return out


class ResNetLinear(nn.Module):

    def __init__(self, inp_shape):
        super(ResNetLinear, self).__init__()
        self.num_inp_channels, self.data_shape = _get_data_shape(inp_shape)
        self.num_flat_features = _get_get_num_flat_features(self.num_inp_channels, self.data_shape)

        # n = round(self.num_flat_features * 1.3)
        n = round(self.num_flat_features * 2.0)
        m = 80

        self.pre = nn.Sequential(
            nn.Linear(in_features=self.num_flat_features, out_features=n),
            nn.BatchNorm1d(n))

        a = []
        for i in range(m):
            a.append(ResBlockLinear(n_features=n))

        self.res = nn.Sequential(*a)

        self.post = nn.Linear(in_features=n, out_features=10)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features)
        x = F.relu(self.pre(x))
        x = self.res(x)
        x = self.post(x)
        return x


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
        self.conv1 = nn.Conv2d(in_channels=self.num_inp_channels, out_channels=10, kernel_size=5, bias=False)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, bias=False)

        self.num_flat_features = _get_get_num_flat_features(self.num_inp_channels, self.data_shape,
                                                            self.forward_preprocess)

        self.fc1 = nn.Linear(in_features=self.num_flat_features, out_features=128, bias=False)
        self.fc2 = nn.Linear(in_features=128, out_features=64, bias=False)
        self.fc3 = nn.Linear(in_features=64, out_features=10, bias=False)

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


class FFsmall(nn.Module):

    def __init__(self, inp_shape):
        super(FFsmall, self).__init__()
        self.num_inp_channels, self.data_shape = _get_data_shape(inp_shape)
        self.num_flat_features = _get_get_num_flat_features(self.num_inp_channels, self.data_shape)

        self.dropout = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(in_features=self.num_flat_features, out_features=10000, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=10000)

        self.fc2 = nn.Linear(in_features=10000, out_features=10000, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=10000)

        self.fc3 = nn.Linear(in_features=10000, out_features=10000, bias=False)
        self.bn3 = nn.BatchNorm1d(num_features=10000)

        self.fc4 = nn.Linear(in_features=10000, out_features=10000, bias=False)
        self.bn4 = nn.BatchNorm1d(num_features=10000)

        self.fc5 = nn.Linear(in_features=10000, out_features=10, bias=False)
        # self.bn5 = nn.BatchNorm1d(num_features=4000)

        # self.fc6 = nn.Linear(in_features=4000, out_features=10, bias=False)
        # self.bn6 = nn.BatchNorm1d(num_features=512)

        # self.fc7 = nn.Linear(in_features=1024, out_features=1024, bias=False)
        # self.bn7 = nn.BatchNorm1d(num_features=1024)
        #
        # self.fc8 = nn.Linear(in_features=1024, out_features=1024, bias=False)
        # # self.bn8 = nn.BatchNorm1d(num_features=512)
        #
        # self.fc9 = nn.Linear(in_features=1024, out_features=512, bias=False)
        # self.bn9 = nn.BatchNorm1d(num_features=512)
        #
        # self.fc10 = nn.Linear(in_features=512, out_features=10, bias=False)

        self.num_out_features = self.fc4.out_features

        # TODO Make this a ResNet (?)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features)
        x = self.dropout(self.bn1(F.relu(self.fc1(x))))
        x = self.dropout(self.bn2(F.relu(self.fc2(x))))
        # x = F.relu(self.fc2(x))
        x = self.dropout(self.bn3(F.relu(self.fc3(x))))
        x = self.dropout(self.bn4(F.relu(self.fc4(x))))
        return self.fc5(x)

        # x = F.relu(self.fc4(x))
        # x = self.dropout(self.bn5(F.relu(self.fc5(x))))
        # # x = self.dropout(self.bn6(F.relu(self.fc6(x))))
        # x = F.relu(self.fc6(x))
        # x = self.dropout(self.bn7(F.relu(self.fc7(x))))
        # # x = self.dropout(self.bn8(F.relu(self.fc8(x))))
        # x = F.relu(self.fc8(x))
        # x = self.bn9(F.relu(self.fc9(x)))
        #
        # return self.fc10(x)

    def forward_part(self, x):
        x = x.view(-1, self.num_flat_features)
        x = self.dropout(self.bn1(F.relu(self.fc1(x))))
        x = self.dropout(self.bn2(F.relu(self.fc2(x))))
        x = F.relu(self.fc3(x))
        return F.relu(self.fc4(x))


class TryNet(nn.Module):
    def __init__(self):
        pass


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



def make_conv_layer(cfg, in_channels, batch_norm=True):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1)]
            if batch_norm:
                layers += [nn.BatchNorm2d(num_features=v)]
            layers += [nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers), in_channels

def make_linear_layer(cfg):
    pass


cfg_conv = ['64', '', '']
cfg_linear = ['', '', '']



class Layers(nn.Module):

    def __init__(self):
        pass


if __name__ == '__main__':


    def printer(Net):
        print('\nNamed children: ')
        for name, paras in Net.named_children():
            print(name, type(paras))
        print('\nNamed modules: ')
        for name, paras, in Net.named_modules():
            print(name, type(paras))
        print('\nNamed parameters: ')
        for name, paras in Net.named_parameters():
            print(name, type(paras), paras.size())

    def s_1(Net):
        for name, layer in Net.named_children():
            print(name)
            print(layer)

            if isinstance(layer, nn.Sequential):
                print(1213)

            # for name, para in layer.named_parameters():
            #     print(name, para.size())

    vgg11 = models.vgg11_bn(pretrained=False, num_classes=100)
    # print(vgg11)
    # printer(vgg11)
    s_1(vgg11)

    # x_real = torch.randn(4, 1, 28, 28)
    # CNet = CNNsmall(x_real.size())
    # print(CNet)
    #
    #
    # model = CNet
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = [np.prod(p.size()) for p in model_parameters]
    #
    # print(params)

    # x_real = torch.randn(4, 1, 28, 28)
    # x_ft = torch.randn(4, 2, 28, 28)
    #
    # # CNet = CNNsmall(x_real.size())
    # FFNet = FFsmall(x_ft.size())
    # # EnsNet = EnsembleMLP([CNet, FFNet])
    #
    # ResNetLinear = ResNetLinear(x_ft.size())
    #
    #
    # # for child in CNet.children():
    # #     print(type(child.bias()))
    # #     print(child)
    #
    # def inspec_Net(Net):
    #     print(Net)
    #     print()
    #     for child in Net.children():
    #
    #         print(child)
    #
    #         if hasattr(child, 'weight'):
    #             print('has weight')
    #             print(child.weight.requires_grad)
    #         if hasattr(child, 'bias'):
    #             if child.bias is not None:
    #                 print('has bias')
    #                 print(child.bias.requires_grad)
    #
    #         for param in child.parameters():
    #             print(param.size())
    #         print()
    #
    #
    # ResNetLinear(x_ft)
    # FFNet(x_ft)
    #
    # # inspec_Net(ResNetLinear)
    #
    # # for child in FFNet.children():
    # #     set_grad_false(child)
    #
    # # inspec_Net(FFNet)
    #
    # # for param in EnsNet.parameters():
    # #     print(param.size())
    #
    # # # sep()
    # # # print(EnsNet.parameters())
    # # # sep()
    # # # CNet.train_Net(x_real)
    # #
    # # # _x_real = CNet.forward(x_real)
    # # # _x_ft = FFNet.forward(x_ft)
    # # # _x_ens = EnsNet([x_real, x_ft])
    # #
    # # transform = torchvision.transforms.Compose([
    # #     torchvision.transforms.ToTensor(),
    # #     torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,))
    # # ])
    # #
    # # train_set = torchvision.datasets.MNIST(root="./data/", train=True, transform=transform, download=True)
    # # train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True, num_workers=10)
    # # test_set = torchvision.datasets.MNIST(root="./data/", train=False, transform=transform, download=True)
    # # test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=False, num_workers=10)
    # #
    # # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # #
    # # criterion = nn.CrossEntropyLoss()
    # # optimizer = optim.SGD(CNet.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
    # # optimizer = optim.SGD(FFNet.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
    # #
    # #
    # # # CNet.train_Net(train_loader, 10, criterion, optimizer, device)
    # # FFNet.train_Net(train_loader, 10, criterion, optimizer, device)
    # #
    # # # pred = CNet.validate_Net(test_loader, device)
    # # pred = FFNet.validate_Net(test_loader, device)
