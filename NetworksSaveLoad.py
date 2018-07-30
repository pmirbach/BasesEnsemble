import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from NeuralNetworks import CNNsmall, FFsmall
import torch.optim as optim

import torchvision

import os
import sys, inspect
import shutil
from MyLittleHelpers import prod

CNet = CNNsmall([1, 28, 28])


# FFNet = FFsmall([2,28,28])
#
# # print(CNet)
#
# para_ges = 0
# for child in CNet.children():
#     print(child)
#     if hasattr(child, 'weight'):
#         print(child.weight.size())
#         print(child.weight.dtype)
#         print(prod(child.weight.size()))
#         para_ges += prod(child.weight.size())
#
# print(para_ges)


# class Netti(nn.Module):
#     def __init__(self):
#         super(Netti, self).__init__()
#
#         self.convs = [nn.Conv2d(1, 6, 5), nn.Conv2d(6, 12, 5)]
#
#         self.base = [nn.Linear(120, 50), nn.Linear(50,10)]
#
#
# Net = Netti()
#
# print(Net)
#
#
#
#
# optim.SGD([
#                 {'params': Net.convs.parameters()},
#                 {'params': Net.base.parameters(), 'lr': 1e-3}
#             ], lr=1e-2, momentum=0.9)
#
#
#

class TryNet(nn.Module):
    def __init__(self):
        super(TryNet, self).__init__()
        self.pre = nn.Sequential(nn.Linear(200, 100), nn.Linear(200, 50))
        self.base = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pre(x)
        x = self.base(x)
        return x


TN = TryNet()

special_params = [{'params': TN.pre.parameters()},
                  {'params': TN.base.parameters(), 'lr': 1e-5}
                  ]

# optimizer_real = optim.SGD(CNet.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
# optimizer_real = optim.Adam(CNet.parameters(), lr=1e-3)
optimizer_real = optim.Adam(special_params, lr=1e-3)

print(optimizer_real)

opt_d = optimizer_real.state_dict()
print(opt_d)
print(len(opt_d['param_groups']))


class Training():
    optim_cls = dict(inspect.getmembers(sys.modules['torch.optim'], inspect.isclass))

    def __init__(self, Net=None, dataloader=None, optimizer=None, loss_function=None, other_params=None):
        self.Net = Net
        self.set_optimizer(optimizer)

    def set_dataloader(self):
        print(1)

    def set_optimizer(self, optimizer):

        if type(optimizer).__name__ in self.optim_cls.keys():
            self.optimizer_dict = {'method': type(optimizer).__name__, 'params': None}

        elif isinstance(optimizer, dict):
            self.optimizer_dict = optimizer

    def set_loss_function(self):
        pass

    def set_other_params(self):
        pass

# training_real = Training(net=CNet, optimizer=optimizer_real)

#
# root = './data'
# transform = torchvision.transforms.Compose([
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(mean=(0.5, ), std=(0.5, ))
#     ])
#
#
# train_set_real = torchvision.datasets.FashionMNIST(root=root, train=True, transform=transform, download=True)
# train_loader = DataLoader(dataset=train_set_real, batch_size=10, shuffle=True, num_workers=10)
#
#
#
# training_real = Training(net=CNet, dataloader=train_loader, optimizer=optimizer_real)
#
# print(train_loader.__dict__)
# # print(train_loader.shuffle)
#
# print('\n\n')
#
# loader = getattr(torchvision.datasets, 'MNIST')
# loader_dict = {'dataset': train_set_real, 'batch_size': 10, 'shuffle': True, 'num_workers': 10}
#
# trainer2 = loader(**loader_dict)


#
# def save_checkpoint(state, is_best, save_path, filename):
#   filename = os.path.join(save_path, filename)
#   torch.save(state, filename)
#   if is_best:
#     bestname = os.path.join(save_path, 'model_best.pth.tar')
#     shutil.copyfile(filename, bestname)
#
# torch.save(net.state_dict(),model_save_path + '_.pth')
#
# save_checkpoint({
#           'epoch': epoch + 1,
#           # 'arch': args.arch,
#           'state_dict': net.state_dict(),
#           'optimizer': optimizer.state_dict(),
#         }, is_best, mPath ,  str(val_acc) + '_' + str(val_los) + "_" + str(epoch) + '_checkpoint.pth.tar')
