import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from NeuralNetworks import CNNsmall, FFsmall
import torch.optim as optim

import numpy as np

import torchvision

import os
import sys, inspect
import shutil
import time

from pympler import asizeof

from MyLittleHelpers import prod, sep

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
        self.pre = nn.Sequential(nn.Linear(200, 100), nn.Linear(100, 50), nn.BatchNorm1d(50))
        self.base = nn.Linear(50, 10)
        self.post = nn.Linear(10, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.base(x)
        return self.post(x)



TN = TryNet()

special_params = [{'params': TN.pre.parameters()},
                  {'params': TN.base.parameters(), 'lr': 1e-5},
                  {'params': TN.post.parameters(), 'lr': 1e-7}
                  ]

def fake_training(model, optimizer):
    for _ in range(2):
        optimizer.zero_grad()
        out = model(torch.randn(200))
        out.backward()
        optimizer.step()



def _optimizer_to_ctrl(optimizer, Net):

    def _create_id_name_map(Net):
        id_name_map = {}
        for name, param in Net.named_parameters():
            id_name_map[id(param)] = name
        return id_name_map

    def _id_to_names(id_list, id_name_map):
        name_list = []
        for ida in id_list:
            name_list.append(id_name_map[ida])
        return name_list

    id_name_map = _create_id_name_map(Net)

    param_groups_names = []
    for param in optimizer.state_dict()['param_groups']:
        param_groups_names.append(_id_to_names(param['params'], id_name_map))

    opti_ctrl = {'method': type(optimizer).__name__, 'params': param_groups_names,
                 'state_dict': optimizer.state_dict()}
    return opti_ctrl


def _params_from_ctrl(opti_ctrl, Net):

    def _create_name_param_map(Net):
        name_param_map = {}
        for name, param in Net.named_parameters():
            name_param_map[name] = param
        return name_param_map

    def _gen_name_to_param(name_list, name_param_map):
        for name in name_list:
            yield name_param_map[name]

    name_param_map = _create_name_param_map(Net)
    params = []
    for group_names in opti_ctrl['params']:
        params.append({'params': _gen_name_to_param(group_names, name_param_map)})

    return params

#TODO Get this into another module (?)


class Training():

    # TODO dataloader, loss, other_params, save, memory estimator
    # TODO dataloader
    # TODO loss
    # TODO other_params
    # TODO save
    # TODO memory estimator

    optim_cls = dict(inspect.getmembers(sys.modules['torch.optim'], inspect.isclass))

    def __init__(self, Net=None, dataloader=None, optimizer=None, loss_function=None, other_params=None):
        self.Net = Net

        self.dataloader = dataloader
        self.opti_ctrl = self.set_opti_ctrl(optimizer)
        self.loss_function = loss_function
        self.device = other_params['device']
        self.n_epoch = other_params['n_epoch']
        self.inp_split = other_params['inp_split']

        # self.optimizer = self.init_optimizer()

    def set_dataloader(self):
        pass

    def set_opti_ctrl(self, optimizer):
        if type(optimizer).__name__ in self.optim_cls.keys():
            opti_ctrl = _optimizer_to_ctrl(optimizer, self.Net)
        else:
            raise ValueError('optimizer must be pytorch optimizer!')
        return opti_ctrl

    def init_optimizer(self):
        opti_method = getattr(torch.optim, self.opti_ctrl['method'])
        if self.opti_ctrl['params'] is None:
            optimizer = opti_method(params=self.Net.parameters(), lr=1e-3)
        else:
            params = _params_from_ctrl(self.opti_ctrl, self.Net)
            optimizer = opti_method(params=params, lr=1e-3)
        optimizer.load_state_dict(self.opti_ctrl['state_dict'])
        return optimizer

    def print_opti(self):
        print('Optimizer:')
        dummy_optimizer = self.init_optimizer()
        print(dummy_optimizer)
        for i in range(len(self.opti_ctrl['params'])):
            st = 'Parameter Group {}: {}'.format(i, self.opti_ctrl['params'][i])
            print(st)

    def set_loss_function(self):
        pass

    def set_other_params(self):
        pass

    def estimate_memory(self):
        pass

    def training(self):
        self.Net.to(self.device)
        time_epoch = np.zeros((self.n_epoch,))

        optimizer = self.init_optimizer()

        print('Start Training...')

        for epoch in range(self.n_epoch):

            t0_epoch = time.time()
            epoch_loss = 0

            for i, data in enumerate(train_loader, start=0):

                if self.inp_split is None:
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                else:
                    inputs, labels = self.inp_split(data, self.device)
                # inputs = data[0]
                # inputs = [inp.to(device) for inp in inputs]
                #
                # labels = data[1].to(device)

                optimizer.zero_grad()
                outputs = self.Net(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            time_epoch[epoch] = time.time() - t0_epoch
            time_left_est = (self.n_epoch - (epoch + 1)) * np.mean(time_epoch[time_epoch.nonzero()])

            print("===> Epoch [{:2d}] / {}: Loss: {:.4f} - Approx. {:5.1f} seconds left".format(
                epoch + 1, self.n_epoch, epoch_loss / len(train_loader), time_left_est))

        print('Finished Training - Duration: {0:5.1f} seconds'.format(np.sum(time_epoch)))

# training_real = Training(Net=TN, optimizer=optim.Adam(special_params, lr=1e-3))
#
# training_real.print_opti()



root = './data'
transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.5, ), std=(0.5, ))
    ])


train_set_real = torchvision.datasets.FashionMNIST(root=root, train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_set_real, batch_size=20, shuffle=True, num_workers=0)

Net_real = CNNsmall(train_set_real[0][0].shape)

sep()
# FF = FFsmall(train_set_real[0][0].shape)
# print(asizeof(Net_real))
# print(asizeof.asizeof(FF), {'derive': True})

for p in Net_real.parameters():
    print(p.storage().__sizeof__())
sep()

optimizer_real = optim.SGD(Net_real.parameters(), lr=1e-3, momentum=0.9, nesterov=True)

criterion = nn.CrossEntropyLoss()

other_params = {'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                'n_epoch': 20, 'inp_split': None}

training = Training(Net_real, train_loader, optimizer_real, criterion, other_params)
# training.training()


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
