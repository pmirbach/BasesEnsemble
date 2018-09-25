import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import torchvision
from torchvision import transforms as transforms

from NeuralNetworks import CNNsmall, FFsmall, EnsembleMLP, ResNetLinear
from Transformations import transformation_fourier, normalize_linear
from Datasets import DatasetBase, MyStackedDataset
from TrainValidateTest import Training

from MyLittleHelpers import mkdir

import time, os, copy
import psutil, errno
import pickle

print('PyTorch Version: {}\nTorchvision Version: {}'.format(torch.__version__, torchvision.__version__))

phases = ['train', 'val', 'test']


data_dir = './data/fashion_mnist/'
data_transforms = {'train': transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.5,), std=(0.5,))]),
                   'test': transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize(mean=(0.5,), std=(0.5,))])
                   }

dset_training = torchvision.datasets.FashionMNIST(root=data_dir, train=True,
                                                  transform=data_transforms['train'], download=True)
train_size = int(len(dset_training) * 0.9)
val_size = len(dset_training) - train_size

image_datasets = {}
image_datasets['train'], image_datasets['val'] = torch.utils.data.random_split(dset_training, [train_size, val_size])
image_datasets['test'] = torchvision.datasets.FashionMNIST(root=data_dir, train=False,
                                                           transform=data_transforms['test'], download=True)

dataloaders = {x: DataLoader(image_datasets[x], batch_size=64, shuffle=True) for x in phases}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Net = CNNsmall(inp_shape=image_datasets['train'][0][0].shape)




if __name__ == '__main__2':

    flg_batchsize = 64

    #TODO Take a look at argparser: What is it used for?

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #TODO Where to put all this stuff? Datasets.py?
    # root = './data'
    root = './data/fashion_mnist/'

    # transform = torchvision.transforms.ToTensor()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.5, ), std=(0.5, ))
    ])

    train_set_real = torchvision.datasets.FashionMNIST(root=root, train=True, transform=transform, download=True)
    test_set_real = torchvision.datasets.FashionMNIST(root=root, train=False, transform=transform, download=True)

    # train_set_real = torchvision.datasets.MNIST(root=root, train=True, transform=transform, download=True)
    # test_set_real = torchvision.datasets.MNIST(root=root, train=False, transform=transform, download=True)

    #TODO Load datasets from /raid/common

    path_ft = {'root': root, 'orig_data': 'processed', 'trafo_data': 'fourier', 'trafo_prefix': 'ft'}
    train_set_ft = DatasetBase(name='Fourier', path=path_ft, train=True, base_trafo=transformation_fourier,
                               normalizer=normalize_linear, recalc=False)
    test_set_ft = DatasetBase(name='Fourier', path=path_ft, train=False, base_trafo=transformation_fourier,
                               normalizer=normalize_linear, recalc=False)

    train_set_total = MyStackedDataset([train_set_real, train_set_ft])
    test_set_total = MyStackedDataset([test_set_real, test_set_ft])

    train_loader = DataLoader(dataset=train_set_real, batch_size=flg_batchsize, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_set_real, batch_size=flg_batchsize, shuffle=False, num_workers=0)
    train_loader_ft = DataLoader(dataset=train_set_ft, batch_size=flg_batchsize, shuffle=True, num_workers=10)
    test_loader_ft = DataLoader(dataset=test_set_ft, batch_size=flg_batchsize, shuffle=False, num_workers=10)
    train_loader_total = DataLoader(dataset=train_set_total, batch_size=64, shuffle=True, num_workers=10)
    test_loader_total = DataLoader(dataset=test_set_total, batch_size=64, shuffle=False, num_workers=10)

    Net_real = CNNsmall(train_set_real[0][0].shape)
    Net_ft = FFsmall(train_set_ft[0][0].shape)
    # Net_ft = ResNetLinear(train_set_ft[0][0].shape)

    criterion = nn.CrossEntropyLoss()

    flg_real = 0
    flg_ft = 0
    flg_ens = 0

    optimizer_real = optim.SGD(Net_real.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
    other_params = {'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    'n_epoch': 50, 'inp_split': None}

    train = Training(Net_real, train_loader, test_loader, optimizer_real, criterion, other_params)

    print(psutil.__version__)
    # print(psutil.cpu_percent())
    # print(psutil.virtual_memory())

    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory use:', memoryUse)

    train.training()
    print(train.train_hist)

    if flg_real:
        print(Net_real)
        optimizer_real = optim.SGD(Net_real.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
        Net_real.train_Net(train_loader, 20, criterion, optimizer_real, device)
        pred_real = Net_real.validate_Net(test_loader, device)

    if flg_ft:
        print(Net_ft)
        optimizer_ft = optim.SGD(Net_ft.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
        Net_ft.train_Net(train_loader_ft, 40, criterion, optimizer_ft, device)
        pred_ft = Net_ft.validate_Net(test_loader_ft, device)

    if flg_ens:
        EnsNet = EnsembleMLP([Net_real, Net_ft])
        optimizer_total = optim.SGD(EnsNet.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
        EnsNet.train_Net(train_loader_total, 40, criterion, optimizer_total, device)
        pred_total = EnsNet.validate_Net(test_loader_total, device)


    if False:

        #TODO Make little Helper:
        model_path = r'./results/models'
        mkdir(model_path)

        #TODO Create a NN Save Load module: Models, Trainstatus, ...
        # file_name = 'FF_fourier_trained.pt'
        file_name = 'CNN_real_trained_2.pt'
        torch.save(Net, os.path.join(model_path, file_name))