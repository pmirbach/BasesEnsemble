import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from NeuralNetworks import CNNsmall, FFsmall, EnsembleMLP
from Transformations import transformation_fourier, normalize_linear
from Datasets import DatasetBase, MyStackedDataset


import torchvision

import torch.optim as optim

import numpy as np

import time

import os
import pickle
import errno








if __name__ == '__main__':

    flg_batchsize = 64

    #TODO Take a look at argparser: What is it used for?

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #TODO Where to put all this stuff? Datasets.py?
    root = './data'
    # transform = torchvision.transforms.ToTensor()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.5, ), std=(0.5, ))
    ])

    train_set_real = torchvision.datasets.MNIST(root=root, train=True, transform=transform, download=True)
    test_set_real = torchvision.datasets.MNIST(root=root, train=False, transform=transform, download=True)


    path_ft = {'root': 'data', 'orig_data': 'processed', 'trafo_data': 'fourier', 'trafo_prefix': 'ft'}
    train_set_ft = DatasetBase(name='Fourier', path=path_ft, train=True, base_trafo=transformation_fourier,
                               normalizer=normalize_linear, recalc=False)
    test_set_ft = DatasetBase(name='Fourier', path=path_ft, train=False, base_trafo=transformation_fourier,
                               normalizer=normalize_linear, recalc=False)

    train_set_total = MyStackedDataset([train_set_real, train_set_ft])
    test_set_total = MyStackedDataset([test_set_real, test_set_ft])

    train_loader = DataLoader(dataset=train_set_real, batch_size=flg_batchsize, shuffle=True, num_workers=10)
    test_loader = DataLoader(dataset=test_set_real, batch_size=flg_batchsize, shuffle=False, num_workers=10)
    train_loader_ft = DataLoader(dataset=train_set_ft, batch_size=flg_batchsize, shuffle=True, num_workers=10)
    test_loader_ft = DataLoader(dataset=test_set_ft, batch_size=flg_batchsize, shuffle=False, num_workers=10)
    train_loader_total = DataLoader(dataset=train_set_total, batch_size=64, shuffle=True, num_workers=10)
    test_loader_total = DataLoader(dataset=test_set_total, batch_size=64, shuffle=False, num_workers=10)

    Net_real = CNNsmall(train_set_real[0][0].shape)
    Net_ft = FFsmall(train_set_ft[0][0].shape)

    criterion = nn.CrossEntropyLoss()

    optimizer_real = optim.SGD(Net_real.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
    optimizer_ft = optim.SGD(Net_ft.parameters(), lr=1e-3, momentum=0.9, nesterov=True)

    Net_real.train_Net(train_loader, 2, criterion, optimizer_real, device)
    Net_ft.train_Net(train_loader_ft, 2, criterion, optimizer_ft, device)

    pred_real = Net_real.validate_Net(test_loader, device)
    pred_ft = Net_ft.validate_Net(test_loader_ft, device)


    EnsNet = EnsembleMLP([Net_real, Net_ft])
    optimizer_total = optim.SGD(EnsNet.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
    EnsNet.train_Net(train_loader_total, 2, criterion, optimizer_total, device)
    pred_total = EnsNet.validate_Net(test_loader_total, device)


    if False:

        #TODO Make little Helper:
        model_path = r'./results/models'
        try:
            os.makedirs(model_path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        #TODO Create a NN Save Load module: Models, Trainstatus, ...
        # file_name = 'FF_fourier_trained.pt'
        file_name = 'CNN_real_trained_2.pt'
        torch.save(Net, os.path.join(model_path, file_name))