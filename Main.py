import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from NeuralNetworks import CNNsmall, FFsmall
from Transformations import transformation_fourier, normalize_linear
from Datasets import DatasetBase


import torchvision

import torch.optim as optim

import numpy as np

import time

import os
import pickle
import errno


flg_batchsize = 64


#TODO Refactor: Get training routine somewhere else! (Mabe NeuralNetworks.py?)
def train_Net(Net, train_loader, N_epoch, criterion, optimizer):
    start_time_0 = time.time()
    time_epoch = np.zeros((N_epoch,))

    print('Start Training')
    for epoch in range(N_epoch):
        start_time_epoch = time.time()
        N_seen = 0
        running_loss = 0
        for i, data in enumerate(train_loader, start=0):
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = Net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            N_seen += flg_batchsize

            # running_loss += loss.item()
            # if N_seen >= train_set.__len__() // 5:
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, N_seen, running_loss / N_seen))
            #     N_seen -= train_set.__len__() // 5
            #     running_loss = 0
        time_epoch[epoch] = time.time() - start_time_epoch
        time_estimate = (N_epoch - (epoch + 1)) * np.mean(time_epoch[time_epoch.nonzero()])
        print('Estimated time remaining: {0:5.1f} seconds'.format(time_estimate))

    print('Finished Training - Duration: {0:5.1f} seconds'.format(time.time() - start_time_0))

#TODO Refactor: Get testing routine somewhere else! (Mabe NeuralNetworks.py?)
#TODO Rename to valid(ate)
def test_Net(Net, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data

            images, labels = images.to(device), labels.to(device)

            outputs = Net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 2500 test images: {} %'.format(100 * correct / total))
    return correct / total




if __name__ == '__main__':

    #TODO Take a look at argparser: What is it used for?

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #TODO Where to put all this stuff? Datasets.py?
    root = './data'
    # transform = torchvision.transforms.ToTensor()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.5, ), std=(0.5, ))
    ])

    train_set = torchvision.datasets.MNIST(root=root, train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_set, batch_size=flg_batchsize, shuffle=True, num_workers=10)
    test_set = torchvision.datasets.MNIST(root=root, train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_set, batch_size=flg_batchsize, shuffle=False, num_workers=10)

    path_ft = {'root': 'data', 'orig_data': 'processed', 'trafo_data': 'fourier', 'trafo_prefix': 'ft'}
    train_set_ft = DatasetBase(name='Fourier', path=path_ft, train=True, base_trafo=transformation_fourier,
                               normalizer=normalize_linear, recalc=False)
    test_set_ft = DatasetBase(name='Fourier', path=path_ft, train=False, base_trafo=transformation_fourier,
                               normalizer=normalize_linear, recalc=False)
    train_loader_ft = DataLoader(dataset=train_set_ft, batch_size=flg_batchsize, shuffle=True, num_workers=10)
    test_loader_ft = DataLoader(dataset=test_set_ft, batch_size=flg_batchsize, shuffle=False, num_workers=10)


    # data_shape = train_set[0][0].shape  # Shape of single image, no batch size!
    data_shape = train_set_ft[0][0].shape  # Shape of single image, no batch size! [ch, H, W]




    # Net = CNNsmall(data_shape)
    Net = FFsmall(data_shape)
    Net.to(device)

    print(Net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(Net.parameters(), lr=1e-3, momentum=0.9, nesterov=True)

    # train_Net(Net=Net, train_loader=train_loader, N_epoch=40, criterion=criterion, optimizer=optimizer)
    # test_Net(Net=Net, test_loader=test_loader)

    train_Net(Net=Net, train_loader=train_loader_ft, N_epoch=80, criterion=criterion, optimizer=optimizer)
    test_Net(Net=Net, test_loader=test_loader_ft)

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
    # torch.save(Net, os.path.join(model_path, file_name))