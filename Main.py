import torch
import torch.nn as nn
import torch.nn.functional as F

from NeuralNetworks import CNN_small

import torchvision

import torch.optim as optim

import numpy as np

import time

import os
import pickle



def train_Net(Net, N_epoch, criterion, optimizer):
    start_time_0 = time.time()
    time_epoch = np.zeros((N_epoch,))

    print('Start Training')
    for epoch in range(N_epoch):
        start_time_epoch = time.time()
        running_loss = 0
        for i, data in enumerate(train_loader, start=0):
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = Net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 2500 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2500))
                running_loss = 0
        time_epoch[epoch] = time.time() - start_time_epoch
        time_estimate = (N_epoch - (epoch + 1)) * np.mean(time_epoch[time_epoch.nonzero()])
        print('Estimated time remaining: {0:5.1f} seconds'.format(time_estimate))

    print('Finished Training - Duration: {0:5.1f} seconds'.format(time.time() - start_time_0))


def test_Net(Net):
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



class CNN(nn.Module):

    def __init__(self, inp_shape):
        super(CNN, self).__init__()

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






if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    root = './data'
    # transform = torchvision.transforms.ToTensor()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.5, ), std=(0.5, ))
    ])

    train_set = torchvision.datasets.MNIST(root=root, train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=256, shuffle=True, num_workers=10)
    test_set = torchvision.datasets.MNIST(root=root, train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=256, shuffle=False, num_workers=10)

    data_shape = train_set[0][0].shape  # Shape of single image, no batch size!

    # Net = CNN(data_shape)
    Net = CNN_small(data_shape)
    Net.to(device)

    print(Net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(Net.parameters(), lr=1e-3, momentum=0.9, nesterov=True)

    train_Net(Net=Net, N_epoch=5, criterion=criterion, optimizer=optimizer)
    test_Net(Net=Net)
