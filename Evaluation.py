import torch
from torch.utils.data import DataLoader, Dataset
import torchvision

import numpy as np

from Transformations import transformation_fourier, normalize_linear
from Datasets import DatasetBase

import os

from NeuralNetworks import EnsembleMLP, FFsmall, CNNsmall

# import matplotlib.pyplot as plt

# TODO Make evaluation


class MyStackedDataset(Dataset):
    def __init__(self, datasets_list):
        self.datasets_list = datasets_list
        self.len = datasets_list[0].__len__()

        # self.data_list = [dataset.data for dataset in datasets_list]
        # self.labels = datasets_list[0].labels

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data_total = []
        for dataset in self.datasets_list:
            data, label = dataset.__getitem__(idx)
            data_total.append(data)
        return data_total, label









device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = r'./results/models'
file_name_ft = 'FF_fourier_trained.pt'
file_name_real = 'CNN_real_trained_2.pt'

Net_ft = torch.load(os.path.join(model_path, file_name_ft), map_location=lambda storage, loc: storage)
Net_real = torch.load(os.path.join(model_path, file_name_real), map_location=lambda storage, loc: storage)


def test_Net(Net, test_loader):
    res = np.zeros(n_test)

    correct = 0
    total = 0
    with torch.no_grad():
        Net.eval()
        for i, data in enumerate(test_loader):
            images, labels = data
            images = images.float()

            images, labels = images.to(device), labels.to(device)

            outputs = Net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            try:
                res[i * 64: (i + 1) * 64] = predicted == labels
            except:
                res[i * 64:] = predicted == labels

            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10.000 test images: {} %'.format(100 * correct / total))
    return np.array(res, dtype=bool)





path_ft = {'root': 'data', 'orig_data': 'processed', 'trafo_data': 'fourier', 'trafo_prefix': 'ft'}
train_set_ft = DatasetBase(name='Fourier', path=path_ft, train=True, base_trafo=transformation_fourier,
                               normalizer=normalize_linear, recalc=False)
test_set_ft = DatasetBase(name='Fourier', path=path_ft, train=False, base_trafo=transformation_fourier,
                          normalizer=normalize_linear, recalc=False)
test_loader_ft = DataLoader(dataset=test_set_ft, batch_size=64, shuffle=False, num_workers=0)


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,))
])

train_set_real = torchvision.datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
test_set_real = torchvision.datasets.MNIST(root='./data/', train=False, transform=transform, download=True)
test_loader_real = DataLoader(dataset=test_set_real, batch_size=64, shuffle=False, num_workers=0)


train_set_total = MyStackedDataset([train_set_real, train_set_ft])
test_set_total = MyStackedDataset([test_set_real, test_set_ft])

train_loader_total = DataLoader(dataset=train_set_total, batch_size=64, shuffle=True, num_workers=0)
test_loader_total = DataLoader(dataset=test_set_total, batch_size=64, shuffle=False, num_workers=0)

# n_test = test_set_ft[:][0].shape[0]
n_test = test_set_real.__len__()
# print(n_test)
#
# res_ft = test_Net(Net=Net_ft, test_loader=test_loader_ft)
# res_real = test_Net(Net=Net_real, test_loader=test_loader_real)
#
# res_total = np.logical_or(res_ft, res_real)
#
# print('Accuracy of the network on the 10.000 test images: {} %'.format(100 * np.sum(res_total) / n_test))


# xt = torch.randn(1,1,28,28)
# yt = Net_real.forward_part(xt)
# print(yt.size())


Net_total = EnsembleMLP(Net_real=Net_real, Net_ft=Net_ft)
par = list(Net_total.parameters())
for p in par:
    print(p.size())


# it = iter(train_loader_total)
# data, label = it.next()
# data_real = data[0]
# data_ft = data[1]
#
# print(Net_total(data))











