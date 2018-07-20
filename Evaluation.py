import torch
from torch.utils.data import DataLoader

import numpy as np

from Transformations import transformation_fourier, normalize_linear
from Datasets import DatasetBase

import os

#TODO Make evaluation


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model_path = r'./results/models'
file_name = 'FF_fourier_trained.pt'


Net = torch.load(os.path.join(model_path, file_name), map_location=lambda storage, loc: storage)



def test_Net(Net, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        Net.eval()
        for data in test_loader:
            images, labels = data
            images = images.float()

            images, labels = images.to(device), labels.to(device)

            outputs = Net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 2500 test images: {} %'.format(100 * correct / total))
    return correct / total





path_ft = {'root': 'data', 'orig_data': 'processed', 'trafo_data': 'fourier', 'trafo_prefix': 'ft'}
test_set_ft = DatasetBase(name='Fourier', path=path_ft, train=False, base_trafo=transformation_fourier,
                           normalizer=normalize_linear, recalc=False)
test_loader_ft = DataLoader(dataset=test_set_ft, batch_size=64, shuffle=False, num_workers=0)





test_Net(Net=Net, test_loader=test_loader_ft)


