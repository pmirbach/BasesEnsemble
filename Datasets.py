"""
The module contains all Datasets I use. All functions create a Dataset object for Pytorch's Dataloader. All transformed
datasets are in the same order as the original dataset.
"""
import torch
import os
import time
import errno
from torch.utils.data import Dataset
from MyLittleHelpers import sep
from Transformations import transformation_fourier, normalize_linear
from torchvision import datasets, transforms
import numpy as np

# from matplotlib import pyplot as plt


def get_dataset(dataset, data_dir, train_transform, test_transform, flg_stats=False):

    dsets = {}
    if dataset == 'cifar10':
        dsets['training'] = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=train_transform)
        dsets['test'] = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=test_transform)
    elif dataset == 'cifar100':
        dsets['training'] = datasets.CIFAR100(root=data_dir, train=True, download=False, transform=train_transform)
        dsets['test'] = datasets.CIFAR100(root=data_dir, train=False, download=False, transform=test_transform)
    elif dataset == 'mnist':
        dsets['training'] = datasets.MNIST(root=data_dir, train=True, download=False, transform=train_transform)
        dsets['test'] = datasets.MNIST(root=data_dir, train=False, download=False, transform=test_transform)
    elif dataset == 'fashion-mnist':
        dsets['training'] = datasets.FashionMNIST(root=data_dir, train=True, download=False, transform=train_transform)
        dsets['test'] = datasets.FashionMNIST(root=data_dir, train=False, download=False, transform=test_transform)

    if flg_stats:
        get_dset_stats(dsets, False)
    return dsets


def get_transforms(dataset):
    stats = get_normalize_stats(dataset)
    train_mean, train_std, test_mean, test_std = stats
    return {'train': transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=train_mean, std=train_std)]),
            'test': transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=test_mean, std=test_std)])
            }

def get_dset_stats(dsets, raw=True):

    num_ch = len(dsets['training'].train_data.shape)

    if raw:
        train_data = dsets['training'].train_data
        test_data = dsets['test'].test_data
        if num_ch > 3:
            train_data = np.transpose(train_data, (0, 3, 1, 2))
            test_data = np.transpose(test_data, (0, 3, 1, 2))
    else:
        train_shape = dsets['training'].train_data.shape
        test_shape = dsets['test'].test_data.shape
        if num_ch > 3:
            train_shape = [train_shape[i] for i in [0,3,1,2]]
            test_shape = [test_shape[i] for i in [0,3,1,2]]

        train_data = np.zeros(train_shape)
        test_data = np.zeros(test_shape)

        for i in range(len(dsets['training'])):
            train_data[i,:,:,:] = dsets['training'][i][0].numpy()
        for i in range(len(dsets['test'])):
            test_data[i,:,:,:] = dsets['test'][i][0].numpy()

    out_str = '{}: shape: {}, mean: {}, std: {}'
    for phase, data in zip(['training', 'test'], [train_data, test_data]):
        if torch.is_tensor(data):
            data = data.numpy()
        if num_ch > 3:
            mean = np.mean(data, axis=(0,2,3)) / 255
            std = np.std(data, axis=(0,2,3)) / 255
        else:
            mean = np.mean(data, axis=(0,1,2)) / 255
            std = np.std(data, axis=(0,1,2)) / 255
        print(out_str.format(phase, data.shape, tuple(mean), tuple(std)))



def get_normalize_stats(dataset):
    if dataset == 'mnist':
        out = (0.1306604762738429,), (0.30810780385646264,), (0.13251460584233693,), (0.3104802479305351,)
    elif dataset == 'fashion-mnist':
        out = (0.2860405969887955,), (0.3530242445149223,), (0.28684928071228494,), (0.35244415324743994,)
    elif dataset == 'cifar10':
        out = (0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784), \
              (0.49421428, 0.48513139, 0.45040909), (0.24665252, 0.24289226, 0.26159238)
    elif dataset == 'cifar100':
        out = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), \
              (0.26733428587924035, 0.2564384629170881, 0.2761504713256853), \
              (0.5087964127604166, 0.48739301317401956, 0.44194221124387256), \
              (0.2682515741720806, 0.2573637364478126, 0.2770957707973048)
    return out

class DatasetBase(Dataset):

    def __init__(self, name, path, train=True, base_trafo=None, normalizer=None, recalc=False):
        self.name = name

        file_name = 'training.pt' if train else 'test.pt'
        self.orig_file = os.path.join(path['root'], path['orig_data'], file_name)
        self.trafo_path = os.path.join(path['root'], path['trafo_data'])
        self.trafo_file = os.path.join(path['root'], path['trafo_data'], path['trafo_prefix'] + '_' + file_name)

        if recalc:
            os.remove(self.trafo_file)

        if base_trafo is not None:
            self.transform_data(transformation=base_trafo, normalization=normalizer)

        if not self._check_exists():
            raise RuntimeError('Dataset not found in given path.' +
                               ' You can use base_trafo=Trafo to create it.')

        self.data, self.labels = torch.load(self.trafo_file)

    def transform_data(self, transformation, normalization):
        if self._check_exists():
            return

        if not self._check_exists(trafo=False):
            raise RuntimeError('Original data can not be found. Provide original data to transform.')

        try:
            os.makedirs(self.trafo_path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        data, labels = torch.load(self.orig_file)
        if len(data.size()) < 4:
            data = data.unsqueeze(1)

        print('Transformation of original data...')
        t0 = time.time()
        data_trafo = transformation(data)
        print('Transformation completed in {:5.1f} seconds.'.format(time.time() - t0))

        if normalization is not None:
            print('Normalization of transformed data...')
            t0 = time.time()
            for i in range(data_trafo.size()[1]):  # Normalize every channel seperatly
                data_trafo[:, i, :, :] = normalization(data_trafo[:, i, :, :])
            print('Normalization completed in {:5.1f} seconds.'.format(time.time() - t0))

        with open(self.trafo_file, 'wb') as f:
            torch.save((data_trafo, labels), f)

    def _check_exists(self, trafo=True):
        if trafo:
            return os.path.exists(self.trafo_file)
        else:
            return os.path.exists(self.orig_file)

    def __len__(self):
        return self.data.size()[0]

    def __getitem__(self, idx):
        return self.data[idx, :, :, :], self.labels[idx]


class MyStackedDataset(Dataset):
    def __init__(self, datasets_list, train=True):
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


if __name__ == '__main__':
    sep()
    path1 = {'root': 'data', 'orig_data': 'processed', 'trafo_data': 'fourier', 'trafo_prefix': 'ft'}
    set1 = DatasetBase(name='Fourier', path=path1, train=True,
                       base_trafo=transformation_fourier, normalizer=normalize_linear, recalc=False)
    # set1 = DatasetBase(name='Fourier', path=path1, train=True,
    #                    base_trafo=transformation_fourier)
    loader1 = torch.utils.data.DataLoader(dataset=set1, batch_size=1, shuffle=False, num_workers=0)

    it = iter(loader1)
    data, label = it.next()
    data = data.numpy()

    # fig, axes = plt.subplots(2, 2)
    #
    # axes[0, 0].imshow(data[0, 0, :, :], cmap=plt.get_cmap('gray'))
    # axes[1, 0].hist(data[0, 0, :, :].flatten(), bins=100)
    # axes[0, 1].imshow(data[0, 1, :, :], cmap=plt.get_cmap('gray'))
    # axes[1, 1].hist(data[0, 1, :, :].flatten(), bins=100)

    sep()
