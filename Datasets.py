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
# from matplotlib import pyplot as plt


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
            print('File vorhanden.')
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