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


class DatasetBase(Dataset):

    def __init__(self, name, path, train=True, base_trafo=None, normalizer=None):
        self.name = name

        file_name = 'training.pt' if train else 'test.pt'
        self.orig_file = os.path.join(path['root'], path['orig_data'], file_name)
        self.trafo_path = os.path.join(path['root'], path['trafo_data'])
        self.trafo_file = os.path.join(path['root'], path['trafo_data'], path['trafo_prefix'] + '_' + file_name)

        if base_trafo is not None:
            self.transform_data(transformation=base_trafo)

    def transform_data(self, transformation):
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
        print('Transforming data...\n')
        t0 = time.time()
        data_trafo = transformation(data)
        print('Transformation completed in {:5.1f} seconds.'.format(time.time() - t0))



    def _check_exists(self, trafo=True):
        if trafo:
            return os.path.exists(self.trafo_file)
        else:
            return os.path.exists(self.orig_file)


def test_trafo(data):
    time.sleep(2)
    return 1


if __name__ == '__main__':

    sep()
    path1 = {'root': 'data', 'orig_data': 'processed', 'trafo_data': 'fourier', 'trafo_prefix': 'ft'}
    set1 = DatasetBase(name='Fourier', path=path1, train=True, base_trafo=test_trafo)


    sep()