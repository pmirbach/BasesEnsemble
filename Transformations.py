import numpy as np
import torch
from torch.utils.data import Dataset
# from matplotlib import pyplot as plt



class MyDataset(Dataset):

    def __init__(self, data, transformation):
        self.len = data.__len__()
        self.data, self.label = transformation(data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (self.data[idx, :, :], self.label[idx])


def change_range(x, range, data_range=None):
    """
    :param x: real data of any shape
    :param range: list or tupel with new range
    :param data_range: list of min and max of data
    :return: data in same format transformed to range
    """
    if data_range is not None:
        xmin, xmax = data_range[0], data_range[1]
    else:
        xmin, xmax = np.min(x), np.max(x)

    x = (range[1] - range[0]) * (x - xmin) / (xmax - xmin) + range[0]
    return x


def trafo(data):
    len = [data.__len__(), ]
    data_shape = len + list(data.__getitem__(0)[0].size())
    data_shape[1] *= 2

    data_trafo = np.zeros(data_shape)
    label_trafo = torch.zeros(data.__len__(), dtype=torch.int64)

    for i in range(data.__len__()):
    # for i in range(11):
        img, label = data.__getitem__(i)
        img_np = img.numpy()
        img_ft_np = np.fft.fft2(img_np)

        data_trafo[i, 0, :, :] = np.real(img_ft_np)
        data_trafo[i, 1, :, :] = np.imag(img_ft_np)

        label_trafo[i] = label

    real_min, real_max = np.min(data_trafo[:, 0, :, :]), np.max(data_trafo[:, 0, :, :])
    imag_min, imag_max = np.min(data_trafo[:, 1, :, :]), np.max(data_trafo[:, 1, :, :])

    data_trafo[:, 0, :, :] = change_range(data_trafo[:, 0, :, :], [-1, 1], [real_min, real_max])
    data_trafo[:, 1, :, :] = change_range(data_trafo[:, 1, :, :], [-1, 1], [imag_min, imag_max])

    return (torch.from_numpy(data_trafo).float(), label_trafo)



def tensor_check():
    pass


def dataset_fourier(dataset):

    dataset_ft = MyDataset(dataset, trafo)

    return dataset_ft

    # for i in range(10):
    #     print('--{}--'.format(i))
    #     img_real, label_real = dataset.__getitem__(i)
    #     img_real_ft, label_real_ft = dataset_ft.__getitem__(i)
    #
    #     print(label_real, label_real_ft)

    # tset = trafo(dataset)

    # print(1)
