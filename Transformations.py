import numpy as np
import torch
from functools import wraps
from torch.utils.data import Dataset
from scipy.stats import boxcox


# from matplotlib import pyplot as plt


def pre_trafo(data):
    """
    :param data: Arbitrary data in numpy or python
    :return: Data in numpy for transformations
    """
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    elif isinstance(data, np.ndarray):
        pass
    else:
        raise Exception('Transformation input data must be torch.Tensor or numpy.ndarray.')
    return data


def post_trafo(data):
    """
    :param data: Arbitrary data in numpy or python
    :return: Data in pytorch for datasets.
    """
    if isinstance(data, torch.Tensor):
        pass
    elif isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    else:
        raise Exception('Transformation output data must be torch.Tensor or numpy.ndarray.')
    return data


def tensor_ndarray_handler(func):
    """
    Decorator for all transformations to handle numpy < - > pytorch
    :param func:
    :return:
    """

    @wraps(func)
    def function_wrapper(data, *args, **kwargs):
        data = pre_trafo(data)
        data = func(data, *args, **kwargs)
        data = post_trafo(data)
        return data

    return function_wrapper


@tensor_ndarray_handler
def transformation_fourier(data):
    """
    Fourier transformation of data along axis 0. Real and complex part saved in different channels.
    :param data: 2d numpy or torch dataset: [N, channel, Height, Width]
    :return: Fourier transform of data along axis 0. [N, channel*2 (real/imag), Height, Width]
    """
    dshape = list(data.shape)
    nCh = dshape[1]
    f_ind = 1
    for iCh in range(nCh):
        data_ft = np.fft.fft2(data[:, iCh, :, :])
        data_ft = np.real_if_close(data_ft, tol=1e6)

        for iTest in np.random.choice(range(dshape[0]), size=5):
            data_ft_test = np.fft.fft2(data[iTest, iCh, :, :])
            if not np.allclose(data_ft[iTest, :, :], data_ft_test):
                raise Exception('Vectorized transformation results vary from non vectorized')

        # Allocate memory for transformed data:
        if iCh == 0:
            if np.any(np.iscomplex(data_ft)):
                dshape[1] *= 2
                f_ind = 2
            data_out = np.zeros(dshape)

        data_out[:, iCh * f_ind, :, :] = np.real(data_ft)
        if f_ind == 2:
            data_out[:, iCh * f_ind + 1, :, :] = np.imag(data_ft)

    return data_out


# TODO Implement Laplace transformation
def transformation_laplace():
    pass


# TODO Implement Hough transformation
def transformation_hough():
    pass


@tensor_ndarray_handler
def normalize_box_cox(data):
    """
    Uses Box-Cox transformation to make data more normal. Use only for normaly distributed data.
    :param data: Numpy or pytorch data of any shape
    :return: pytorch data of input shape
    """
    data_shape = data.shape
    data = data.reshape(-1, 1)

    box_cox_shift = np.min([np.min(data), 0]) - 1e-3
    data -= box_cox_shift

    data, _ = boxcox(data)
    data += box_cox_shift

    return data.reshape(data_shape)


@tensor_ndarray_handler
def normalize_linear(data, range=[-1, 1], data_range=None):
    """
    Transforms data of arbitrary shape linear into a given range.
    :param data: real data of any shape
    :param range: list or tupel with new range
    :param data_range: list of min and max of data
    :return: data in same format transformed to range
    """
    if data_range is not None:
        xmin, xmax = data_range[0], data_range[1]
    else:
        xmin, xmax = np.min(data), np.max(data)

    data = (range[1] - range[0]) * (data - xmin) / (xmax - xmin) + range[0]
    return data


if __name__ == '__main__':
    x = torch.randn(100, 1, 28, 28)
    y = transformation_fourier(x)

    # print(y)
    # x = torch.randn(5,5)
    # y = np.ndarray([1,2,3])
    # pre_trafo(y)
