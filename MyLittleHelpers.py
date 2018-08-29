import os
import errno
import numpy as np
import time

def prod(x_list):
    """
    Calculates the product of all elements in an list.
    :param x: list or tupel of scalars
    :return: Product of all scalars
    """
    y = 1
    for x in x_list:
        y *= x
    return y


def sep():
    print('\n{}\n'.format('-'*70))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


def display_time(seconds):

    intervals = (
        ('d', 86400),  # 60 * 60 * 24
        ('h', 3600),  # 60 * 60
        ('m', 60),
        ('s', 1),
    )

    disp_out = {'d': '{d}-{h:02d}:{m:02d}:{s:02d}',
            'h': '{h:d}:{m:02d}:{s:02d}',
            'm': '{m:d}:{s:02d}',
            's': '0:{s:02d}'}

    result = {}
    largest_unit = None

    for name, count in intervals:
        value = seconds // count

        if value and largest_unit is None:
            largest_unit = name

        seconds -= value * count
        result[name] = value

    return disp_out[largest_unit].format(**result)


class LossAccStats():
    """
    Class to keep track of training/validation loss and accuracy.
    """
    def __init__(self, datasize=None):
        self.reset()
        self.datasize = datasize

    def reset(self):
        self.running_loss = 0.0
        self.running_corrects, self.running_seen = 0, 0

    def update(self, batch_loss, batch_acc, n_batch=None):
        self.running_corrects += np.sum(batch_acc)
        if n_batch is None:
            n_batch = len(batch_acc)
        self.running_loss += batch_loss * n_batch
        self.running_seen += n_batch

    def get_stats(self):
        if self.datasize is not None:
            if self.datasize != self.running_seen:
                raise ValueError('Datasize and sum of batch sizes do not match!')
        N = self.running_seen
        return self.running_loss / N, self.running_corrects / N


class Timer():

    def __init__(self):
        self.step_times = []
        self.start()

    def start(self):
        self.start_time = self.last_timestamp = time.time()

    def step(self):
        self.step_times.append(time.time() - self.last_timestamp)
        self.last_timestamp = time.time()

    def get_avg_time(self, num_avg=5):
        return np.mean(self.step_times[-num_avg:])

    def get_total_time(self):
        return time.time() - self.last_timestamp


if __name__ == '__main__':

    print(display_time(45615604540))

