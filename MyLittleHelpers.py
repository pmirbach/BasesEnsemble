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

    seconds = int(np.round(seconds))

    if seconds == 0:
        return '0:00'

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
    def __init__(self, datasize):
        self.reset()
        self.datasize = datasize

    def reset(self):
        self.running_loss = 0.0
        self.running_corrects = 0
        self.running_seen = 0

    def update(self, batch_avg_loss, batch_corrects, n_batch):
        self.running_seen += n_batch
        self.running_corrects += np.sum(batch_corrects)
        self.running_loss += batch_avg_loss * n_batch

    def get_stats_epoch(self):
        if self.datasize != self.running_seen:
            raise ValueError('Given dataset size and number of seen samples do not match!')
        return self.get_stats()

    def get_stats(self):
        N = self.running_seen
        return self.running_loss / N, self.running_corrects / N * 100


class Timer():

    def __init__(self, names=['main']):

        self.step_times = {x: [] for x in names}
        self.num_timers = len(names)

        self.start_time = time.time()
        self.start()

    def start(self):
        self.last_timestamp = time.time()

    def step(self, name='main'):
        time_diff = time.time() - self.last_timestamp
        self.step_times[name].append(time_diff)

        self.last_timestamp = time.time()
        return time_diff

    def get_avg_time(self, num_avg=5, name='main'):

        step_times = self.step_times[name]
        return np.mean(step_times[-num_avg:])

    def get_avg_time_all(self, num_avg=5):
        out = 0.0
        for name in self.step_times:
            out += self.get_avg_time(num_avg=num_avg, name=name)
        return out

    def get_total_time(self, name='main'):
        return np.sum(self.step_times[name])

    def get_total_time_all(self):
        out = 0.0
        for name in self.step_times:
            out += self.get_total_time(name=name)
        return out

    def total_time(self):
        return time.time() - self.start_time


if __name__ == '__main__':

    # phases = ['train', 'val']
    # test_timer = Timer(phases)
    #
    # test_timer.start()
    # for i in range(8):
    #     for phase in phases:
    #         if phase == 'train':
    #             time.sleep(1.5)
    #         else:
    #             time.sleep(1.2)
    #         test_timer.step(phase)
    #
    # print('avg_train_5: {}'.format(display_time(test_timer.get_avg_time(num_avg=5, name='train'))))
    # print('avg_val_5: {}'.format(display_time(test_timer.get_avg_time(num_avg=5, name='val'))))
    # print('avg_all_5: {}'.format(display_time(test_timer.get_avg_time_all(num_avg=5))))
    #
    # print('total_train: {}'.format(display_time(test_timer.get_total_time(name='train'))))
    # print('total_val: {}'.format(display_time(test_timer.get_total_time(name='val'))))
    # print('total_all: {}'.format(display_time(test_timer.get_total_time_all())))


    # test_stats = LossAccStats(1000)
    # n_batch = 100
    # for i in range(10):
    #     loss_avg = 0.5 - i * 0.1
    #     acc = 30 + i * 5
    #     test_stats.update(loss_avg, acc, n_batch)
    #
    #     print(test_stats.get_stats())
    #
    # a = test_stats.get_stats_epoch()
    # print(a)





    print()
    print(display_time(0))
