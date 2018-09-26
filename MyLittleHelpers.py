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

    def __init__(self, names=['main']):

        self.step_times = {x: [] for x in names}
        self.num_timers = len(names)

        self.start_time = None
        self.start()

    def start(self):
        if self.start_time is None:
            self.start_time = time.time()

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


if __name__ == '__main__':

    phases = ['train', 'val']
    test_timer = Timer(phases)

    test_timer.start()
    for i in range(8):
        for phase in phases:
            if phase == 'train':
                time.sleep(1.5)
            else:
                time.sleep(1.2)
            test_timer.step(phase)

    print('avg_train_5: {}'.format(display_time(test_timer.get_avg_time(num_avg=5, name='train'))))
    print('avg_val_5: {}'.format(display_time(test_timer.get_avg_time(num_avg=5, name='val'))))
    print('avg_all_5: {}'.format(display_time(test_timer.get_avg_time_all(num_avg=5))))

    print('total_train: {}'.format(display_time(test_timer.get_total_time(name='train'))))
    print('total_val: {}'.format(display_time(test_timer.get_total_time(name='val'))))
    print('total_all: {}'.format(display_time(test_timer.get_total_time_all())))





    print()
    # print(display_time(1.5))
