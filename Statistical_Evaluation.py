import numpy as np
from itertools import product
import copy
import os

import pickle
from matplotlib import pyplot as plt


def get_options(id, features, num_statistical=1):
    combinations = product(*features.values())
    comb_list = (list(combinations))

    t, run = divmod(id, num_statistical)

    if id >= len(comb_list) * num_statistical:
        raise ValueError('id surpasses expected number of runs!')
    if t >= len(comb_list):
        raise ValueError('test2')

    d = dict((key, value) for (key, value) in zip(features.keys(), comb_list[t]))
    return d, run


features = {'batchsize': [50, 100, 150, 200, 250, 300, 350, 400], 'adLR': [0, 1]}
combinations = product(*features.values())

metrics = ['train_loss', 'validate_loss', 'train_accuracy', 'validate_accuracy', 'train_time', 'validate_time']
dict_blueprint = {metric: {'all': np.zeros((60,20)), 'mean': np.zeros(60,), 'std': np.zeros(60,)} for metric in metrics}
results = dict((key, copy.deepcopy(dict_blueprint)) for key in list(combinations))
# print(results)




for i in range(320):
    params, run = get_options(i, features, num_statistical=20)

    result_path = './results/ex1/fashion-mnist/'
    result_file = 'bs' + str(params['batchsize']) + 'adLR' + str(params['adLR']) + '_run' + str(run) + '.pkl'

    infile = open(result_path + result_file, 'rb')
    (hyper_params, stats) = pickle.load(infile)
    infile.close()

    key = (hyper_params['batchsize'], hyper_params['adLR'])

    for metric, data in stats.items():
        results[key][metric]['all'][:, run] = data

# print(results[(50, 0)]['train_loss']['all'])
for data in results.values():
    for metric in metrics:
        data[metric]['mean'] = np.mean(data[metric]['all'], axis=1)
        data[metric]['std'] = np.std(data[metric]['all'], axis=1)


plot_bs = 200

x = np.arange(1,61)

for key in [(plot_bs, 0), (plot_bs, 1)]:

    if key[1] == 0:
        label = 'normal'
        c = [0,0,1,1]
        fc = [0,0,1,0.2]
        ec = [0,0,0.8,0.4]
    else:
        label = 'adaptive'
        c = [1,0,0,1]
        fc = [1,0,0,0.2]
        ec = [0.8,0,0,0.4]

    mean = results[key]['validate_accuracy']['mean']
    std = results[key]['validate_accuracy']['std']

    plt.plot(x, mean, color=c, label=label)
    plt.fill_between(x, mean-std, mean+std, facecolor=fc, edgecolor=ec)

plt.axis([1,61,70,95])
# plt.axis('scaled')
plt.legend(loc='lower right')
plt.show()





# fig, axes = plt.subplots(2,3)
#
# for metric, ax in zip(metrics, axes.flatten()):
#     ax.set_title(metric)
#     ax.plot(x, mean)
#     ax.fill_between(x,mean-std,mean+std)


plt.show()


















