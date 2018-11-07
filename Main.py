import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import torchvision
from torchvision import datasets as datasets
from torchvision import transforms as transforms

from NeuralNetworks import CNNsmall, FFsmall, EnsembleMLP, ResNetLinear
from Transformations import transformation_fourier, normalize_linear
from Datasets import DatasetBase, MyStackedDataset, get_dataset, get_transforms, get_normalize_stats
from TrainValidateTest import Training, train_model, training, validate_model

from itertools import product

# import matplotlib.pyplot as plt

from MyLittleHelpers import mkdir
from comet_ml import Experiment

import time, os, copy, socket
import psutil, errno
import pickle

import argparse
import ResNet_cifar


flg_visualize = 0


dataset_names = ('mnist', 'fashion-mnist', 'emnist', 'cifar10', 'cifar100')

parser = argparse.ArgumentParser(description='PyTorchLab')
parser.add_argument('-d', '--dataset', metavar='data', default='cifar10', choices=dataset_names,
                    help='dataset to be used: ' + ' | '.join(dataset_names) + ' (default: cifar10)')

parser.add_argument('-b', '--batchsize', type=int, default=128, help='training batch size (default: 128)')

parser.add_argument('-c', '--comet', action='store_true', help='track training data to comet.ml')

parser.add_argument('--adLR', type=int, default=0, help='adaptive layer learning rate function (default: 0)')

parser.add_argument('--num_epochs', type=int, default=60, help='number of epoch for training (default: 60)')

parser.add_argument('--lr_initial', type=float, default=1e-3, help='initial learning rate for all layers')

parser.add_argument('--lr_step', type=int, default=30, help='epochs to reduce lr to 1/10')

parser.add_argument('-id', '--slurmId', type=int, help='slurm id for array jobs')

parser.add_argument('--testset', action='store_true', help='False: use test set for validation')

parser.add_argument('-tvr', '--train_validate_ratio', metavar='ratio', type=float, default=0.9,
                    help='ratio training vs. validation, if testset is used (default: 0.9)')

args = parser.parse_args()

hyper_params = vars(args)
flg_comet = not hyper_params.pop('comet')
slurmId = hyper_params.pop('slurmId')


comet_exp = Experiment(api_key="dI9d0Dyizku98SyNE6ODNjw3L",
                        project_name="adLR_resnet", workspace="pmirbach",
                        disabled=flg_comet)

### Slurm Array Handling

if slurmId is not None:

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
    param_update, run = get_options(slurmId, features, num_statistical=20)

    for key, value in param_update.items():
        hyper_params[key] = value

else:
    run = 0


print(hyper_params)

comet_exp.log_multiple_params(hyper_params)

# print(slurmId)


### Data dir based on host

current_host = socket.gethostname()
if current_host == 'nevada':
    data_dir = '/raid/common/' + hyper_params['dataset']
else:
    data_dir = './data/' + hyper_params['dataset']



train_mean, train_std, test_mean, test_std = get_normalize_stats(hyper_params['dataset'])
train_transform = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(train_mean, train_std),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(test_mean, test_std),
])
data_transforms = {'train': train_transform, 'test': test_transform}

# TODO Loading routines for EMNIST, imageNet
dset = get_dataset(hyper_params['dataset'], data_dir, data_transforms['train'], data_transforms['test'])

flg_test_as_val = True
if flg_test_as_val:
    phases = ['train', 'validate']
    dset['train'], dset['validate'] = dset['training'], dset['test']
else:
    phases = ['train', 'validate', 'test']
    train_size = int(len(dset['training']) * hyper_params['train_validate_ratio'])
    val_size = len(dset['training']) - train_size
    dset['train'], dset['validate'] = torch.utils.data.random_split(dset['training'], [train_size, val_size])

dataloaders = {x: DataLoader(dset[x], batch_size=hyper_params['batchsize'], shuffle=True) for x in phases}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def get_layer_params(model, lr):
    special_params = []
    for name, module in model.named_children():
        if len(list(module.parameters())) != 0:
            special_params.append(
                {'params': module.parameters(), 'lr': lr}
            )
    return special_params

###
# Net = CNNsmall(inp_shape=dset['train'][0][0].shape)
###
# Net = torchvision.models.vgg11_bn(pretrained=False, num_classes=10)
# Net.classifier[0] = nn.Linear(in_features=512, out_features=4096)

# Net = torchvision.models.vgg16_bn(pretrained=False, num_classes=10)
## Net.classifier[0] = nn.Linear(in_features=512, out_features=4096)
# Net.classifier = nn.Linear(in_features=512, out_features=10)

# Net = torchvision.models.vgg11(pretrained=False)
# Net.classifier = nn.Linear(in_features=512, out_features=10)

# Net = torchvision.models.resnet18(pretrained=False, num_classes=10)
Net = ResNet_cifar.ResNet18()


# Net = FFsmall(inp_shape=image_datasets['train'][0][0].shape)
Net.to(device)
criterion = nn.CrossEntropyLoss()


params_layer = get_layer_params(Net, hyper_params['lr_initial'])

# optimizer_normal = optim.SGD(Net.parameters(), lr=1e-3, momentum=0.9, nesterov=False, weight_decay=5e-4)
# scheduler = optim.lr_scheduler.StepLR(optimizer_normal, step_size=hyper_params['lr_step'], gamma=0.1)
##
optimizer_layers = optim.SGD(params_layer, momentum=0.9, nesterov=True)
scheduler = optim.lr_scheduler.StepLR(optimizer_layers, step_size=hyper_params['lr_step'], gamma=0.1)
##

### Adaptive layer learning
def adlr_1(x):
    return 1 + np.log(1 + 1 / x)
def adlr_2(x):
    return 1

if hyper_params['adLR'] == 0:
    adlr_fun = None
elif hyper_params['adLR'] == 1:
    adlr_fun = adlr_1
elif hyper_params['adLR'] == 2:
    adlr_fun = adlr_2



###
# Net, stats, weight_gradient_norms = training(Net, dataloaders, criterion, optimizer_layers, scheduler, device, comet_exp,
#                num_epochs=hyper_params['num_epochs'], adlr_fun=adlr_fun)
###
Net, stats, weight_gradient_norms = training(Net, dataloaders, criterion, optimizer_layers, scheduler, device, comet_exp,
               num_epochs=hyper_params['num_epochs'], adlr_fun=adlr_fun)

if 'test' in phases:
    test_loss, test_acc = validate_model(Net, dataloaders['test'], criterion, device)
    print('test - : [{:>7.4f}, {:>7.3f}%]'.format(test_loss, test_acc))

    comet_exp.log_multiple_metrics({'test_loss': test_loss, 'test_accuracy': test_acc})


# result_path = './results/ex1_2/fashion-mnist/'
# # result_file = 'bs' + str(hyper_params['batchsize']) + 'adLR' + str(hyper_params['adLR']) + '_run' + str(run) + '.pkl'
# result_file = 'weigths_grads_norm' + '_adLR' + str(hyper_params['adLR']) + '.pkl'
#
#
# outfile = open(result_path + result_file, 'wb')
# pickle.dump((hyper_params, stats, weight_gradient_norms), outfile)
# outfile.close()





if __name__ == '__main__2':

    flg_batchsize = 64

    # TODO Take a look at argparser: What is it used for?

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO Where to put all this stuff? Datasets.py?
    # root = './data'
    root = './data/fashion_mnist/'

    # transform = torchvision.transforms.ToTensor()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    train_set_real = torchvision.datasets.FashionMNIST(root=root, train=True, transform=transform, download=True)
    test_set_real = torchvision.datasets.FashionMNIST(root=root, train=False, transform=transform, download=True)

    # train_set_real = torchvision.datasets.MNIST(root=root, train=True, transform=transform, download=True)
    # test_set_real = torchvision.datasets.MNIST(root=root, train=False, transform=transform, download=True)

    # TODO Load datasets from /raid/common

    path_ft = {'root': root, 'orig_data': 'processed', 'trafo_data': 'fourier', 'trafo_prefix': 'ft'}
    train_set_ft = DatasetBase(name='Fourier', path=path_ft, train=True, base_trafo=transformation_fourier,
                               normalizer=normalize_linear, recalc=False)
    test_set_ft = DatasetBase(name='Fourier', path=path_ft, train=False, base_trafo=transformation_fourier,
                              normalizer=normalize_linear, recalc=False)

    train_set_total = MyStackedDataset([train_set_real, train_set_ft])
    test_set_total = MyStackedDataset([test_set_real, test_set_ft])

    train_loader = DataLoader(dataset=train_set_real, batch_size=flg_batchsize, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_set_real, batch_size=flg_batchsize, shuffle=False, num_workers=0)
    train_loader_ft = DataLoader(dataset=train_set_ft, batch_size=flg_batchsize, shuffle=True, num_workers=10)
    test_loader_ft = DataLoader(dataset=test_set_ft, batch_size=flg_batchsize, shuffle=False, num_workers=10)
    train_loader_total = DataLoader(dataset=train_set_total, batch_size=64, shuffle=True, num_workers=10)
    test_loader_total = DataLoader(dataset=test_set_total, batch_size=64, shuffle=False, num_workers=10)

    Net_real = CNNsmall(train_set_real[0][0].shape)
    Net_ft = FFsmall(train_set_ft[0][0].shape)
    # Net_ft = ResNetLinear(train_set_ft[0][0].shape)

    criterion = nn.CrossEntropyLoss()

    flg_real = 0
    flg_ft = 0
    flg_ens = 0

    optimizer_real = optim.SGD(Net_real.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
    other_params = {'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    'n_epoch': 50, 'inp_split': None}

    train = Training(Net_real, train_loader, test_loader, optimizer_real, criterion, other_params)

    print(psutil.__version__)
    # print(psutil.cpu_percent())
    # print(psutil.virtual_memory())

    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory use:', memoryUse)

    train.training()
    print(train.train_hist)

    if flg_real:
        print(Net_real)
        optimizer_real = optim.SGD(Net_real.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
        Net_real.train_Net(train_loader, 20, criterion, optimizer_real, device)
        pred_real = Net_real.validate_Net(test_loader, device)

    if flg_ft:
        print(Net_ft)
        optimizer_ft = optim.SGD(Net_ft.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
        Net_ft.train_Net(train_loader_ft, 40, criterion, optimizer_ft, device)
        pred_ft = Net_ft.validate_Net(test_loader_ft, device)

    if flg_ens:
        EnsNet = EnsembleMLP([Net_real, Net_ft])
        optimizer_total = optim.SGD(EnsNet.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
        EnsNet.train_Net(train_loader_total, 40, criterion, optimizer_total, device)
        pred_total = EnsNet.validate_Net(test_loader_total, device)

    if False:
        # TODO Make little Helper:
        model_path = r'./results/models'
        mkdir(model_path)

        # TODO Create a NN Save Load module: Models, Trainstatus, ...
        # file_name = 'FF_fourier_trained.pt'
        file_name = 'CNN_real_trained_2.pt'
        torch.save(Net, os.path.join(model_path, file_name))
