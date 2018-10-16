import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import sys, inspect, copy
import time

import os, psutil

from MyLittleHelpers import display_time, Timer, LossAccStats

import pickle

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, experiment, num_epochs=25, ad_lr=True):
    timer = Timer(['train', 'validate'])
    stats = {x: LossAccStats(len(dataloaders[x].dataset)) for x in ['train', 'validate']}
    steps = {x: 0 for x in ['train', 'validate']}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    disp_epoch = 'Epoch {f}/{f}'.format(f='{:' + str(len(str(num_epochs))) + 'd}')
    disp_phase_stats = ' - {}: [{:>7.4f}, {:>7.3f}%]'
    disp_time_remaining = ' - Approx. time left: {}'

    for epoch in range(num_epochs):
        print(disp_epoch.format(epoch + 1, num_epochs), end='', flush=True)

        for phase in ['train', 'validate']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            epoch_lr = optimizer.param_groups[0]['lr']
            def layer_lr_reset():
                for group in optimizer.param_groups:
                    group['lr'] = epoch_lr

            stats[phase].reset()

            for inputs, labels in dataloaders[phase]:
                num_inp = inputs.size(0)

                inputs = inputs.to(device)
                labels = labels.to(device)

                # if phase == 'train':
                #     def closure():
                #         optimizer.zero_grad()
                #         with torch.set_grad_enabled(True):
                #             outputs = model(inputs)
                #             _, preds = torch.max(outputs, 1)
                #             loss = criterion(outputs, labels)
                #             loss.backward()
                #             return loss
                #     loss = optimizer.step(closure)
                #
                # if phase == 'validate':
                #     optimizer.zero_grad()
                #     with torch.set_grad_enabled(False):
                #         outputs = model(inputs)
                #         _, preds = torch.max(outputs, 1)
                #         loss = criterion(outputs, labels)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        def closure():
                            loss.backward()

                            if ad_lr:
                                for group in optimizer.param_groups:
                                    buf = 0.0
                                    for p in group['params']:
                                        buf += p.norm(2).item()
                                    group['lr'] *= 1 + np.log(1 + 1 / buf)

                            return loss

                        optimizer.step(closure)
                        if ad_lr:
                            layer_lr_reset()

                stats[phase].update(loss.item(), torch.sum(preds == labels.data).item(), num_inp)
                step_loss, step_acc = stats[phase].get_stats()

                if phase == 'train':
                    experiment.log_metric(phase + '_loss', step_loss, step=steps[phase])
                # else:
                #     experiment.log_metric(phase + '_accuracy', step_acc, step=steps[phase])
                steps[phase] += 1

            epoch_loss, epoch_acc = stats[phase].get_stats_epoch()
            experiment.log_metric(phase + '_accuracy', epoch_acc, step=epoch+1)
            print(disp_phase_stats.format(phase.title(), epoch_loss, epoch_acc), end='', flush=True)

            if phase == 'validate' and epoch_acc > best_acc:
                best_model_wts = copy.deepcopy(model.state_dict())
                best_acc = epoch_acc

            time_elapsed = timer.step(phase)
            experiment.log_metric(phase + '_time', time_elapsed, step=epoch+1)

        experiment.log_epoch_end(epoch_cnt=epoch+1)

        time_estimate = timer.get_avg_time_all(num_avg=5) * (num_epochs - (epoch + 1))
        print(disp_time_remaining.format(display_time(time_estimate)))

    time_elapsed = timer.total_time()
    print('Training complete in {} - Best val Acc: {:4f}'.format(display_time(time_elapsed), best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# TODO implement different adaptive learning functions

def ad_lr_fun1(params):
    buf = 0
    for p in params:
        g = p.grad.data
        buf += g.norm(2).item()
    return 1 + np.log(1 + 1 / buf)

def ad_lr_fun2(params):
    buf = 0
    for p in params:
        buf += p.norm(2).item()
    return 1 + np.log(1 + 1 / buf)



def train_epoch(model, dataloader, criterion, optimizer, device, comet_exp=None, ad_lr=0):

    model.train()

    stats = LossAccStats(len(dataloader.dataset))
    if ad_lr is not None:
        epoch_lr = optimizer.param_groups[0]['lr']
        def layer_lr_reset():
            for group in optimizer.param_groups:
                group['lr'] = epoch_lr

    for inputs, labels in dataloader:
        num_inp = inputs.size(0)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        def closure():
            loss.backward()

            if ad_lr is not None:
                for group in optimizer.param_groups:
                    fac = ad_lr_fun1(group['params'])
                    group['lr'] *= fac
            return loss

        optimizer.step(closure)
        if ad_lr is not None:
            layer_lr_reset()

        stats.update(loss.item(), torch.sum(preds == labels.data).item(), num_inp)
        step_loss, step_acc = stats.get_stats()

        if comet_exp is not None:
            comet_exp.log_metric('train_loss', step_loss)

    epoch_loss, epoch_acc = stats.get_stats_epoch()
    return epoch_loss, epoch_acc


def validate_model(model, dataloader, criterion, device):

    model.eval()

    stats = LossAccStats(len(dataloader.dataset))

    for inputs, labels in dataloader:
        num_inp = inputs.size(0)
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        stats.update(loss.item(), torch.sum(preds == labels.data).item(), num_inp)

    epoch_loss, epoch_acc = stats.get_stats_epoch()
    return epoch_loss, epoch_acc


def training(model, dataloaders, criterion, optimizer, scheduler, device, comet_exp, num_epochs=25, ad_lr=0):
    timer = Timer(['train', 'validate'])
    # steps = {x: 0 for x in ['train', 'validate']}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    disp_epoch = 'Epoch {f}/{f}'.format(f='{:' + str(len(str(num_epochs))) + 'd}')
    disp_phase_stats = ' - {}: [{:>7.4f}, {:>7.3f}%]'
    disp_time_remaining = ' - Approx. time left: {}'

    for epoch in range(num_epochs):
        scheduler.step()
        train_epoch_loss, train_epoch_acc = train_epoch(model, dataloaders['train'], criterion, optimizer, device,
                                            comet_exp=None, ad_lr=ad_lr)
        train_epoch_time = timer.step('train')

        validate_epoch_loss, validate_epoch_acc = validate_model(model, dataloaders['validate'], criterion, device)
        validate_epoch_time = timer.step('validate')

        if validate_epoch_acc > best_acc:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc = validate_epoch_acc

        comet_log = {'train_loss': train_epoch_loss,
                     'train_accuracy': train_epoch_acc, 'validate_accuracy': validate_epoch_acc,
                     'train_time': train_epoch_time, 'validate_time': validate_epoch_time}
        comet_exp.log_multiple_metrics(comet_log, step=epoch+1)
        comet_exp.log_epoch_end(epoch_cnt=epoch+1)

        time_estimate = timer.get_avg_time_all(num_avg=5) * (num_epochs - (epoch + 1))

        print(disp_epoch.format(epoch + 1, num_epochs),
              disp_phase_stats.format('train', train_epoch_loss, train_epoch_acc),
              disp_phase_stats.format('train', validate_epoch_loss, validate_epoch_acc),
              disp_time_remaining.format(display_time(time_estimate)), flush=True)

    time_elapsed = timer.total_time()
    print('Training complete in {} - Best validation Acc: {:4f}'.format(display_time(time_elapsed), best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model




def _optimizer_to_ctrl(optimizer, Net):

    def _create_id_name_map(Net):
        id_name_map = {}
        for name, param in Net.named_parameters():
            id_name_map[id(param)] = name
        return id_name_map

    def _id_to_names(id_list, id_name_map):
        name_list = []
        for ida in id_list:
            name_list.append(id_name_map[ida])
        return name_list

    id_name_map = _create_id_name_map(Net)

    param_groups_names = []
    for param in optimizer.state_dict()['param_groups']:
        param_groups_names.append(_id_to_names(param['params'], id_name_map))

    opti_ctrl = {'method': type(optimizer).__name__, 'params': param_groups_names,
                 'state_dict': optimizer.state_dict()}
    return opti_ctrl


def _params_from_ctrl(opti_ctrl, Net):

    def _create_name_param_map(Net):
        name_param_map = {}
        for name, param in Net.named_parameters():
            name_param_map[name] = param
        return name_param_map

    def _gen_name_to_param(name_list, name_param_map):
        for name in name_list:
            yield name_param_map[name]

    name_param_map = _create_name_param_map(Net)
    params = []
    for group_names in opti_ctrl['params']:
        params.append({'params': _gen_name_to_param(group_names, name_param_map)})

    return params


class Training():

    # TODO dataloader, loss, other_params, save, memory estimator
    # TODO dataloader
    # TODO loss
    # TODO other_params
    # TODO save
    # TODO memory estimator

    optim_cls = dict(inspect.getmembers(sys.modules['torch.optim'], inspect.isclass))

    def __init__(self, Net=None, dataloader=None, validloader=None, optimizer=None, loss_function=None, other_params=None):
        self.Net = Net

        self.dataloader = dataloader
        self.opti_ctrl = self.set_opti_ctrl(optimizer)
        self.loss_function = loss_function

        self.validloader = validloader

        self.device = other_params['device']
        self.n_epoch = other_params['n_epoch']
        self.inp_split = other_params['inp_split']

        self.train_hist = {'time_hist': [], 'loss_hist': [], 'valid_hist': []}
        # self.train_state = {'time': [], 'loss': [], 'validation_error': []}


    def set_dataloader(self):
        pass

    def set_opti_ctrl(self, optimizer):
        if type(optimizer).__name__ in self.optim_cls.keys():
            opti_ctrl = _optimizer_to_ctrl(optimizer, self.Net)
        else:
            raise ValueError('optimizer must be pytorch optimizer!')
        return opti_ctrl

    def init_optimizer(self):
        opti_method = getattr(torch.optim, self.opti_ctrl['method'])
        if self.opti_ctrl['params'] is None:
            optimizer = opti_method(params=self.Net.parameters(), lr=1e-3)
        else:
            params = _params_from_ctrl(self.opti_ctrl, self.Net)
            optimizer = opti_method(params=params, lr=1e-3)
        optimizer.load_state_dict(self.opti_ctrl['state_dict'])
        return optimizer

    def print_opti(self):
        print('Optimizer:')
        dummy_optimizer = self.init_optimizer()
        print(dummy_optimizer)
        for i in range(len(self.opti_ctrl['params'])):
            st = 'Parameter Group {}: {}'.format(i, self.opti_ctrl['params'][i])
            print(st)

    def set_loss_function(self):
        pass

    def set_other_params(self):
        pass

    def estimate_memory(self):
        pass

    def save_hist(self):
        pass

    def training(self):
        self.Net.to(self.device)

        pid = os.getpid()
        py = psutil.Process(pid)

        optimizer = self.init_optimizer()

        print('Start Training...')

        best_pred = 0.0

        for epoch in range(self.n_epoch):

            self.Net.train()

            t0_epoch = time.time()
            epoch_loss = 0

            for i, data in enumerate(self.dataloader, start=0):

                if self.inp_split is None:
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                else:
                    inputs, labels = self.inp_split(data, self.device)
                # inputs = data[0]
                # inputs = [inp.to(device) for inp in inputs]
                #
                # labels = data[1].to(device)

                optimizer.zero_grad()
                outputs = self.Net(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
            print('memory use:', memoryUse)

            self.Net.eval()

            num_images = len(self.validloader.dataset)
            batchsize = self.validloader.batch_size

            res = np.zeros(num_images)

            with torch.no_grad():
                for i, data in enumerate(self.validloader):

                    inputs, labels = data[0].to(self.device), data[1].to(self.device)

                    outputs = self.Net(inputs)
                    _, predicted = torch.max(outputs.data, 1)

                    comp = predicted == labels

                    try:
                        res[i * batchsize: (i + 1) * batchsize] = comp
                    except ValueError:
                        res[-comp.shape[0]:] = comp

            memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
            print('memory use:', memoryUse)

            pred = 100 * np.sum(res) / num_images
                # print('Accuracy of the network on the {} test images: {} %'.format(num_images,
                #                                                                    100 * np.sum(res) / num_images))


            self.train_hist['time_hist'].append(time.time() - t0_epoch)
            mean_time = np.mean(self.train_hist['time_hist'][-5:])
            time_left_est = (self.n_epoch - (epoch + 1)) * mean_time

            self.train_hist['loss_hist'].append(epoch_loss / len(self.dataloader))

            self.train_hist['valid_hist'].append(pred)

            print("===> Epoch [{:2d}] / {}: Loss: {:.4f} - Valid: {:5.2f} - Approx. {:5.1f} seconds left".format(
                epoch + 1, self.n_epoch,
                self.train_hist['loss_hist'][-1],
                self.train_hist['valid_hist'][-1], time_left_est))

        print('Finished Training - Duration: {0:5.1f} seconds'.format(np.sum(self.train_hist['time_hist'])))





if __name__ == '__main__':
    from NeuralNetworks import CNNsmall
    from torch.utils.data import DataLoader
    import torchvision


    class TryNet(nn.Module):
        def __init__(self):
            super(TryNet, self).__init__()
            self.pre = nn.Sequential(nn.Linear(200, 100), nn.Linear(100, 50), nn.BatchNorm1d(50))
            self.base = nn.Linear(50, 10)
            self.post = nn.Linear(10, 1)

        def forward(self, x):
            x = self.pre(x)
            x = self.base(x)
            return self.post(x)


    TN = TryNet()

    special_params = [{'params': TN.pre.parameters()},
                      {'params': TN.base.parameters(), 'lr': 1e-5},
                      {'params': TN.post.parameters(), 'lr': 1e-7}
                      ]


    def fake_training(model, optimizer):
        for _ in range(2):
            optimizer.zero_grad()
            out = model(torch.randn(200))
            out.backward()
            optimizer.step()


    root = './data'
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    train_set_real = torchvision.datasets.FashionMNIST(root=root, train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_set_real, batch_size=20, shuffle=True, num_workers=0)

    Net_real = CNNsmall(train_set_real[0][0].shape)

    optimizer_real = optim.SGD(Net_real.parameters(), lr=1e-3, momentum=0.9, nesterov=True)

    criterion = nn.CrossEntropyLoss()

    other_params = {'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    'n_epoch': 20, 'inp_split': None}

    training = Training(Net_real, train_loader, optimizer_real, criterion, other_params)
    # training.training()

