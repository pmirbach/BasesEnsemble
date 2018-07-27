import torch
import torch.nn as nn
from NeuralNetworks import CNNsmall, FFsmall
import torch.optim as optim

import os
import shutil
from MyLittleHelpers import prod


CNet = CNNsmall([1,28,28])
# FFNet = FFsmall([2,28,28])
#
# # print(CNet)
#
# para_ges = 0
# for child in CNet.children():
#     print(child)
#     if hasattr(child, 'weight'):
#         print(child.weight.size())
#         print(child.weight.dtype)
#         print(prod(child.weight.size()))
#         para_ges += prod(child.weight.size())
#
# print(para_ges)






class Netti(nn.Module):
    def __init__(self):
        super(Netti, self).__init__()

        self.convs = [nn.Conv2d(1, 6, 5), nn.Conv2d(6, 12, 5)]

        self.base = [nn.Linear(120, 50), nn.Linear(50,10)]


Net = Netti()

print(Net)




optim.SGD([
                {'params': Net.convs.parameters()},
                {'params': Net.base.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)





# optimizer_real = optim.SGD(CNet.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
optimizer_real = optim.Adam(CNet.parameters(), lr=1e-3)

print(optimizer_real)
print(optimizer_real.state_dict())










class Training():

    def __init__(self, net=None, dataloader=None, optimizer=None, loss_function=None, other_params=None):

        pass

    def set_dataloader(self):
        pass

    def set_optimizer(self):
        pass

    def set_loss_function(self):
        pass

    def set_other_params(self):
        pass














#
# def save_checkpoint(state, is_best, save_path, filename):
#   filename = os.path.join(save_path, filename)
#   torch.save(state, filename)
#   if is_best:
#     bestname = os.path.join(save_path, 'model_best.pth.tar')
#     shutil.copyfile(filename, bestname)
#
# torch.save(net.state_dict(),model_save_path + '_.pth')
#
# save_checkpoint({
#           'epoch': epoch + 1,
#           # 'arch': args.arch,
#           'state_dict': net.state_dict(),
#           'optimizer': optimizer.state_dict(),
#         }, is_best, mPath ,  str(val_acc) + '_' + str(val_los) + "_" + str(epoch) + '_checkpoint.pth.tar')