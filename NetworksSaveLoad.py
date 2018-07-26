import torch
import torch.nn as nn
# from NeuralNetworks import CNNsmall, FFsmall
import torch.optim as optim

import os
import shutil
from MyLittleHelpers import prod


# CNet = CNNsmall([1,28,28])
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













# optimizer_real = optim.SGD(CNet.parameters(), lr=1e-3, momentum=0.9, nesterov=True)

# print(optimizer_real.state_dict())




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