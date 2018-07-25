import torch
from NeuralNetworks import CNNsmall


CNet = CNNsmall([1,28,28])

print(CNet)



def save_checkpoint(state, is_best, save_path, filename):
  filename = os.path.join(save_path, filename)
  torch.save(state, filename)
  if is_best:
    bestname = os.path.join(save_path, 'model_best.pth.tar')
    shutil.copyfile(filename, bestname)

torch.save(net.state_dict(),model_save_path + '_.pth')

save_checkpoint({
          'epoch': epoch + 1,
          # 'arch': args.arch,
          'state_dict': net.state_dict(),
          'optimizer': optimizer.state_dict(),
        }, is_best, mPath ,  str(val_acc) + '_' + str(val_los) + "_" + str(epoch) + '_checkpoint.pth.tar')