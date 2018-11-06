import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pickle
from scipy import interpolate


matplotlib.rcParams.update({'font.size': 16})

result_path = './results/ex1_2/fashion-mnist/'
files = ('weigths_grads_norm' + '_adLR0' + '.pkl', 'weigths_grads_norm' + '_adLR1' + '.pkl')

res = []
for file in files:
    with open(result_path + file, 'rb') as infile:
        (hyper_params, stats, weight_gradient_norms) = pickle.load(infile)
    res.append(np.concatenate(weight_gradient_norms, axis=1))

# print(res[0].shape)

def smooth_inter_fun(r):
    s = interpolate.interp1d(np.arange(len(r)), r)
    xnew = np.arange(0, len(r)-1, .1)
    return s(xnew)
#
# new_data = np.vstack([smooth_inter_fun(r) for r in data])


layer_parameter_counts = np.array([250, 5000, 40960, 8192, 640]).reshape((5,1))

tt = ['weights', 'gradients']

fig, axes = plt.subplots(2,2)

fig.suptitle('Evolution of the L2-norm of weights and gradients of network layers')

for it, t in enumerate(tt):

    vmin, vmax = 0.0, 0.0
    for i in range(len(res)):
        data = res[i][:,:,it]
        # data /= layer_parameter_counts / 1000

        vmin = np.min([np.min(data), vmin])
        vmax = np.max([np.max(data), vmax])

    for i in range(len(res)):
        data = res[i][:,:,it]
        # data /= layer_parameter_counts / 1000
        # data -= np.mean(data)

        # new_data = np.vstack([smooth_inter_fun(r) for r in data])

        # im = axes[i, it].imshow(data, cmap='jet', vmin=vmin, vmax=vmax)
        im = axes[i, it].imshow(data, cmap='coolwarm')
        axes[i, it].axis('tight')
        # if i == len(res) - 1:
        #     print(3)
        #     axes[i, it].xlabel('test')

    fig.colorbar(im, ax=axes[:,it].ravel().tolist())


axes[0,0].title.set_text('Layer weights')
axes[0,1].title.set_text('Layer gradients')


# left, width = .25, .5
# bottom, height = .25, .5
# right = left + width
# top = bottom + height
#
# axes[0,0].text(left, 0.5*(bottom+top), 'right center',
#         horizontalalignment='right',
#         verticalalignment='center',
#         rotation='vertical',
#         transform=axes[0,0].transAxes)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

plt.show()




# Out of main - show some dataset images

# if flg_visualize:
#     train_dataset = dset['training']
#     fig = plt.figure(figsize=(8,8))
#     columns = 4
#     rows = 5
#     for i in range(1, columns*rows +1):
#         img_xy = np.random.randint(len(train_dataset))
#         img = train_dataset[img_xy][0][0,:,:]
#         fig.add_subplot(rows, columns, i)
#         # plt.title(labels_map[train_dataset[img_xy][1]])
#         plt.axis('off')
#         plt.imshow(img, cmap='gray')
#     plt.show()


