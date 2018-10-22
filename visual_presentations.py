import numpy as np
from matplotlib import pyplot as plt
import pickle
from scipy import interpolate



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

    fig.colorbar(im, ax=axes[:,it].ravel().tolist())
plt.show()


