import matplotlib
gui_env = [i for i in matplotlib.rcsetup.interactive_bk]
non_gui_backends = matplotlib.rcsetup.non_interactive_bk
print ("Non Gui backends are:", non_gui_backends)
print ("Gui backends I will test for", gui_env)
for gui in gui_env:
    print ("testing", gui)
    try:
        matplotlib.use(gui,warn=False, force=True)
        from matplotlib import pyplot as plt
        print ("    ",gui, "Is Available")
        plt.plot([1.5,2.0,2.5])
        fig = plt.gcf()
        fig.suptitle(gui)
        plt.show()
        print ("Using ..... ",matplotlib.get_backend())
    except:
        print ("    ",gui, "Not found")



# import matplotlib
#
# gui_env = ['TKAgg', 'GTKAgg', 'Qt4Agg', 'WXAgg']
# for gui in gui_env:
#     try:
#         print("testing", gui)
#         matplotlib.use(gui, warn=False, force=True)
#         from matplotlib import pyplot as plt
#
#         break
#     except:
#         continue
# print("Using:", matplotlib.get_backend())
#
#
# # import matplotlib
# # matplotlib.use('WXAgg')
# import matplotlib.pyplot as plt


# import torch
#
# import subprocess
# import time
#
# from matplotlib import pyplot as plt
#
#
# device = torch.device('cuda:0')
#
# print(device)
#
# x = torch.randn(100,100,100,100)
#
# x.to(device)
#
# # import torch.cuda as cutorch
#
#
#
# def get_gpu_memory_map():
#     """Get the current gpu usage.
#
#     Returns
#     -------
#     usage: dict
#         Keys are device ids as integers.
#         Values are memory usage as integers in MB.
#     """
#     result = subprocess.check_output(
#         [
#             'nvidia-smi', '--query-gpu=memory.used',
#             '--format=csv,nounits,noheader'
#         ])
#     # Convert lines into a dictionary
#     gpu_memory = [int(x) for x in result.strip().split('\n')]
#     gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
#     return gpu_memory_map
#
# # print(get_gpu_memory_map())
#
# result = subprocess.check_output(
#     [
#         'nvidia-smi', '--query-gpu=memory.used',
#         '--format=csv,nounits,noheader'
#     ])
#
# print(result)
#
# plt.plot([1,2,3,4,5,6])
# plt.show()
