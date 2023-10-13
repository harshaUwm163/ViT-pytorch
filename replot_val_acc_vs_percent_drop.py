import torch
import matplotlib.pyplot as plt
import os

data_paths = ['output/inet1k_birds-frame_analysis-2023-10-09-02-20-56/', 
              'output/inet1k_cats-frame_analysis-2023-10-09-02-21-19', 
              'output/inet1k_dogs-frame_analysis-2023-10-09-02-22-45', 
              'output/inet1k_snakes-frame_analysis-2023-10-09-02-25-32', 
              'output/inet1k_trucks-frame_analysis-2023-10-09-02-27-18' ]

for data_path in data_paths:
    res_mat = torch.load(os.path.join(data_path, 'final_res.pt'))
    val_dict = res_mat['opt_res_dict']

    k_attn = 64
    k_is = list(val_dict.keys())
    vals = list(val_dict.values())
    percent_drop = [1. - k_i / k_attn for k_i in k_is]

    dataset = data_path.split('/')[1].split('-')[0]

    plt.plot(percent_drop, vals)
    plt.title(f'Val Accuracy Vs % frames used on {dataset}')
    plt.xlabel(f'% of Frames used')
    plt.ylabel(f'Validation Accuracy')
    plt.savefig(os.path.join(data_path, 'val_acc_vs_percent_drop.png'))
    plt.close()

