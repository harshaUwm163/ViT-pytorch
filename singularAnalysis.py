# %%
from train import *
from addict import Dict
import logging
from utils.construct_tff import construct_real_tff
import matplotlib.pyplot as plt
from utils.data_utils import get_loader

# %%
logger = logging.getLogger(__name__)

# %%
args = Dict()
args.model_type = 'ViT-B_16'
args.img_size = 224
args.pretrained_dir = 'checkpoint/ViT-B_16.npz'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
args.local_rank = -1
args.train_batch_size = 128 
args.eval_batch_size = 128

args.dataset = 'inet1k_birds'
args.dataset_dir = 'data/inet1k_classes/birds'
ckpt_path = 'output/inet1k_birds-2023-10-02-22-25-22/inet1k_birds_final_ckpt.bin'

# args.dataset = 'inet1k_cats'
# args.dataset_dir = 'data/inet1k_classes/cats'
# ckpt_path = 'output/inet1k_cats-2023-10-02-22-19-15/inet1k_cats_final_ckpt.bin' 
# args.dataset = 'inet1k_dogs'
# args.dataset_dir = 'data/inet1k_classes/dogs'
# ckpt_path = 'output/inet1k_dogs-2023-09-24-21-00-17/inet1k_dogs_final_ckpt.bin'
# args.dataset = 'inet1k_snakes'
# args.dataset_dir = 'data/inet1k_classes/snakes'
# ckpt_path = 'output/inet1k_snakes-2023-10-02-22-28-06/inet1k_snakes_final_ckpt.bin'
# args.dataset = 'inet1k_trucks'
# args.dataset_dir = 'data/inet1k_classes/trucks'
# ckpt_path = 'output/inet1k_trucks-2023-09-24-20-47-28/inet1k_trucks_final_ckpt.bin'

# %%
train_loader, test_loader = get_loader(args)

# %%
eig_vals_ns = torch.linspace(0,768,steps=10, dtype=int)[1:]
val_accs = []
with torch.no_grad():
    norm_percent_i = 0
    for eig_val_n in eig_vals_ns:
        args, model = setup(args)
        model.load_state_dict(torch.load(ckpt_path))
        for (name, param) in model.named_parameters():
            if 'attn' in name:
                if 'weight' in name:
                    U, S, Vh = torch.linalg.svd(param)
                    S[eig_val_n:] = 0
                    plt.plot(S.cpu())

                    param.data = U @ torch.diag(S) @ Vh

        plt.savefig(f'singularValues_at_{eig_val_n}.png')
        plt.close()

        val_accs.append(valid(args, model, writer=None, test_loader=test_loader, global_step=0))
mdict = {'eig_vals_ns':eig_vals_ns, 'val_accs':val_accs}
torch.save(mdict, './temp.pt')

# %%
sing_drop_percent = 1. - eig_vals_ns / 768.

# %%
plt.plot(sing_drop_percent, val_accs)
plt.title(f'Val Accuracy Vs % Singular values dropped on {args.dataset}')
plt.xlabel(f'% of singular values dropped')
plt.ylabel(f'Validation Accuracy')
plt.savefig(f'singularAnalysis_{args.dataset}_val_vs_sings.png')
plt.show()

# plt.plot(frame_percents, norm_percents)
# plt.title(f'% Norm retained Vs % frames used on {args.dataset}')
# plt.xlabel(f'% of Frames used')
# plt.ylabel(f'avg % of norm retained')
# plt.savefig(f'{args.dataset}_norm_vs_frames.png')
# plt.show()
# 
# plt.plot(norm_percents, val_accs)
# plt.title(f'Val accuracy Vs % Norm retained on {args.dataset}')
# plt.xlabel(f'avg % of norm retained')
# plt.ylabel(f'Validation accuracy')
# plt.savefig(f'{args.dataset}_Val_vs_norm.png')
# plt.show()
