from train import valid, set_seed
from apex import amp
import torch
import argparse
from addict import Dict
import logging
import matplotlib.pyplot as plt
import logging
from utils.data_utils import get_loader
from models.modeling import VisionTransformer, CONFIGS
import numpy as np
from torch.utils.data import Subset
import torch.nn.functional as F
import os
import datetime
from tqdm import tqdm

from utils.construct_tff import construct_real_tff, construct_tight_frames

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

parser = argparse.ArgumentParser()
# Required parameters
parser.add_argument("--exp_name", type=str, default='debug_thread',
                    help="Name of this run. Used for monitoring.")
parser.add_argument("--model_type", type=str, default='ViT-B_16',
                    help="Type of the ViT you are using")
parser.add_argument("--img_size", type=int, default=224,
                    help="resolution of the square images")
parser.add_argument("--pretrained_dir", type=str, default='checkpoint/ViT-B_16-224.npz',
                    help="resolution of the square images")
parser.add_argument("--device", type=str, default=None,
                    help="device that you want to run on; currently this is just a placeholder")
parser.add_argument("--local_rank", type=int, default=-1,
                    help="relevant when you have multiple devices")
parser.add_argument("--train_batch_size", type=int, default=16,
                    help="training batch size")
parser.add_argument("--eval_batch_size", type=int, default=16,
                    help="eval batch size")
parser.add_argument("--dataset", type=str, default='inet1k_birds',
                    help="name of the dataset")
parser.add_argument("--dataset_dir", type=str, default='data/inet1k_classes/birds',
                    help="path to the dataset")
parser.add_argument("--ckpt_path", type=str, default='output/inet1k_birds-2023-10-17-03-04-30/inet1k_birds_final_ckpt.bin',
                    help="path to the saved checkpoint of the model")
parser.add_argument("--coef_est_type", type=str, default='weiner', choices = ['weiner', 'naive'],
                    help="how to estimate the weights from the quantized versions")
parser.add_argument("--save_path", type=str, default=None, 
                    help="provide the savepath; otherwise a cat of exp_name and current time will be used")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
ckpt_path = args.ckpt_path

exp_name = args.exp_name # 'mlp_attn_quant_weiner_full'
if args.save_path is None:
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    directory_path = os.path.join("output", f'{exp_name}_{current_datetime}')
    args.save_path = directory_path
else:
    directory_path = args.save_path

os.makedirs(directory_path, exist_ok=True)

logging.basicConfig(filename= os.path.join(directory_path, 'log.log'), level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Prepare model
config = CONFIGS[args.model_type]

if args.dataset == "cifar10":
    num_classes = 10
elif args.dataset == "cifar100":
    num_classes = 100
elif 'inet' in args.dataset:
    num_classes = 10

print('args = ')
logging.info('args = ')
for k,v in vars(args).items():
    print(f'{k}: {v}')
    logging.info(f'{k}: {v}')

model = VisionTransformer(config, args.img_size, zero_head=False, num_classes=num_classes)
# model.load_from(np.load(args.pretrained_dir))
model.load_state_dict(torch.load(ckpt_path))
model.to(args.device)
model.eval()
num_params = count_parameters(model)

logging.info("{}".format(config))
logging.info("Training parameters %s", args)
logging.info("Total Parameter: \t%2.1fM" % num_params)

train_loader, test_loader = get_loader(args)
classes = train_loader.dataset.dataset.classes

val_acc = valid(args, model, writer=None, test_loader=test_loader, global_step=0)
print(f'original model validation accuracy = {val_acc}')
logging.info(f'original model validation accuracy = {val_acc}')

def quantize_qfna(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

# This is with the Weiner Filter based estimation
redundancies = [1]
bws = [2,3,4,8,12,16]

val_accs = {}
compressions = {}
for redundancy in redundancies:
    print(f'{redundancy = }')
    logging.info(f'{redundancy = }')
    # set tff related params
    tffs = {}
    k_attn = int(96 * redundancy)
    l_attn = 8
    n_attn = 768
    tffs[n_attn] = construct_real_tff(k_attn, l_attn // 2, n_attn // 2).to(args.device)

    k_mlp = int(384 * redundancy)
    l_mlp = 8
    n_mlp = 3072
    tffs[n_mlp] = construct_real_tff(k_mlp, l_mlp // 2, n_mlp // 2).to(args.device)

    k_head = int(5 * redundancy)
    l_head = 2
    n_head = 10
    tffs[n_head] = construct_real_tff(k_head, l_head // 2, n_head // 2).to(args.device)

    val_accs_r = []
    compressions_r = []
    for bw in bws:
        print(f'{bw = }')
        logging.info(f'{bw = }')
        model = VisionTransformer(config, args.img_size, zero_head=False, num_classes=num_classes)
        model.load_state_dict(torch.load(ckpt_path))
        model.eval()
        model.to(args.device)

        # set bit width related params
        maxq = torch.tensor(2**bw - 1)
        for name, param in tqdm(model.named_parameters(), desc='params', ncols=100):
            # if 'attn.' in name or 'ffn.' in name or 'head.' in name or 'patch_embeddings' in name:
            if 'attn.' in name or 'head.' in name or 'patch_embeddings' in name:
                wmat = param.data
                # if 'bias' in name:
                if len(wmat.shape) == 1:
                    continue
                    wmat = wmat.view(-1,1)
                elif 'patch_embeddings.weight' in name:
                    wmat = wmat.view(768,-1)
                
                wmean = wmat.mean(dim=1, keepdim=True)
                wmat = wmat - wmean
                tff_n = wmat.shape[0]
                projs = torch.matmul(tffs[tff_n], wmat)
                num_frames = projs.shape[0]*projs.shape[1]
                # quantize the projs
                xmax = projs.max()
                xmin = projs.min()
                scale = (xmax - xmin) / maxq
                zero = torch.round(-xmin / scale)
                projs_qntzd = quantize_qfna(projs, scale, zero, maxq) 
                if args.coef_est_type == 'weiner':
                    # compute the weiner filter
                    z = projs_qntzd.view(num_frames, -1)
                    p = projs.view(num_frames, -1)
                    n = p-z
                    var_x = (wmat**2).mean(dim=1)
                    var_n = (n**2).mean(dim=1)
                    Rxz = torch.diag(var_x) @ tffs[tff_n].view(-1,tff_n).T
                    Rzz = tffs[tff_n].view(-1,tff_n) @ torch.diag(var_x) @ tffs[tff_n].view(-1,tff_n).T + var_n
                    wmat_est = (Rxz @ torch.linalg.pinv(Rzz) @ z).squeeze()
                elif args.coef_est_type == 'naive':
                    wmat_est = (tffs[tff_n].view(-1,tff_n).T @ projs_qntzd.view(num_frames, -1)).squeeze()

                # add the mean back
                wmat_est += wmean

                if 'patch_embeddings.weight' in name:
                    wmat_est = wmat_est.view(768,3,16,16)
                param.data = wmat_est

        val_acc = valid(args, model, writer=None, test_loader=test_loader, global_step=0)
        val_accs_r.append(val_acc)

        compression = 32/(redundancy*bw)
        compressions_r.append(compression)
    val_accs[redundancy] = val_accs_r
    compressions[redundancy] = compressions_r

    weiner_full_results = {'redundancies':redundancies, 'bws': bws, 'val_accs':val_accs, 'compressions':compressions, 'args': args}
    torch.save(weiner_full_results, os.path.join(directory_path, f'{exp_name}.pt'))

    for _r in val_accs.keys():
        plt.plot(bws, val_accs[_r])
    plt.legend(val_accs.keys())
    plt.savefig(os.path.join(directory_path, f'{exp_name}_accs.png'))
    plt.close()

    for _r in compressions.keys():
        plt.plot(bws, compressions[_r])
    plt.legend(compressions.keys())
    plt.savefig(os.path.join(directory_path, f'{exp_name}_cmprs.png'))
    plt.close()


# save the results
weiner_full_results = {'redundancies':redundancies, 'bws': bws, 'val_accs':val_accs, 'compressions':compressions, 'args': args}
torch.save(weiner_full_results, os.path.join(directory_path, f'{exp_name}.pt'))

for redundancy in redundancies:
    plt.plot(bws, val_accs[redundancy])
plt.legend(redundancies)
plt.savefig(os.path.join(directory_path, f'{exp_name}_accs.png'))
plt.close()

for redundancy in redundancies:
    plt.plot(bws, compressions[redundancy])
plt.legend(redundancies)
plt.savefig(os.path.join(directory_path, f'{exp_name}_cmprs.png'))
plt.close()

