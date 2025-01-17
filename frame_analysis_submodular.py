from models.modeling import VisionTransformer, CONFIGS
from train import valid, set_seed
from apex import amp
import torch
import argparse
from utils.construct_tff import construct_real_tff
import matplotlib.pyplot as plt
from utils.data_utils import get_loader
import os
from datetime import timedelta, datetime
import numpy as np

def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "cifar100":
        num_classes = 100
    elif 'dogs' in args.dataset:
        num_classes = 10
    elif 'trucks' in args.dataset:
        num_classes = 6
    elif 'cats' in args.dataset:
        num_classes = 6
    elif 'birds' in args.dataset:
        num_classes = 10
    elif 'snakes' in args.dataset:
        num_classes = 10

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes, vis=True)
    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--dataset_dir", type=str, default="data/inet1k_classes/dogs",
                        help="Where to load the dataset from")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    parser.add_argument("--save_every", default=500, type=int,
                        help="save the checkpoint every few steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help="Path to the trained model ckpt")
    args = parser.parse_args()
    args.output_dir = f"{args.output_dir}/{args.name}-frame_analysis-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    os.makedirs(args.output_dir, exist_ok=True)

    print('running frame analysis python script')

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
        print(f'{args.n_gpu = }')
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Set seed
    set_seed(args)

    # TODO- change the model to DDP
    args, model = setup(args)
    if args.fp16:
        model = amp.initialize( models=model,
                                opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20
    if args.ckpt_path is not None:
        model.load_state_dict(torch.load(args.ckpt_path))
    
    train_loader, test_loader = get_loader(args)

    # TODO- change the k,l values
    k_attn = 128
    l_attn = 48
    n_attn = 768
    tffs = construct_real_tff(k_attn, l_attn // 2, n_attn // 2).permute(0,2,1).to(args.device)
    print('At the breakpoint')
    val_acc = valid(args, model, writer=None, test_loader=test_loader, global_step=0)
    breakpoint()

    with torch.no_grad():
        tffs_weights_dict = {}
        tffs_enabled_frames = {}
        tffs_disabled_frames = {}
        for (name, param) in model.named_parameters():
            if 'attn' in name:
                if 'weight' in name:
                    projs = torch.matmul(tffs.permute(0,2,1), param)
                    tffs_weights_dict[name] = param.data.clone()
                    tffs_enabled_frames[name] = []
                    tffs_disabled_frames[name] = list(range(k_attn))

        num_disabled_frames = k_attn
        opt_res_dict = {}
        avg_norm_ki = []
        avg_rank_ki = []
        for k_i in range(k_attn):
            print(f'added frame {k_i}/{k_attn}')
            # run the submodular optimization
            val_accs = []
            for v_i in range(num_disabled_frames):
                for name, param in model.named_parameters():
                    if 'attn' in name:
                        if 'weight' in name:
                            if k_i != 0:
                                frames_enabled = tffs[tffs_enabled_frames[name]].permute(0,2,1).view(-1,n_attn).permute(1,0)
                                frames_cat = torch.cat((frames_enabled, tffs[tffs_disabled_frames[name][v_i]]), dim=1)
                            else:
                                frames_cat = tffs[tffs_disabled_frames[name][v_i]]
                            param.data = frames_cat @ torch.linalg.lstsq(frames_cat, tffs_weights_dict[name])[0] # add all the best ones and the new one
                val_acc = valid(args, model, writer=None, test_loader=test_loader, global_step=0)
                val_accs.append(val_acc)

            # get the max valid acc
            va_max = np.max(val_accs)
            va_amax = np.argmax(val_accs)

            # avg norm and avg rank
            avg_norm = 0.
            avg_rank = 0.
            num_params = 0
            # update the used and unused frames with that particular index
            for name, params in model.named_parameters():
                if 'attn' in name:
                    if 'weight' in name:
                        rem_idx = tffs_disabled_frames[name].pop(va_amax)
                        tffs_enabled_frames[name].append(rem_idx)
                        frames_enabled = tffs[tffs_enabled_frames[name]].permute(0,2,1).view(-1,n_attn).permute(1,0)

                        proj_data = frames_cat @ torch.linalg.lstsq(frames_cat, tffs_weights_dict[name])[0] # add all the best ones and the new one
                        avg_rank += torch.linalg.matrix_rank(frames_enabled)
                        avg_norm += torch.norm(proj_data)
                        num_params += 1
            avg_norm_ki.append(avg_norm/ num_params)
            avg_rank_ki.append(avg_rank/ num_params)
            # we are updating one frame at a time, so, decrement the num_disabled_frames by 1
            num_disabled_frames -= 1

            # book keeping and repeat
            opt_res_dict[k_i] = va_max
            print(f'{k_i = }, {va_max = }')
            print(f'{k_i = }, {avg_norm_ki[-1] = }')
            print(f'{k_i = }, {avg_rank_ki[-1] = }')

            # logging
            mdict = {'k_attn': k_attn,
                    'l_attn': l_attn,
                    'n_attn': n_attn,
                    'args': args,
                    'tffs_enabled_frames': tffs_enabled_frames,
                    'tffs_disabled_frames': tffs_disabled_frames,
                    'opt_res_dict' : opt_res_dict,
                    'val_accs' : val_accs, 
                    'avg_norm_ki': avg_norm_ki,
                    'avg_rank_ki': avg_rank_ki
                    }
            torch.save(mdict, os.path.join(args.output_dir, 'results.pt'))
            plt.plot(opt_res_dict.keys(), opt_res_dict.values())
            plt.savefig(os.path.join(args.output_dir, f'valAcc_vs_ki.png'))
            plt.close()

        # final_model
        for name, param in model.named_parameters():
            if 'attn' in name:
                if 'weight' in name:
                        frames_cat = tffs[tffs_enabled_frames[name]].permute(0,2,1).view(-1,n_attn).permute(1,0)
                        param.data = frames_cat @ torch.linalg.lstsq(frames_cat, tffs_weights_dict[name])[0] # add all the best ones and the new one
        val_acc = valid(args, model, writer=None, test_loader=test_loader, global_step=0)
        print(f'final validation accuracy = {val_acc}')

        mdict = {'k_attn': k_attn,
                 'l_attn': l_attn,
                 'n_attn': n_attn,
                 'args': args,
                 'tffs_enabled_frames': tffs_enabled_frames,
                 'tffs_disabled_frames': tffs_disabled_frames,
                 'opt_res_dict' : opt_res_dict,
                 'avg_norm_ki': avg_norm_ki,
                 'avg_rank_ki': avg_rank_ki
                 }
        torch.save(mdict, os.path.join(args.output_dir, 'final_res.pt'))

    breakpoint()
        

if __name__ == '__main__':
    main()
