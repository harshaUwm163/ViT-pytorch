from train import setup,valid, set_seed
from apex import amp
import torch
import argparse
from utils.construct_tff import construct_real_tff
import matplotlib.pyplot as plt
from utils.data_utils import get_loader
import os
from datetime import timedelta, datetime
import numpy as np

from utils.func_utils import PriorityQueue

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

    # this model is for comparing the val acc on the right hand side for the heuristic method
    model_2 = torch.clone(model)
    
    train_loader, test_loader = get_loader(args)

    # TODO- change the k,l values
    k_attn = 64
    l_attn = 384
    n_attn = 768
    # TODO- transpose the Frames
    tffs = construct_real_tff(k_attn, l_attn // 2, n_attn // 2).permute(0,2,1).to(args.device)

    with torch.no_grad():
        tffs_weights_dict = {}
        tffs_enabled_frames = {}
        tffs_disabled_frames = {}
        tffs_rank_ki = {}
        for (name, param) in model.named_parameters():
            if 'attn' in name:
                if 'weight' in name:
                    projs = torch.matmul(tffs.permute(0,2,1), param)
                    tffs_weights_dict[name] = param.data.clone()
                    tffs_enabled_frames[name] = []
                    tffs_rank_ki[name] = []
                    tffs_disabled_frames[name] = PriorityQueue()

        # populate the queue
        # TODO- maybe need to to k_attn * l_attn here
        for k_i in range(k_attn):
            for name, param in model.named_parameters():
                if 'attn' in name:
                    if 'weight' in name:
                        frames_cat = tffs[k_i]
                        param.data = frames_cat @ torch.linalg.lstsq(frames_cat, tffs_weights_dict[name])[0] # add all the best ones and the new one
            val_acc = valid(args, model, writer=None, test_loader=test_loader, global_step=0)
            tffs_disabled_frames[name].push((k_i, val_acc))

        # pop the frame with the best accuracy and enable it
        for name, param in model.named_parameters():
            if 'attn' in name:
                if 'weight' in name:
                    frame_idx,val_acc = tffs_disabled_frames[name].pop()
                    tffs_enabled_frames[name].append(frame_idx)

        num_disabled_frames = k_attn - 1
        opt_res_dict = [val_acc]
        curr_frames = {}
        timeout = 2*k_attn # TODO- multi with l_attn here
        # add the last one directly
        while num_disabled_frames > 1 and timeout > 0:
            for name, param in model.named_parameters():
                if 'attn' in name and 'weight' in name:
                    curr_frame_id,_ = tffs_disabled_frames[name].pop()
                    next_frame_id,_ = tffs_disabled_frames[name].top()
                    curr_frames[name] = curr_frame_id
                    # enabled frames are the same for both the models
                    frames_enabled = tffs[tffs_enabled_frames[name]].permute(0,2,1).view(-1,n_attn).permute(1,0)
                    # update the param of the model
                    frames_cat = torch.cat((frames_enabled, tffs[curr_frame_id]), dim=1)
                    param.data = frames_cat @ torch.linalg.lstsq(frames_cat, tffs_weights_dict[name])[0] # add all the best ones and the new one
                    # update the param of the comparision model
                    frames_cat = torch.cat((frames_enabled, tffs[next_frame_id]), dim=1)
                    param2 = model_2.get_parameter(name)
                    param2.data = frames_cat @ torch.linalg.lstsq(frames_cat, tffs_weights_dict[name])[0] # add all the best ones and the new one
            
            # get the val accuracies for both the models
            val_acc = valid(args, model, writer=None, test_loader=test_loader, global_step=0)
            val_acc2 = valid(args, model_2, writer=None, test_loader=test_loader, global_step=0)
            # check if the val accuracy with current frame is better than next one
            if val_acc >= val_acc2:
                # add the current frames to the list of frames
                for name, param in model.named_parameters():
                    if 'attn' in name and 'weight' in name:
                        tffs_enabled_frames[name].append(curr_frames[name])
                num_disabled_frames -= 1
                opt_res_dict.append(val_acc)

                # logging
                mdict = {'k_attn': k_attn,
                        'l_attn': l_attn,
                        'n_attn': n_attn,
                        'args': args,
                        'tffs_enabled_frames': tffs_enabled_frames,
                        'tffs_disabled_frames': tffs_disabled_frames,
                        'opt_res_dict' : opt_res_dict,
                        }
                torch.save(mdict, os.path.join(args.output_dir, 'results.pt'))
                plt.plot(opt_res_dict)
                plt.savefig(os.path.join(args.output_dir, f'valAcc_vs_ki.png'))
                plt.close()
            else:
                # add the current frames back to the list with updated accuracies
                for name, param in model.named_parameters():
                    if 'attn' in name and 'weight' in name:
                        tffs_disabled_frames[name].append((curr_frames[name], val_acc))

            # update the counters
            timeout -= 1

        # append the last frame if the algo hasn't timed out yet
        if num_disabled_frames == 1:
            for name, param in model.named_parameters():
                if 'attn' in name and 'weight' in name:
                    curr_frame_id,_ = tffs_disabled_frames[name].pop()
                    tffs_enabled_frames[name].append(curr_frame_id)
                    # update the param of the model
                    frames_enabled = tffs[tffs_enabled_frames[name]].permute(0,2,1).view(-1,n_attn).permute(1,0)
                    frames_cat = torch.cat((frames_enabled, tffs[curr_frame_id]), dim=1)
                    param.data = frames_cat @ torch.linalg.lstsq(frames_cat, tffs_weights_dict[name])[0] # add all the best ones and the new one
            val_acc = valid(args, model, writer=None, test_loader=test_loader, global_step=0)
            opt_res_dict.append(val_acc)

            mdict = {   'k_attn': k_attn,
                        'l_attn': l_attn,
                        'n_attn': n_attn,
                        'args': args,
                        'tffs_enabled_frames': tffs_enabled_frames,
                        'tffs_disabled_frames': tffs_disabled_frames,
                        'opt_res_dict' : opt_res_dict,
                 }
            torch.save(mdict, os.path.join(args.output_dir, 'final_res.pt'))

    breakpoint()
        

if __name__ == '__main__':
    main()
