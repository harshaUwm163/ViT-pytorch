from train import *
from addict import Dict
import logging
from utils.construct_tff import construct_real_tff
import matplotlib.pyplot as plt
from utils.data_utils import get_loader

logger = logging.getLogger(__name__)

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
    if args.ckpt_path is not None:
        model.load_state_dict(torch.load(args.ckpt_path))
    
    train_loader, test_loader = get_loader(args)

    # TODO- change the k,l values
    k_attn = 2
    l_attn = 6144
    n_attn = 768
    # TODO- transpose the Frames
    tffs = construct_real_tff(k_attn, l_attn // 2, n_attn // 2).to(args.device)
    breakpoint()

    with torch.no_grad():
        tffs_projs_dict = {}
        tffs_enabled_frames = {}
        tffs_disabled_frames = {}
        for (name, param) in model.named_parameters():
            if 'attn' in name:
                if 'weight' in name:
                    projs = torch.matmul(tffs, param.weight)
                    tffs_projs_dict[name] = projs
                    tffs_enabled_frames[name] = []
                    tffs_disabled_frames[name] = list(range(k_attn))

        num_disabled_frames = k_attn
        opt_res_dict = {}
        for k_i in range(k_attn):
            # run the submodular optimization
            val_accs = []
            for v_i in range(num_disabled_frames):
                for name, params in model.named_parameters():
                    if 'attn' in name:
                        if 'weight' in name:
                                frames_enabled = tffs_enabled_frames[name]
                                enabled_frames_contrib = torch.matmul(tffs[frames_enabled], tffs_projs_dict[name]).sum(0)
                                disabled_frames_contrib = torch.matmul(tffs[tffs_disabled_frames[name][v_i]], tffs_projs_dict[name][v_i])
                                param.weight.data = enabled_frames_contrib + disabled_frames_contrib # add all the best ones and the new one
                val_acc = valid(args, model, writer=None, test_loader=test_loader, global_step=0)
                val_accs.append(val_acc)

            # get the max valid acc
            va_max = np.max(val_accs)
            va_amax = np.argmax(val_accs)
            # update the used and unused frames with that particular index
            for name, params in model.named_parameters():
                if 'attn' in name:
                    if 'weight' in name:
                        rem_idx = tffs_disabled_frames[name].pop(va_amax)
                        tffs_enabled_frames[name].append(rem_idx)
            # we are updating one frame at a time, so, decrement the num_disabled_frames by 1
            num_disabled_frames -= 1

            # book keeping and repeat
            opt_res_dict[k_i] = va_max

