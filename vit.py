import torch
from datetime import timedelta
import datetime
import time
import logging
import argparse
import os
import random
import numpy as np
from utils.data_utils import get_loader
from utils.dist_util import get_world_size
from utils.train_utils import AverageMeter, count_parameters, simple_accuracy

from models.modeling import VisionTransformer, CONFIGS

from quantize.datautils import *
from quantize.gptq import *
from quantize.modelutils import *
from quantize.quant import *
from utils.construct_tff import construct_real_tff

from tqdm import tqdm


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

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

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    model.load_state_dict(torch.load(args.ckpt_path))    
    model.to(args.device)
    num_params = count_parameters(model)

    logging.info("{}".format(config))
    logging.info("Training parameters %s", args)
    logging.info("Total Parameter: \t%2.1fM" % num_params)
    print(f'{num_params = }M')
    return args, model

def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logging.info("***** Running Validation *****")
    logging.info("  Num steps = %d", len(test_loader))
    logging.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logging.info("\n")
    logging.info("Validation Results")
    logging.info("Global Steps: %d" % global_step)
    logging.info("Valid Loss: %2.5f" % eval_losses.avg)
    logging.info("Valid Accuracy: %2.5f" % accuracy)

    if writer is not None:
        print('writing')
        writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy

@torch.no_grad()
def quantize_vit(model, dataloader, dev, args):
    print('Starting ...')

    layers = model.transformer.encoder.layer

    for batch in dataloader:
        inps = model.transformer.embeddings(batch[0].to(device))
        break

    tffs = {}
    k_attn = int(96 * args.tff_redundancy)
    l_attn = 8
    n_attn = 768
    tffs[n_attn] = construct_real_tff(k_attn, l_attn // 2, n_attn // 2).to(dev)

    k_mlp = int(384 * args.tff_redundancy)
    l_mlp = 8
    n_mlp = 3072
    tffs[n_mlp] = construct_real_tff(k_mlp, l_mlp // 2, n_mlp // 2).to(dev)

    outs = torch.zeros_like(inps)
    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(
                args.wbits, perchannel=True, sym=args.sym, mse=False, trits=args.trits
            )
            tff_n = subset[name].weight.shape[0]
            gptq[name].tff = tffs[tff_n].view(-1, tff_n)
            gptq[name].inps = []

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.train_batch_size):
            outs[j] = layer(inps[j].unsqueeze(0))[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(i, name)
            print(f'Quantizing {name} ...')
            gptq[name].fasterquant(
                percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
            )
            quantizers['model.decoder.layers.%d.%s' % (i, name)] = gptq[name].quantizer
            gptq[name].free()
        outs = layer(inps)[0]
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    return quantizers

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
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
    parser.add_argument("--train_batch_size", type=int, default=128,
                        help="training batch size; Here it serves as nsamples as well")
    parser.add_argument("--eval_batch_size", type=int, default=128,
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
    parser.add_argument("--parent_dir", type=str, default=None, 
                        help="parent dir for storing the results")
    # GPTQ params
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, 
        help='#bits to use for quantization; use 16 for evaluating base model.' # choices=[2, 3, 4, 16],
    )
    parser.add_argument(
        '--trits', action='store_true',
        help='Whether to use trits for quantization.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--load', type=str, default='',
        help='Load quantized model.'
    )
    parser.add_argument(
        '--benchmark', type=int, default=0,
        help='Number of tokens to use for benchmarking.'
    )
    parser.add_argument(
        '--check', action='store_true',
        help='Whether to compute perplexity during benchmarking for verification.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--faster-kernel', action='store_true',
        help='Whether to use the new faster kernel for benchmarking.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )
    # redundancy parameters
    parser.add_argument('--tff_redundancy', type=int, default=1,
                        help="Redundancy in tffs")

    args = parser.parse_args()

    exp_name = args.exp_name # 'mlp_attn_quant_weiner_full'
    if exp_name != 'debug_thread':
        if args.save_path is None:
            current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            directory_path = os.path.join("output", f'{args.parent_dir}', f'{exp_name}_{current_datetime}')
            args.save_path = directory_path
        else:
            directory_path = args.save_path

        os.makedirs(directory_path, exist_ok=True)

        logging.basicConfig(filename= os.path.join(directory_path, 'log.log'), level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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

    print('args = ')
    logging.info('args = ')
    for k,v in vars(args).items():
        print(f'{k}: {v}')
        logging.info(f'{k}: {v}')

    # Model & Tokenizer Setup
    args, model = setup(args)
    model.eval()

    # Prepare dataset
    train_loader, test_loader = get_loader(args)
    val_acc = valid(args, model, writer=None, test_loader=test_loader, global_step=0)
    print(f'initial val acc = {val_acc}')
    logging.info(f'initial val acc = {val_acc}')

    orig_wt = model.transformer.encoder.layer[0].ffn.fc1.weight.data.clone()
    if args.wbits <= 16 and not args.nearest:
        tick = time.time()
        quantizers = quantize_vit(model, train_loader, args.device, args)

    print(f'{orig_wt = }')
    print(f'{model.transformer.encoder.layer[0].ffn.fc1.weight.data = }')

    val_acc = valid(args, model, writer=None, test_loader=test_loader, global_step=0)
    print(f'final val acc = {val_acc}')
    logging.info(f'final val acc = {val_acc}')

    if exp_name != 'debug_thread':
        torch.save(model.state_dict(), os.path.join(directory_path, 'Qmodel.pth'))