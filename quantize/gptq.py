import math
import time

import torch
import torch.nn as nn
import transformers

from quantize.quant import *
from lookupFFN.lookup import BH4

DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.bh4 = BH4(in_dim=self.rows, out_dim=self.rows, block_size=16, decay_coeff = 0.7)
        self.bh4_lr = 30e-2

    # def preproc(self, preproc_gptqH=False, percdamp=.01,
    #             preproc_rescale=False, preproc_proj=False, preproc_proj_extra=0):
    #     """
    #     optional preprocessing: scales w,H diagonally, or random projection
    #     run gptqH last
    #     preproc_proj_extra:
    #     0: 2 factor butterfly + permute
    #     1: 2 factor butterfly + permute + no blocking (default)
    #     2: 2 factor butterfly + no permute
    #     3: random orthogonal
    #     """
    #     self.preproc_gptqH   = preproc_gptqH
    #     self.preproc_rescale = preproc_rescale
    #     self.preproc_proj    = preproc_proj
    #     if preproc_rescale:
    #         w = self.layer.weight.data.clone().to(torch.float32)
    #         H = self.H.to(torch.float32)
    #         H /= H.abs().max()
    #         diagH = torch.diag(H)
    #         diagW2 = torch.diag(w.T @ w)
    #         diagH = torch.clamp(diagH, min=1e-8)
    #         diagW2 = torch.clamp(diagW2, min=1e-8)
    #         scaleWH = (diagH / diagW2).sqrt().sqrt().to(torch.float32)
    #         scaleWH = scaleWH.clamp(min=1e-8)
    #         w *= scaleWH[None,:]
    #         H /= scaleWH[None,:]
    #         H /= scaleWH[:,None]
    #         w = w.to(torch.float32)
    #         scaleWH = scaleWH.to(torch.float32)
    #         self.scaleWH = scaleWH.cpu()
    #         self.layer.weight.data = w.to(self.layer.weight.data.dtype)
    #         self.H.data = H.to(self.H.data.dtype)
    #     if preproc_proj:
    #         w = self.layer.weight.data.clone().to(torch.float32)
    #         H = self.H.data.clone().to(torch.float32)
    #         # 
    #         if preproc_proj_extra == 0:
    #             U = rand_ortho_butterfly(w.shape[0]).to(torch.float32).to(w.device)
    #             V = rand_ortho_butterfly(w.shape[1]).to(torch.float32).to(w.device)
    #         elif preproc_proj_extra == 1:
    #             U = rand_ortho_butterfly_noblock(w.shape[0]).to(torch.float32).to(w.device)
    #             V = rand_ortho_butterfly_noblock(w.shape[1]).to(torch.float32).to(w.device)
    #         elif preproc_proj_extra == 2:
    #             U = rand_ortho_butterfly_nopermute(w.shape[0]).to(torch.float32).to(w.device)
    #             V = rand_ortho_butterfly_nopermute(w.shape[1]).to(torch.float32).to(w.device)
    #         #EH = torch.linalg.eigh(H)
    #         #H = (EH.eigenvectors @ torch.diag(EH.eigenvalues.relu() * H.shape[0] / (EH.eigenvalues.relu().sum() + 1e-8) + 1e-2) @ EH.eigenvectors.T).to(w.device)
    #         #H = H.to(torch.float32)
    #         H = H * (H.shape[0] / (torch.trace(H) + 1e-8)) + 1e-2 * torch.eye(H.shape[0], device=w.device)
    #         H = H.to(torch.float32)
    #         w = U @ w @ V.T
    #         H = V @ H @ V.T
    #         self.projU = U.cpu()
    #         self.projV = V.cpu()
    #         self.layer.weight.data = w.to(self.layer.weight.data.dtype)
    #         self.H.data = H.to(self.H.data.dtype)
    #     # H modification from gptq
    #     if self.preproc_gptqH:
    #         w = self.layer.weight.data.clone()
    #         H = self.H.data.clone()
    #         dead = torch.diag(H) == 0
    #         H[dead, dead] = 1
    #         w[:, dead] = 0
    #         damp = percdamp * torch.mean(torch.diag(H))
    #         diag = torch.arange(self.columns, device=self.dev)
    #         H[diag, diag] += damp
    #         self.layer.weight.data = w.to(self.layer.weight.data.dtype)
    #         self.H.data = H.to(self.H.data.dtype)
    #     self.preproc_done = True

    # we can vectorize this
    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.inps.append(inp.clone())
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        # # project onto TFF
        W = torch.matmul(self.tff, W)
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = quantize(
                    w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                ).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if actorder:
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        # # naive reconstruction
        # self.layer.weight.data = self.tff.T @ Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        # # using the full blown Weiner Filter
        Q = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        self.inps = torch.stack(self.inps, dim=0).permute(0,2,1)
        x = nn.functional.linear(self.inps, self.layer.weight.data).view(-1, self.rows).T
        z = nn.functional.linear(self.inps, Q).view(-1, self.rows).T
        Rxz = x @ z.T
        Rzz = z @ z.T
        Weiner_F = Rxz @ torch.linalg.pinv(Rzz)
        # do the bh4 approximation
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.bh4.parameters(),
                                lr=self.bh4_lr,
                                momentum=0.9,
                                weight_decay=0)
        self.bh4 = self.bh4.to(self.dev)
        self.bh4.zero_grad()
        self.bh4.train()
        true_outs = (Weiner_F @ Q).T
        for epoch in range(1):
            outs = self.bh4(Q.T)
            loss = criterion(true_outs, outs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f'{epoch = }, loss = {loss.item()}')

        self.bh4.eval()
        del optimizer ,criterion ,x ,z , Rxz, Rzz, Weiner_F
        torch.cuda.empty_cache()


        # Import matplotlib.pyplot as plt
        # Plt.plot(losses)
        # Plt.savefig('temp.png')
        # plt.close()
        # breakpoint()
        # using the full Weiner filter
        # self.layer.weight.data = Weiner_F @ Q
        # using the BH4 version
        self.layer.weight.data = self.bh4(Q.T).T
        # # using gptq alone
        # self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        # if DEBUG:
        #     print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
