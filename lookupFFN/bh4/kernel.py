
import torch
import torch.nn as nn
# from lookupFFN.bh4.cpu.kernel import BH4Function
from lookupFFN.bh4.fast_hadamard_transform.kernel import hadamard
import os
import time
import math

def bh4(x, w, training):
    if training:
        assert x.device.type == 'cuda'
        assert w.device.type == 'cuda'
        return bh4_cuda(x, w)
    else:
        if x.device.type == 'cuda':
            assert w.device.type == 'cuda'
            return bh4_cuda(x, w)
        # elif x.device == torch.device("cpu"):
        #     assert w.device == torch.device("cpu")
        #     return BH4Function.apply(x, w)
        else:
            raise NotImplementedError

def bh4_cuda(x, w):
    BS = w.shape[-1]
    B, D = x.shape
    NB = D // BS
    out = []
    for i in range(w.shape[0]):
        y = x
        for j in range(w.shape[1]):
            y = y.reshape(B, NB, BS)
            y = torch.einsum("bni,noi->bno", y, w[i, j])
            y = y.reshape(B, D)
            y = hadamard(y, True)
        out.append(y)
    return torch.stack(out, dim = -1)

# if __name__ == "__main__":
#     unit_test()
#     autograd_unit_test()
#     profile()