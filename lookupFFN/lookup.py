import torch.nn as nn
import torch
import math
import numpy as np
from .bh4.kernel import bh4
import time

class BH4(nn.Module):
    def __init__(self, in_dim, out_dim, block_size, decay_coeff = 0.7):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.block_size = block_size
        self.decay_coeff = decay_coeff

        self.padded_in_dim = int(2 ** math.ceil(math.log2(in_dim)))
        self.num_repeat = max(1, int(math.ceil(self.out_dim / self.padded_in_dim)))
        self.num_block = self.padded_in_dim // self.block_size
        
        coeff = math.sqrt(self.block_size * self.padded_in_dim)
        self.weight = nn.Parameter(torch.randn(self.num_repeat, 4, self.num_block, self.block_size, self.block_size) / coeff)
        # self.bias = nn.Parameter(torch.zeros(self.out_dim))

    def extra_repr(self):
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}, block_size={self.block_size}, decay_coeff={self.decay_coeff}"

    def _forward(self, x):
        
        batch_size, dim = x.shape
        if dim < self.padded_in_dim:
            padding_dim = self.padded_in_dim - dim
            x = torch.cat([x, torch.zeros(batch_size, padding_dim, dtype = x.dtype, device = x.device)], dim = -1).contiguous()

        x = self.decay_coeff * bh4(x, self.weight, training = self.training).squeeze() + (1 - self.decay_coeff) * x.repeat(1, self.num_repeat)
        x = x[:, :self.out_dim].contiguous() #  + self.bias
        
        return x

    def forward(self, xs):
        shape = xs.shape[:-1]
        dim = xs.shape[-1]
        xs = xs.reshape(np.prod(shape).item(), dim)
        outputs = self._forward(xs)
        outputs = outputs.reshape(*shape, self.out_dim)
        return outputs