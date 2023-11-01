import torch
from lookupFFN.lookup import BH4

if __name__ == '__main__':
    test_bh4 = BH4(in_dim=768, out_dim=768, block_size=64, decay_coeff = 0.7)
    breakpoint()