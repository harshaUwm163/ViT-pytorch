import torch
import torch.nn as nn
import math
import numpy as np
from lookupFFN.lookup import BH4

if __name__ == "__main__":
    n = 768
    dev = torch.device('cuda')
    bh4 = BH4(in_dim=n, out_dim=n, block_size=64, decay_coeff = 0.7)
    bh4_lr = 3e-2
    Weiner_F = torch.rand(n, n).to(dev)
    Q = torch.rand(n,2*n).to(dev)
    true_outs = (Weiner_F @ Q).T
    # do the bh4 approximation
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(bh4.parameters(),
                            lr=bh4_lr,
                            momentum=0.9,
                            weight_decay=0)
    bh4 = bh4.to(dev)
    bh4.zero_grad()
    bh4.train()
    for epoch in range(10):
        outs = bh4(Q.T)
        loss = criterion(true_outs, outs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f'{epoch = }, loss = {loss.item()}')