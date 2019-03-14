import params
from torchtext import data
import torch
import time
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import os

def nopeakMask(size):
    # 半角矩阵
    np_mask = np.triu(np.ones((1, size, size)),
    k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0)
    if params.is_cuda:
        np_mask = np_mask.cuda()
    return np_mask

def createMask(src, src_pad, trg, trg_pad):
    src_mask = (src !=src_pad).unsqueeze(-2)
    if trg is not None:
        trg_mask = (trg != trg_pad).unsqueeze(-2)
        size = trg.size(-1)
        # print("size: ", size)
        np_mask = nopeakMask(size)
        if params.is_cuda:
            trg_mask = trg_mask.cuda()
            np_mask = np_mask.cuda()
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None
    return src_mask, trg_mask