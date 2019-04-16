from new_process import createDataSet, getData, preparedData
import params
from model import Transformer
from torchtext import data
import torch
import time
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import os
from tensorboardX import SummaryWriter
import mask
from performance import Performance
from my_optim import ScheduledOptim

SRC, TRG,  train_iter, test_iter = preparedData(params.data_path, params.eval_batch_size)
src_pad = SRC.vocab.stoi['<pad>']
trg_pad = TRG.vocab.stoi['<pad>']
model = Transformer(len(SRC.vocab), len(TRG.vocab), params.d_model, params.n_layers, params.heads, params.dropout)
if params.is_cuda:
    model = model.cuda()

# print(model)
print('trg_vocal_len: ', len(TRG.vocab))
print('src_vocab_len: ', len(SRC.vocab))

model.load_state_dict(torch.load('models/tfs.pkl'))
model.eval()

cnt = 10
for i, batch in enumerate(train_iter):
    # print(batch)
    src = batch.src.transpose(0, 1)
    trg = batch.trg.transpose(0, 1)
    print(src.size())
    print(trg.size())
    # print(src)
    print("trg: ", trg)
    trg_input = trg[:, 0].unsqueeze(1)
    # print(trg_input.size())

    src_mask = None
    trg_mask = None
    for i in range(params.max_len):
        preds = model(src, trg_input, src_mask, trg_mask)
        # print("pred ", torch.max(preds, 2)[1])
        pred = torch.max(preds, 2)[1]
        trg_input = torch.cat((trg_input, pred[:, i].unsqueeze(1)), dim=1)
        # print(pred[:, i].unsqueeze(1))
    print("pred:", trg_input)
    cnt -= 1
    if cnt==0:
        break
    # break



