from new_process import createDataSet, getData, preparedData
import params
from model import Transformer
from torchtext import data
import torch
import nltk
import time
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import os
from tensorboardX import SummaryWriter
import mask
from performance import Performance

from my_optim import ScheduledOptim
from my_data import MyData


my_data = MyData(params.src_lang, params.trg_lang)




model = Transformer(len(my_data.src_word2idx), len(my_data.trg_word2idx), params.d_model, params.n_layers, params.heads, params.dropout)
if params.is_cuda:
    model = model.cuda()

# print(model)
print('trg_vocal_len: ', len(my_data.trg_word2idx))
print('src_vocab_len: ', len(my_data.src_word2idx))

torch.save(model.state_dict(), "models/tfs.pkl")
model.load_state_dict(torch.load('models/tfs.pkl'))
model.eval()


# print(batch)
# src = batch.src.transpose(0, 1)
# trg = batch.trg.transpose(0, 1)
src_sentence = 'He was brave.'
sentence = [i for i in nltk.word_tokenize(src_sentence)]
src = my_data.turn_to_idx(sentence, params.src_lang)
src = src.unsqueeze(0)

trg_sentence = ['<sos>']
trg_input = my_data.turn_to_idx(trg_sentence, params.trg_lang)
trg_input = trg_input.unsqueeze(0)

print(src.size())
print(trg_input.size())
# print(src)

# print(trg_input.size())

src_mask = None
trg_mask = None
for i in range(params.max_len):
    preds = model(src, trg_input, src_mask, trg_mask)
    # print("pred ", torch.max(preds, 2)[1])
    pred = torch.max(preds, 2)[1]
    trg_input = torch.cat((trg_input, pred[:, i].unsqueeze(1)), dim=1)
    # print(pred[:, i].unsqueeze(1))
# print("pred:", trg_input)

trg_sentence = my_data.batch_turn_to_word(trg_input, params.trg_lang)
print("src: ", src_sentence)
print('trg: ', trg_sentence)