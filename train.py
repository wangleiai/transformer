from new_process import createDataSet, getData, preparedData, vocab_to_json
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


SRC, TRG,  train_iter, test_iter = preparedData(params.data_path, params.batch_size)
src_pad = SRC.vocab.stoi['<pad>']
trg_pad = TRG.vocab.stoi['<pad>']
model = Transformer(len(SRC.vocab), len(TRG.vocab), params.d_model, params.n_layers, params.heads, params.dropout)
if params.is_cuda:
    model = model.cuda()

# print(model)
print('trg_vocal_len: ', len(TRG.vocab))
print('src_vocab_len: ', len(SRC.vocab))

vocab_to_json(TRG.vocab, params.word_json_file, params.trg_lang)
vocab_to_json(SRC.vocab, params.word_json_file, params.src_lang)
print("write data to json finished !")

# optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.98), eps=1e-9)
optimizer = ScheduledOptim(
    torch.optim.Adam(
        model.parameters(),
        lr=params.lr,
        betas=(0.9, 0.98), eps=1e-09),
    params.d_model, params.n_warmup_steps)
performance = Performance(len(TRG.vocab), trg_pad, is_smooth=params.is_label_smooth)
print("\nbegin training model")
if os.path.exists("models/tfs.pkl"):
    print('load previous trained model')
    model.load_state_dict(torch.load("models/tfs.pkl"))

best_loss = None
train_global_steps = 0

writer = SummaryWriter()
for epoch in range(params.epochs):
    start = time.time()
    total_loss = 0.0
    step = 0

    train_n_word_total = 0
    train_n_word_correct = 0

    test_n_word_total = 0
    test_n_word_correct = 0
    model.train()
    for i, batch in enumerate(train_iter):
        # print(batch)
        src = batch.src.transpose(0, 1)
        trg = batch.trg.transpose(0, 1)

        trg_input = trg[:, :-1]
        src_mask, trg_mask = mask.createMask(src, src_pad, trg_input, trg_pad)

        preds = model(src, trg_input, src_mask, trg_mask)
        # _, pred = torch.max(preds, 2)
        # print(preds.size())
        # print(pred.size())
        # print(pred[0, :].size())
        # print(trg[0,1:].size())
        # print(pred[0,:].cpu().numpy())
        # print(trg[0,1:].cpu().numpy())
        # exit(1)
        # ys = trg[:, 1:].contiguous().view(-1)
        # print(trg.size())
        ys = trg[:, 1:].contiguous()
        # print(ys.size())

        if params.is_cuda:
            params = ys.cuda()

        optimizer.zero_grad()
        # loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=trg_pad)
        loss, n_correct = performance.get_performace(preds, ys)
        loss.backward()
        optimizer.step_and_update_lr()
        # optimizer.step()
        total_loss += loss.item()
        non_pad_mask = ys.ne(trg_pad)
        n_word = non_pad_mask.sum().item()
        train_n_word_total += n_word
        train_n_word_correct += n_correct
        step = i+1
        writer.add_scalar('steps/train_loss', loss.item(), train_global_steps)
        train_global_steps += 1
    train_loss_per_word = total_loss / train_n_word_total
    train_accuracy = train_n_word_correct / train_n_word_total
        # print(src.size())
        # print(trg.size())
        # print(trg_input.size())
        # print(src_mask.size())
        # print(trg_mask.size())
        # print(preds.size())
        # print(ys.size())
        # exit(1)
        # print(loss.item())
        # break
    test_total_loss = 0.0
    test_total_step = 0

    model.eval()
    for i, batch in enumerate(test_iter):
        # print(batch)
        src = batch.src.transpose(0, 1)
        trg = batch.trg.transpose(0, 1)

        trg_input = trg[:, :-1]
        src_mask, trg_mask = mask.createMask(src, src_pad, trg_input, trg_pad)

        preds = model(src, trg_input, src_mask, trg_mask)
        # _, pred = torch.max(preds, 2)
        # print(pred[0, :].cpu().numpy())
        # print(trg[0, 1:].cpu().numpy())
        # exit(1)
        # ys = trg[:, 1:].contiguous().view(-1)
        ys = trg[:, 1:].contiguous()
        if params.is_cuda:
            params = ys.cuda()
        loss, n_correct = performance.get_performace(preds, ys)

        test_total_loss += loss.item()
        non_pad_mask = ys.ne(trg_pad)
        n_word = non_pad_mask.sum().item()
        test_n_word_total += n_word
        test_n_word_correct += n_correct
        # loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=trg_pad)

        test_total_loss += loss.item()
        test_total_step = i + 1
        # print(src.size())
        # print(trg.size())
        # print(trg_input.size())
        # print(src_mask.size())
        # print(trg_mask.size())
        # print(preds.size())
        # print(ys.size())
        # exit(1)
        # print(loss.item())

        # break
    test_loss_per_word = test_total_loss / test_n_word_total
    test_accuracy = test_n_word_correct / test_n_word_total

    writer.add_scalar('scalar/train_loss', train_loss_per_word, epoch)
    writer.add_scalar('scalar/train_acc', train_accuracy, epoch)

    writer.add_scalar('scalar/test_loss', test_loss_per_word, epoch)
    writer.add_scalar('scalar/test_acc', test_accuracy, epoch)

    print("epoch:{} train_loss:{} train_acc:{:3f}  test_loss:{} test_acc:{:3f} time:{:.2f}".format(epoch+1,
                                                                                                  train_loss_per_word,
                                                                                                  train_accuracy, test_loss_per_word,
                                                                                                  test_accuracy, time.time()-start))
    # print("epoch:{} train_loss:{} test_loss:{} time:{:.2f}".format(epoch+1, total_loss/step, test_total_loss/test_total_step, time.time()-start))
    if best_loss is None:
        best_loss = test_total_loss/test_total_step
        torch.save(model.state_dict(), "models/tfs.pkl")
    if best_loss > test_total_loss/test_total_step :
        best_loss = test_total_loss/test_total_step
        torch.save(model.state_dict(), "models/tfs.pkl")

print("training end!")


