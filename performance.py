import math
import time

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import params


class Performance():

    def __init__(self, n_class, pad_idx, is_smooth=False, eps=0.1):
        self.n_class = n_class
        self.is_smooth = is_smooth
        self.eps = eps
        self.pad_idx = pad_idx

    def cal_n_correct(self, preds, gold):
        _, pred = torch.max(preds, 2)
        # gold = gold.contiguous().view(-1)

        non_pad_mask = gold.ne(self.pad_idx)

        n_correct = pred.eq(gold)
        n_correct = n_correct.masked_select(non_pad_mask).sum().item()

        return n_correct
    
    def get_performace(self, preds, gold):
        loss = self.cal_loss(preds, gold)
        n_correct = self.cal_n_correct(preds, gold)
        return loss, n_correct



    def cal_loss(self, preds, gold):
        gold = gold.contiguous()
        if self.is_smooth:
            eps = self.eps
            n_class = self.n_class

            log_prb = F.log_softmax(preds, dim=-1)
            # print(preds.size())
            # print(gold.size())

            one_hot = torch.zeros(preds.size(0), preds.size(1), self.n_class)
            if params.is_cuda:
                one_hot = one_hot.cuda()
            one_hot = one_hot.scatter_(2, gold.unsqueeze(-1),  1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)

            non_pad_mask = gold.ne(self.pad_idx)
            loss = -(one_hot * log_prb).sum(dim=-1)
            loss = loss.masked_select(non_pad_mask).sum()  # average later

        else:
            _, pred = torch.max(preds, 2)
            # print(pred.size())
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), gold.view(-1), ignore_index=self.pad_idx, reduction='sum')
            # loss = F.cross_entropy(pred.float(), gold, ignore_index=self.pad_idx, reduction='sum')
        return loss

