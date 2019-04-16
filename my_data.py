import os
import json
import torch
import params
import numpy as np
import nltk

class MyData:
    def __init__(self, src, trg, reverse=True):
        self.src = src
        self.trg = trg
        self.reverse = reverse

        self.src_json_path = os.path.join("data", src+"_word2idx.json")
        self.trg_json_path = os.path.join("data", trg+"_word2idx.json")
        if reverse:
            self.src_reverse_path = os.path.join("data", src+"_idx2word.json")
            self.trg_reverse_path = os.path.join("data", trg+"_idx2word.json")

        self.src_word2idx = {}
        self.src_idx2word = {}
        self.trg_word2idx = {}
        self.trg_idx2word = {}

        self.load_dict()

    def load_dict(self):
        src_file = open(self.src_json_path, mode='r', encoding='utf-8')
        trg_file = open(self.trg_json_path, mode='r', encoding='utf-8')
        self.src_word2idx = json.load(src_file)
        self.trg_word2idx = json.load(trg_file)

        if self.reverse:
            src_reverse_file = open(self.src_reverse_path, mode='r', encoding='utf-8')
            trg_reverse_file = open(self.trg_reverse_path, mode='r', encoding='utf-8')

            self.src_idx2word = json.load(src_reverse_file)
            self.trg_idx2word = json.load(trg_reverse_file)
        print("load dict finished!")

    def turn_to_idx(self, sentence, lang):
        ''' sentence 是经过分词之后得到的句子，如[ 哈哈, !] '''
        idx = []
        if lang == self.src:
            if '<sos>' in self.src_word2idx.keys():
                idx.append(self.src_word2idx['<sos>'])

            for word in sentence:
                if word in self.src_word2idx.keys():
                    idx.append(self.src_word2idx[word])
                else:
                    idx.append(self.src_word2idx['<unk>'])

            if '<eos>' in self.src_word2idx.keys():
                idx.append(self.src_word2idx['<eos>'])

        elif lang ==self.trg:
            if '<sos>' in self.trg_word2idx.keys():
                idx.append(self.trg_word2idx['<sos>'])

            for word in sentence:
                idx.append(self.trg_word2idx[word])

            if '<eos>' in self.trg_word2idx.keys():
                idx.append(self.trg_word2idx['<eos>'])
        else:
            print("lang is neither src or trg!")
            exit(1)

        idx = np.array(idx)
        idx = torch.from_numpy(idx)
        if params.is_cuda:
            idx = idx.cuda()
        return idx.long()

    def turn_to_word(self, word_idxs, lang):
        # print(type(word_idxs), ' ', type([]))
        if type(word_idxs)!=type([]):
            word_idxs = word_idxs.cpu().numpy()
            word_idxs = list(word_idxs)


        words = []
        if lang == self.src:
            for idx in word_idxs:
                words.append(self.src_idx2word[idx])
        elif lang ==self.trg:

            for idx in word_idxs:
                idx = str(idx)
                words.append(self.trg_idx2word[idx])
        else:
            print("lang is neither src or trg!")
            exit(1)

        return words

    def get_src_len(self):
        return len(self.src_word2idx)

    def get_trg_len(self):
        return len(self.trg_word2idx)

    def batch_turn_to_idx(self, sentences, lang):
        idxes = []
        idxes = np.array(idxes)
        idxes = torch.from_numpy(idxes)
        if params.is_cuda:
            idxes = idxes.cuda()
        if lang == self.src or lang ==self.trg:
            for sentence in sentences:
                idx = self.turn_to_idx(sentence, lang)
                torch.cat((idxes, idx), 0)
        else:
            print("lang is neither src or trg!")
            exit(1)
        return idxes.long()


    def batch_turn_to_word(self, sentences, lang):
        sentences = sentences.cpu().numpy()
        sentences = list(sentences)
        sentences = [list(i) for i in sentences]
        words = []
        # words = np.array(words)
        # words = torch.from_numpy(words)
        # if params.is_cuda:
        #     words = words.cuda()
        if lang == self.src or lang ==self.trg:
            for sentence in sentences:
                word = self.turn_to_word(sentence, lang)
                words.append(word)
        else:
            print("lang is neither src or trg!")
            exit(1)
        return words

if __name__ == '__main__':
    data = MyData("en", "cn")
    print(data.get_src_len())
    print(data.get_trg_len())
    seg_list = nltk.word_tokenize("like you!")
    a = [i for i in seg_list]
    b = data.turn_to_idx(a, "en")
    print(b)