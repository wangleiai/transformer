import os
import jieba
import nltk
import torch
from torchtext import data
import pandas as pd
import params
from sklearn.utils import shuffle
import json

DEVICE=torch.device('cuda:0')
BATCH_SIZE = 2
SRC_MIN_FREQ = 5
TRG_MIN_FREQ = 5

def getData(path):
    data = []
    with open(path, encoding='utf-8') as f:
        data = f.readlines()
    lang = []
    lang2 = []

    for idx, da in enumerate(data):
        #         print(da)
        d = da.split("\t")
        lang.append(d[0].strip())
        lang2.append(d[1].strip())
    # print(lang[0:3])
    #     print(lang2[0:3])
    return lang, lang2


def cn_tokenize(sentence):
    seg_list = jieba.cut(sentence)
#     print(", ".join(seg_list))
    return [toekn for toekn in  seg_list]
# cn_tokenize(cn_lang[307])

def en_tokenize(sentence):
    seg_list = nltk.word_tokenize(sentence)
    return [token for token in seg_list]
# en_tokenize(en_lang[307])

def createDataSet(SRC, TRG, lang, lang2, batch_size):
    raw_data = {'src': lang, 'trg': lang2}
    df = pd.DataFrame(raw_data, columns=['src', 'trg'])

    if not os.path.exists("data/train.csv") or not os.path.exists("data/test.csv"):
        df = shuffle(df) # 打乱数据
        test_df = df[:params.test_nums] # 得到后test nums 个数据做测试
        train_df = df[params.test_nums:] # 剩下的做训练
        test_df.to_csv("data/test.csv", index=False, encoding='utf-8')
        train_df.to_csv("data/train.csv", index=False, encoding='utf-8')

    data_fields = [('src', SRC), ('trg', TRG)]
    train = data.TabularDataset('./data/train.csv', format='csv', fields=data_fields, skip_header=True)
    test = data.TabularDataset('./data/test.csv', format='csv', fields=data_fields, skip_header=True)

    if params.is_cuda:
        device = torch.device("cuda:0")
    else:
        device =torch.device("cpu:0")
    train_iter = data.Iterator(train, sort_key=lambda x: (len(x.src), len(x.trg)),
                                      batch_size=params.batch_size,device=device)
    test_iter = data.Iterator(test, sort_key=lambda x: (len(x.src), len(x.trg)),
                               batch_size=params.test_batch_size, device=device)

    SRC.build_vocab(train, min_freq=params.src_min_freq)
    TRG.build_vocab(train, min_freq=params.trg_min_freq)

    return train_iter, test_iter

def getFiled(path, batch_size=1):
    en_lang, cn_lang = getData(path)
    SRC = data.Field(lower=True, tokenize=en_tokenize, init_token='<sos>', eos_token='<eos>')
    # trg是中文
    TRG = data.Field(tokenize=cn_tokenize, eos_token='<eos>', init_token='<sos>')
    createDataSet(SRC, TRG, en_lang, cn_lang, batch_size)

    return SRC, TRG

def preparedData(path, batch_size):
    # path = "./data/en-cn.txt"
    en_lang, cn_lang = getData(path)
    SRC = data.Field(lower=True, tokenize=en_tokenize, init_token='<sos>', eos_token='<eos>')
    # trg是中文
    TRG = data.Field(tokenize=cn_tokenize, eos_token='<eos>', init_token='<sos>')
    createDataSet(SRC, TRG, en_lang, cn_lang, batch_size)
    train_iter, test_iter = createDataSet(SRC, TRG, en_lang, cn_lang, batch_size)
    return SRC, TRG, train_iter, test_iter

def vocab_to_json(vocab, path, lang, is_revserce=True):
    # print(vocab)
    word_dict = {}
    for i in range(len(vocab)):
        word_dict[i] = vocab.itos[i]

    save_path = os.path.join(path, lang+"_idx2word.json")
    with open(save_path, encoding='utf-8', mode='w') as f:
        json.dump(word_dict, f)

    if is_revserce:
        save_path = os.path.join(path, lang+"_word2idx.json")
        word_dict1 = {v: k for k, v in word_dict.items()}
        with open(save_path, encoding='utf-8', mode='w') as f:
            json.dump(word_dict1, f)

if __name__ == '__main__':

    path = "./data/en-cn.txt"
    en_lang, cn_lang = getData(path)
    SRC = data.Field(lower=True, tokenize=en_tokenize, init_token='<sos>', eos_token='<eos>')
    # trg是中文
    TRG = data.Field(tokenize=cn_tokenize, eos_token='<eos>', init_token='<sos>')

    train_iter, test_iter = createDataSet(SRC, TRG, en_lang, cn_lang, batch_size=1)
    i = 0
    for i, batch in enumerate(train_iter):
        print(batch.src.size())
        print(batch.src.transpose(0, 1))
        break
    # for i, bc in enumerate(test_iter):
    #     print(bc.src.size())
    #     print(bc.src.transpose(0, 1))
    #     for b in bc.src.transpose(0, 1)[0]:
    #         print(SRC.vocab.itos[b])
    #     break

    print(len(SRC.vocab))
    print(len(TRG.vocab))
    print(SRC.vocab.stoi['<sos>']) # 2
    print(SRC.vocab.stoi['<eos>']) # 3
    print(SRC.vocab.stoi['<pad>']) # 1
    print(SRC.vocab.stoi['<unk>']) # 0

    print(TRG.vocab.stoi['<sos>']) # 2
    print(TRG.vocab.stoi['<eos>']) # 3
    print(TRG.vocab.stoi['<pad>']) # 1
    print(TRG.vocab.stoi['<unk>']) # 0
    # TRG.vocab.itos[tok]
    # print(i*5)
    for i, batch in enumerate(test_iter):
        print(batch.trg.size())
        print(batch.trg.transpose(0, 1))
        break

    s, t = getFiled(path = "./data/en-cn.txt")
    print(len(s.vocab))
    print(len(t.vocab))