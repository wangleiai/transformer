import os
import jieba
import nltk
import params
import tqdm
import time

ai_challege_path = "/home/wl/Desktop/ai_challenger_MTEnglishtoChinese_trainingset_20180827.txt"
data_save_path = "data/en_cn.txt"
num_lines = 3000000

data_save = open(data_save_path, mode='w', encoding='utf-8')

start  = time.time()

with open(ai_challege_path, encoding='utf-8') as f:
    j = 0
    for i in f:
        line = i[6:].strip().split("\t")
        # print(line)
        line[0] = line[0].replace("", "")
        en_len = len(nltk.word_tokenize(line[0]))
        cn_len = len([i for i in jieba.cut(line[1])])
        if en_len<=params.max_len and  cn_len<=params.max_len:
            data_save.write(line[0]+"\t"+line[1] + "\n")
            j += 1
            if j==num_lines:
                break
    f.close()
    print("the number of lines is" + str(j))
    print(time.time() - start)