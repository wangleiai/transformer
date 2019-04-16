

data_path = "./data/en_cn.txt"
store_path = "models"
word_json_file = "data"

src_lang = "en"
trg_lang = "cn"


src_min_freq = 10
trg_min_freq = 10
max_len = 20
is_label_smooth = True
# n_warmup_steps = 10000
n_warmup_steps = 1000000

test_nums = 10000
# batch_size = 16
# test_batch_size = 4
batch_size = 1
test_batch_size = 1
epochs = 12
is_cuda = True

d_model = 512
n_layers = 6
heads = 8
dropout = 0.1
lr = 0.0001

eval_batch_size = 1