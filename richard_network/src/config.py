import os

data_dir = os.path.join(os.path.expanduser('~'), 'co-attention/data')
log_root = os.path.join(os.path.expanduser('~'), 'co-attention/log')

context_len = 600
question_len = 30

hidden_dim = 200
embedding_size = 300

max_dec_steps = 4
maxout_pool_size = 16

lr = 3e-4
dropout_ratio = 0.15

max_grad_norm = 5.0
batch_size = 200
num_epochs = 50

print_every = 10
save_every = 50000
eval_every = 1000

model_type = 'co-attention'
reg_lambda = 0.00007
