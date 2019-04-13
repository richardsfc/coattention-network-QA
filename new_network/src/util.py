import torch
import torch.nn as nn
import torch.nn.functional as F

def initial_forget_bias(lstm):
    for name, param in lstm.named_parameters():
        if 'bias' in name:
            length = getattr(lstm, name).shape[0]
            start, end = length // 4, length // 2
            param.data.fill_(0)
            param.data[start:end].fill_(1)
