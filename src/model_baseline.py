from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

use_cuda = torch.cuda.is_available()

# AJ: Need this function? Could get everything in one below or call this one hidden size
# out-of-vocabulary words to zero
def get_pretrained_embedding(np_embd):
    embedding = nn.Embedding(*np_embd.shape)
    embedding.weight = nn.Parameter(torch.from_numpy(np_embd).float())
    embedding.weight.requires_grad = False
    return embedding

# AJ: This is a general Encoder (Question or Document). It takes in
# Embeddings of words or else. The relevant dimension are the hidden dimension and the embedding
# dimension. The encoder itself is a Gated Recurrent Unit, instead of a RNN.
# Source of code below: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
class Encoder(nn.Module):
    def __init__(self, hidden_dim, emb_matrix, dropout_ratio):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim # AJ: variable naming size instead of dim? --> hidden_size

        self.embedding = get_pretrained_embedding(emb_matrix)
        self.emb_dim = self.embedding.embedding_dim # AJ: This is not clean code

        # Review
        # Should try RNN first, then LSTM, and only then GRU?
        self.encoder = nn.GRU(self.emb_dim, hidden_dim, 1, batch_first=True,
                              bidirectional=True, dropout=dropout_ratio) # two first: hidden_size x hidden_size
        self.dropout_emb = nn.Dropout(p=dropout_ratio)

    # Review
    # Adjustes lenghts and lets go through
    def forward(self, seq, mask):
        lens = torch.sum(mask, 1) # AJ: lens is length of sentences? Mask is one-hot vector (1=yes, 0=no)
        lens_sorted, lens_argsort = torch.sort(lens, 0, True) # Sort longest to shortest
        _, lens_argsort_argsort = torch.sort(lens_argsort, 0) # Again? Will use to index later
        seq_ = torch.index_select(seq, 0, lens_argsort) #]
        seq_embd = self.embedding(seq_)

        # AJ: This is for dealing with variable length mini batch sequences
        packed = pack_padded_sequence(seq_embd, lens_sorted, batch_first=True)
        output, _ = self.encoder(packed) # Let packed go through GRU
        e, _ = pad_packed_sequence(output, batch_first=True) # Undo packing
        e = e.contiguous() # tensor in contiguous memory
        e = torch.index_select(e, 0, lens_argsort_argsort)  # B x m x 2l # Indexing tensor: Input, dimension, which we want to index
        e = self.dropout_emb(e) # ??
        return e

class Baseline(nn.Module):
    def __init__(self, hidden_dim, emb_matrix, dropout_ratio):
        super(Baseline, self).__init__()
        self.hidden_dim = hidden_dim

        self.encoder = Encoder(hidden_dim, emb_matrix, dropout_ratio)

        self.dropout_att = nn.Dropout(p=dropout_ratio)
        self.fc = nn.Linear(4*hidden_dim, hidden_dim)
        self.fc_start = nn.Linear(hidden_dim, 1)
        self.fc_end = nn.Linear(hidden_dim, 1)

        self.loss = nn.CrossEntropyLoss()


    def forward(self, q_seq, q_mask, d_seq, d_mask, span=None):
        Q = self.encoder(q_seq, q_mask)
        D = self.encoder(d_seq, d_mask)

        b, m, _ = list(D.size())

        # query processing ends
        # attention
        Q_t = torch.transpose(Q, 1, 2)  # B x 2l x n
        A = torch.bmm(D, Q_t)  # B x m x n      # Review: Batch product of matrix times matrix, do it other way around than in paper!
        A = F.softmax(A, dim=2)  # B x m x n
        C = torch.bmm(A, Q) # (B x m x n) X (B x n x 2l) => b x m x 2l
        B = torch.cat([C, D], 2) # B x m x 4l  # AJ: What this?
        B = self.dropout_att(B)
        # attention ends
        B_hat = F.relu(self.fc(B.view(-1, 4*self.hidden_dim))) # B*m x l

        mask_mult = (1.0 - d_mask.float())*(-1e30)

        logit_start = self.fc_start(B_hat).view(-1, m)  # B x m
        logit_start = logit_start + mask_mult # B x m
        _, start_i = torch.max(logit_start, dim=1)

        logit_end = self.fc_end(B_hat).view(-1, m)  # B x m
        logit_end = logit_end + mask_mult
        _, end_i = torch.max(logit_end, dim=1)

        if span is not None:
            loss_value = self.loss(logit_start, span[:, 0])
            loss_value += self.loss(logit_end, span[:, 1])

            loss_value = torch.mean(loss_value)
            return loss_value, start_i, end_i
        else:
            return start_i, end_i
