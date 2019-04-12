import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from allennlp.nn.initializers import lstm_hidden_bias


class SimpleEncoder(nn.Module):
    """ Document and Question Encoder """

    def __init__(self, embedding, hidden_dim, dropout_ratio):
        super(SimpleEncoder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding)
        self.hidden_dim = hidden_dim  # dimension of the LSTM hidden state
        self.encoder = nn.LSTM(self.embedding.embedding_dim, hidden_dim, num_layers=1, batch_first=True,
                               bidirectional=False, dropout=dropout_ratio)
        # Initialize forget gate biases to 1.0 as per "An Empirical
        # Exploration of Recurrent Network Architectures" (Jozefowicz, 2015)
        lstm_hidden_bias(self.encoder)
        self.dropout = nn.Dropout(p=dropout_ratio)  # dropout layer for the final output
        self.sentinel = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, seq, mask):
        # seq, mask -> batch_size * doc_len or question_len (b*m or n)
        seq_lens = torch.sum(mask, 1)
        seq_lens_sorted, seq_lens_argsort = torch.sort(seq_lens, descending=True)
        _, seq_lens_argsort_reverse = torch.sort(seq_lens_argsort)

        # seq is rearranged by descending length
        seq_descending = torch.index_select(seq, 0, seq_lens_argsort)
        seq_embedding = self.embedding(seq_descending)  # b * m * embedding_size (b*m*l)
        seq_packed = pack_padded_sequence(seq_embedding, seq_lens_sorted, True)
        output, _ = self.encoder(seq_packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = output.contiguous()
        # output is rearranged to its original order
        output = torch.index_select(output, 0, seq_lens_argsort_reverse)
        output = self.dropout(output)

        # Todo add sentinel vector to the output

        return output


class CoattentionEncoder(nn.Module):
    """ Coattention Encoder"""

    def __init__(self, embedding, hidden_dim, dropout_ratio):
        super(CoattentionEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder_dq = SimpleEncoder(embedding, hidden_dim, dropout_ratio)
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_fusion = nn.LSTM(3*hidden_dim, 2*hidden_dim, num_layers=1, batch_first=True,
                                      bidirectional=True, dropout=dropout_ratio)
        # Initialize forget gate biases to 1.0 as per "An Empirical
        # Exploration of Recurrent Network Architectures" (Jozefowicz, 2015)
        lstm_hidden_bias(self.encoder_fusion)
        self.dropout = nn.Dropout(p=dropout_ratio)  # dropout layer for the final output

    def forward(self, q_seq, q_mask, d_seq, d_mask):
        q_encoding_intermediate = self.encoder_dq(q_seq, q_mask)
        d_encoding = self.encoder_dq(d_seq, d_mask)  # b*(m+1)*l

        # q_encoding projection
        q_encoding = F.tanh(self.q_linear(q_encoding_intermediate.view(-1, self.hidden_dim))).\
            view(q_encoding_intermediate.size())  # b*(n+1)*l

        L = torch.bmm(q_encoding, torch.transpose(d_encoding, 1, 2))  # b*(n+1)*(m+1)
        A_q = F.softmax(L, dim=1)  # b*(n+1)*(m+1)
        A_d = F.softmax(L, dim=2)  # b*(n+1)*(m+1)
        C_q = torch.bmm(A_q, d_encoding)  # b*(n+1)*l
        C_d = torch.bmm(torch.transpose(A_d, 1, 2), torch.cat((q_encoding, C_q), 2))  # b*(m+1)*2l

        input_bilstm = torch.cat((d_encoding, C_d), 2)  # b*(m+1)*3l
        # Should do dropout?
        input_lens = torch.sum(d_mask, 1)
        input_lens_sorted, input_lens_argsort = torch.sort(input_lens, descending=True)
        _, input_lens_argsort_reverse = torch.sort(input_lens_argsort)
        input_descending = torch.index_select(input_bilstm, 0, input_lens_argsort)
        input_packed = pack_padded_sequence(input_descending, input_lens_sorted, True)
        output, _ = self.encoder_fusion(input_packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = output.contiguous()
        output = torch.index_select(output, 0, input_lens_argsort_reverse)
        output = self.dropout(output)
        return output


class HighwayMaxoutModel(nn.Module):
    """ Highway Maxout Network to calculate start and end scores"""

    def __init__(self, hidden_dim, pool_size):
        super(HighwayMaxoutModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.pool_size = pool_size
        self.f_r = nn.Linear(5*hidden_dim, hidden_dim, bias=False)
        self.f_m_1 = nn.Linear(3*hidden_dim, pool_size*hidden_dim)
        self.f_m_2 = nn.Linear(hidden_dim, pool_size*hidden_dim)
        self.f_final = nn.Linear(2*hidden_dim, pool_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, seq_encoding, mask, hidden_state, u_s, u_e, target=None):
        # seq_encoding -> b*m*2l, mask -> b*m, hidden_state -> b*l, u_s,u_e -> b*2l, target -> b
        b, m, _ = list(seq_encoding.size())
        r = F.tanh(self.f_r(torch.cat((hidden_state, u_s, u_e), 1)))  # b*l
        m_1 = self.f_m_1(torch.cat((seq_encoding, r.expand(b, m, -1).contiguous()), 2).view(-1, 3*self.hidden_dim)).\
            view(b, m, self.pool_size, self.hidden_dim)  # b*m*p*l
        m_1, _ = torch.max(m_1, 2)  # b*m*l
        m_2 = self.f_m_2(m_1.view(-1, self.hidden_dim)).view(b, m, self.pool_size, self.hidden_dim)  # b*m*p*l
        m_2, _ = torch.max(m_2, 2)  # b*m*l
        output = self.f_final(torch.cat((m_1, m_2), 2).view(-1, 2*self.hidden_dim)).view(b, m, self.pool_size)  # b*m*p
        output, _ = torch.max(output, 2)  # b*m
        output = output + mask  # b*m
        _, idx_output = torch.max(output, 1)  # b

        loss = None
        # Calculate the loss
        if target is not None:
            scores = F.log_softmax(output)
            loss = self.loss(scores, target)  # b

        return idx_output, loss  # the indices of the largest scores and the loss


class DynamicDecoder(nn.Module):
    """ Dynamic Pointer Decoder"""

    def __init__(self, hidden_dim, pool_size, dropout_ratio, max_iter):
        super(DynamicDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_iter = max_iter
        self.decoder = nn.LSTM(4*hidden_dim, hidden_dim, num_layers=1, batch_first=True,
                               bidirectional=False, dropout=dropout_ratio)
        # Initialize forget gate biases to 1.0 as per "An Empirical
        # Exploration of Recurrent Network Architectures" (Jozefowicz, 2015)
        lstm_hidden_bias(self.decoder)

        self.hmn_s = HighwayMaxoutModel(hidden_dim, pool_size)
        self.hmn_e = HighwayMaxoutModel(hidden_dim, pool_size)

    def forward(self, seq_encoding, mask, ans_span):
        # seq_encoding -> b*m*2l, mask -> b*m, ans_span -> b*2
        b, m, _ = list(seq_encoding.size())
        # Initialize s to be the first word, e to be the last word
        s = torch.zeros(b).long()
        e = torch.sum(mask, 1) - 1
        indices = torch.arange(0, b).long()
        # Pass to HighwayMaxout to make choosing padding impossible
        mask_hmn = (1 - mask).float() * -1e15

        # s, e, indices are created in the method, need to move to CUDA if possible
        if torch.cuda.is_available():
            s = s.cuda()
            e = e.cuda()
            indices = indices.cuda()

        target_s, target_e = None, None
        if ans_span is not None:
            target_s = ans_span[:, 0]  # b
            target_e = ans_span[:, 1]  # b

        lstm_states = None
        losses = []

        for _ in range(self.max_iter):
            u_s = seq_encoding[indices, s, :]  # b*2l
            u_e = seq_encoding[indices, e, :]  # b*2l

            _, lstm_states = self.decoder(torch.cat((u_s, u_e), 1), lstm_states)
            hidden_state, _ = lstm_states
            hidden_state.view(-1, self.hidden_dim)  # b*l

            s_new, loss_s = self.hmn_s(seq_encoding, mask_hmn, hidden_state, u_s, u_e, target_s)
            e_new, loss_e = self.hmn_e(seq_encoding, mask_hmn, hidden_state, u_s, u_e, target_e)
            if ans_span is not None:
                losses.append(loss_s + loss_e)

            if torch.sum(s_new != s).item() == 0 and torch.sum(e_new != e).item() == 0:
                s = s_new
                e = e_new
                break

        cumulative_loss = None

        if ans_span is not None:
            cumulative_loss = torch.sum(torch.stack(losses, 1), 1)  # b
            cumulative_loss = cumulative_loss / self.max_iter  # b
            cumulative_loss = torch.mean(cumulative_loss)  # 1

        return s, e, cumulative_loss


class DCNModel(nn.Module):
    """ Complete Implementation of the DCN Network"""

    def __init__(self, embedding, hidden_dim, dropout_ratio, pool_size, max_iter):
        super(DCNModel, self).__init__()
        self.encoder = CoattentionEncoder(embedding, hidden_dim, dropout_ratio)
        self.decoder = DynamicDecoder(hidden_dim, pool_size, dropout_ratio, max_iter)

    def forward(self, q_seq, q_mask, d_seq, d_mask, ans_span=None):
        U = self.encoder(q_seq, q_mask, d_seq, d_mask)
        s, e = self.decoder(U, d_mask, ans_span)
        return s, e
