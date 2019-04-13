import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# Improvement: Initialize forget gate biases to 1.0 as per
# "An Empirical Exploration of Recurrent Network Architectures" (Jozefowicz, 2015)
def init_bias(lstm):
    for name, param in lstm.named_parameters():
        if 'bias' in name:
            length = getattr(lstm, name).shape[0]
            start, end = length // 4, length // 2
            param.data.fill_(0)
            param.data[start:end].fill_(1)


class SimpleEncoder(nn.Module):
    """ Document and Question Encoder """

    def __init__(self, embedding, hidden_dim, dropout_ratio):
        super(SimpleEncoder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embedding))
        self.hidden_dim = hidden_dim  # dimension of the LSTM hidden state

        # Improvement: modify the encoder LSTM to a bidirectional LSTM to encode the other direction
        # to achieve better result
        self.encoder = nn.LSTM(self.embedding.embedding_dim, hidden_dim, num_layers=1, batch_first=True,
                               bidirectional=True, dropout=dropout_ratio)

        init_bias(self.encoder)
        self.dropout = nn.Dropout(p=dropout_ratio)  # dropout layer for the final output
        # self.sentinel = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, seq, mask):
        # seq, mask -> batch_size * doc_len or question_len (b*m or n)
        seq_lens = torch.sum(mask, 1)
        seq_lens_sorted, seq_lens_argsort = torch.sort(seq_lens, descending=True)
        _, seq_lens_argsort_reverse = torch.sort(seq_lens_argsort)

        # seq is rearranged by descending length
        seq_descending = torch.index_select(seq, 0, seq_lens_argsort)
        seq_embedding = self.embedding(seq_descending)  # b * m * embedding_size (b*m*l)
        seq_packed = pack_padded_sequence(seq_embedding.float(), seq_lens_sorted, True)
        output, _ = self.encoder(seq_packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = output.contiguous()
        # output is rearranged to its original order
        output = torch.index_select(output, 0, seq_lens_argsort_reverse)
        output = self.dropout(output)

        # Todo add sentinel vector to the output
        # Improvement: Empirically do not increase improvement very much, thus omit here
        # See "Pay More Attention: Neural Architectures for Question-Answering" (Hasan, 2018)
        # https://arxiv.org/pdf/1803.09230.pdf

        return output


class CoattentionEncoder(nn.Module):
    """ Coattention Encoder"""

    def __init__(self, embedding, hidden_dim, dropout_ratio):
        super(CoattentionEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder_dq = SimpleEncoder(embedding, hidden_dim, dropout_ratio)
        self.q_linear = nn.Linear(2*hidden_dim, 2*hidden_dim)
        self.encoder_fusion = nn.LSTM(6*hidden_dim, hidden_dim, num_layers=1, batch_first=True,
                                      bidirectional=True, dropout=dropout_ratio)
        init_bias(self.encoder_fusion)
        self.dropout_input = nn.Dropout(p=dropout_ratio)  # dropout layer for the LSTM input
        self.dropout_output = nn.Dropout(p=dropout_ratio)  # dropout layer for the final output

    def forward(self, q_seq, q_mask, d_seq, d_mask):
        q_encoding_intermediate = self.encoder_dq(q_seq, q_mask)
        d_encoding = self.encoder_dq(d_seq, d_mask)  # b*m*l

        # q_encoding projection
        q_encoding = torch.tanh(self.q_linear(q_encoding_intermediate.view(-1, 2*self.hidden_dim).float())).\
            view(q_encoding_intermediate.size())  # b*n*l

        L = torch.bmm(q_encoding, torch.transpose(d_encoding, 1, 2).float())  # b*n*m
        A_q = F.softmax(L, dim=1)  # b*n*m
        A_d = F.softmax(L, dim=2)  # b*n*m
        C_q = torch.bmm(A_q, d_encoding.float())  # b*n*l
        C_d = torch.bmm(torch.transpose(A_d, 1, 2), torch.cat((q_encoding, C_q), 2))  # b*m*2l

        input_bilstm = torch.cat((d_encoding.double(), C_d.double()), 2)  # b*m*3l
        input_bilstm = self.dropout_input(input_bilstm)
        input_lens = torch.sum(d_mask, 1)
        input_lens_sorted, input_lens_argsort = torch.sort(input_lens, descending=True)
        _, input_lens_argsort_reverse = torch.sort(input_lens_argsort)
        input_descending = torch.index_select(input_bilstm, 0, input_lens_argsort)
        input_packed = pack_padded_sequence(input_descending.float(), input_lens_sorted, True)
        output, _ = self.encoder_fusion(input_packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = output.contiguous()
        output = torch.index_select(output, 0, input_lens_argsort_reverse)
        output = self.dropout_output(output)
        return output


class DoubleCrossAttentionEncoder(nn.Module):
    """ Double Cross Attention Encoder
        Refer to "Pay More Attention: Neural Architectures for Question-Answering" (Hasan, 2018)
        https://arxiv.org/pdf/1803.09230.pdf """

    def __init__(self, embedding, hidden_dim, dropout_ratio):
        super(DoubleCrossAttentionEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder_dq = SimpleEncoder(embedding, hidden_dim, dropout_ratio)
        self.q_linear = nn.Linear(2*hidden_dim, 2*hidden_dim)
        self.encoder_fusion = nn.LSTM(6*hidden_dim, hidden_dim, num_layers=1, batch_first=True,
                                      bidirectional=True, dropout=dropout_ratio)
        init_bias(self.encoder_fusion)
        self.dropout_input = nn.Dropout(p=dropout_ratio)  # dropout layer for the LSTM input
        self.dropout_output = nn.Dropout(p=dropout_ratio)  # dropout layer for the final output

    def forward(self, q_seq, q_mask, d_seq, d_mask):
        q_encoding_intermediate = self.encoder_dq(q_seq, q_mask)
        d_encoding = self.encoder_dq(d_seq, d_mask)  # b*m*l

        # q_encoding projection
        q_encoding = torch.tanh(self.q_linear(q_encoding_intermediate.view(-1, 2*self.hidden_dim).float())).\
            view(q_encoding_intermediate.size())  # b*n*l

        L = torch.bmm(q_encoding, torch.transpose(d_encoding, 1, 2).float())  # b*n*m
        A_q = F.softmax(L, dim=1)  # b*n*m
        A_d = F.softmax(L, dim=2)  # b*n*m
        C_q = torch.bmm(A_q, d_encoding.float())  # b*n*l
        # C_d = torch.bmm(torch.transpose(A_d, 1, 2), torch.cat((q_encoding, C_q), 2))  # b*m*2l

        # Improvement: Add another layer of attention as Double Cross Attention to the old model
        C_d = torch.bmm(torch.transpose(A_d, 1, 2), q_encoding)  # b*m*l
        R = torch.bmm(C_d, torch.transpose(C_q, 1, 2))  # b*m*n
        A_r = F.softmax(R, dim=2)
        C_r = torch.bmm(R, C_q)  # b*m*l

        # input_bilstm = torch.cat((d_encoding.double(), C_d.double()), 2)  # b*m*3l
        input_bilstm = torch.cat((d_encoding.double(), C_d.double(), C_r.double()), 2)  # b*m*3l
        input_bilstm = self.dropout_input(input_bilstm)
        input_lens = torch.sum(d_mask, 1)
        input_lens_sorted, input_lens_argsort = torch.sort(input_lens, descending=True)
        _, input_lens_argsort_reverse = torch.sort(input_lens_argsort)
        input_descending = torch.index_select(input_bilstm, 0, input_lens_argsort)
        input_packed = pack_padded_sequence(input_descending.float(), input_lens_sorted, True)
        output, _ = self.encoder_fusion(input_packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = output.contiguous()
        output = torch.index_select(output, 0, input_lens_argsort_reverse)
        output = self.dropout_output(output)
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
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, seq_encoding, mask, loss_mask, old_idx, hidden_state, u_s, u_e, target=None):
        # seq_encoding -> b*m*2l, mask -> b*m, hidden_state -> b*l, u_s,u_e -> b*2l, target -> b
        b, m, _ = list(seq_encoding.size())
        r = torch.tanh(self.f_r(torch.cat((hidden_state, u_s, u_e), 1).float()))  # b*l
        m_1 = self.f_m_1(torch.cat((seq_encoding.float(), r.unsqueeze(1).expand(b, m, -1).contiguous()), 2).\
                         view(-1, 3*self.hidden_dim)).view(b, m, self.pool_size, self.hidden_dim)  # b*m*p*l
        m_1, _ = torch.max(m_1, 2)  # b*m*l
        m_2 = self.f_m_2(m_1.view(-1, self.hidden_dim)).view(b, m, self.pool_size, self.hidden_dim)  # b*m*p*l
        m_2, _ = torch.max(m_2, 2)  # b*m*l
        output = self.f_final(torch.cat((m_1, m_2), 2).view(-1, 2*self.hidden_dim)).view(b, m, self.pool_size)  # b*m*p
        output, _ = torch.max(output, 2)  # b*m
        output = output + mask  # b*m
        _, idx_output = torch.max(output, 1)  # b

        # Eliminate unnecessary loss values
        if loss_mask is None:
            loss_mask = (idx_output == idx_output)
        else:
            old_idx_ = old_idx * loss_mask.long()
            idx_output_ = idx_output * loss_mask.long()
            loss_mask = (old_idx_ != idx_output_)

        loss = None
        # Calculate the loss
        if target is not None:
            scores = F.log_softmax(output, 1)
            loss = self.loss(scores, target)  # b
            loss = loss * loss_mask.float()

        return idx_output, loss_mask, loss  # the indices of the largest scores and the loss


class DynamicDecoder(nn.Module):
    """ Dynamic Pointer Decoder"""

    def __init__(self, hidden_dim, pool_size, dropout_ratio, max_iter):
        super(DynamicDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_iter = max_iter
        self.decoder = nn.LSTM(4*hidden_dim, hidden_dim, num_layers=1, batch_first=True,
                               bidirectional=False, dropout=dropout_ratio)
        init_bias(self.decoder)
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

            _, lstm_states = self.decoder(torch.cat((u_s, u_e), 1).unsqueeze(1), lstm_states)
            hidden_state, _ = lstm_states
            hidden_state = hidden_state.view(-1, self.hidden_dim)  # b*l

            loss_mask_s, loss_mask_e = None, None

            s_new, loss_mask_s, loss_s = self.hmn_s(seq_encoding, mask_hmn, loss_mask_s, s,
                                                    hidden_state, u_s, u_e, target_s)
            e_new, loss_mask_e, loss_e = self.hmn_e(seq_encoding, mask_hmn, loss_mask_s, e,
                                                    hidden_state, u_s, u_e, target_e)
            if ans_span is not None:
                losses.append(loss_s + loss_e)

            if torch.sum(s_new != s).item() == 0 and torch.sum(e_new != e).item() == 0:
                s = s_new
                e = e_new
                break

            s = s_new
            e = e_new

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
        s, e, loss = self.decoder(U, d_mask, ans_span)
        if ans_span is not None:
            return loss, s, e
        else:
            return s, e


class DCAModel(nn.Module):
    """ Complete Implementation of the DCN Network with double cross attention layer"""

    def __init__(self, embedding, hidden_dim, dropout_ratio, pool_size, max_iter):
        super(DCAModel, self).__init__()
        self.encoder = DoubleCrossAttentionEncoder(embedding, hidden_dim, dropout_ratio)
        self.decoder = DynamicDecoder(hidden_dim, pool_size, dropout_ratio, max_iter)

    def forward(self, q_seq, q_mask, d_seq, d_mask, ans_span=None):
        U = self.encoder(q_seq, q_mask, d_seq, d_mask)
        s, e, loss = self.decoder(U, d_mask, ans_span)
        if ans_span is not None:
            return loss, s, e
        else:
            return s, e
