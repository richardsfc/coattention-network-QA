import torch
import torch.nn as nn


'''
LSTM Encoder
    Input: ...
    Output: ...
'''

class Encoder(nn.Module):
    def __init__(self): # missing param
        super(Encoder, self).__init__()

        # get embeddings
        self.embedding = nn.Embedding.from_pretrained(emb_matrix, freeze=True, sparse=False) # investigate function, shoudl change freeze?

        # dimensions of model
        self.input_size  = self.embedding.embedding_dim # make sure, possible mistake
        self.hidden_size = hidden_size
        self.drop_ratio  = drop_ratio                   # define this, usually hyperparameter when call coattention model
        self.num_layers  = 1                            # or set variable? num_layers

        # LSTM encoder
        self.encoder = nn.LSTM(self.input_size, self.hidden_size, self.num_layers,
                               batch_first=True, bidirectional=False,
                               dropout=dropout_ratio) # leave dropout_ratio ?

        # No forget bias function

        # Dropout
        self.dropout = nn.Dropout(p=dropout_ratio) # leave dropout_ratio ? also named var differentlys

        # Sentinel vector
        self.sentinel = nn.Parameter(torch.rand(hidden_size,))

        #### Coattention Model ####
        self.q_proj = nn.Linear(hidden_size, hidden_size) # standard should be hidden_size or self.hidden_size
        # need Fusion BiLSTM and decoder

    def forward(self, input): # missing param
        # here a bunch of sum, sort, and index_select of input, necessary?

        # get pretrained embedding
        embedded = self.embedding(input)

        # here before encoding here packed padded sequence, see function

        # encode embedding with LSTM
        output, _ = self.encoder(embedded)

        # here un pad previously padded, might also need contiguous function

        output = self.dropout(output)

        # sentinel unsqueeze expand unsqueeze, same for lens
        # missing part, not understanding its function


        #### Coattention Model ####

        # Embed Question and Document with LSTM
        Q_ = self.encoder(q_input, q_mask) # make sure these two var are imported or passed somewheres
        D  = self.encoder(d_input, d_mask)

        # Project question embedidngs
        Q = torch.tanh(self.q_proj(Q_.view(-1, self.hidden_size))).view(Q_.size()) # Why view and then size ? Why this and not linear combination?

        # Affinity Matrix
        D_t = torch.transpose(D, 1, 2)
        L = torch.bmm(D_t, Q) # bmm = batch matrix multiplication

        # Row-wise attention weights
        A_Q_ = F.softmax(L, dim=1)
        A_Q = torch.transpose(A^Q_, 1, 2)
        C_Q = torch.bmm(D_t, A_Q)

        # Column-wise attention weights
        A_D = F.softmax(L, dim=2)
        # missing Q_t, why would need to transpose Q? Never transposed in paper. Maybe for dot product?
        C_D = torch.bmm(torch.cat((Q, C_Q), 1), A_D) # in other code Q_t instead of Q. Why?

        C_D_t = torch.transpose(C_D, 1, 2)




        return output



class Decoder(nn.Module):
    def __init__(self):
