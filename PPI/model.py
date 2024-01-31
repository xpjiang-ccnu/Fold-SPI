import math

import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
        output, _ = torch.max(x, 1)
        return output


class ConvNN(nn.Module):
    def __init__(self, in_dim, c_dim, kernel_size):
        super(ConvNN, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=c_dim, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=c_dim, out_channels=c_dim * 2, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=c_dim * 2, out_channels=c_dim * 3, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            # GlobalMaxPool1d() # 192
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class Fold_PPI(nn.Module):
    def __init__(self):
        super(Fold_PPI, self).__init__()
        # self.config = config
        self.embed_seq = nn.Embedding(65 + 1, 128)  # padding_idx=0, vocab_size = 65/25, embedding_size=128
        self.embed_ss = nn.Embedding(75 + 1, 128)
        self.embed_two = nn.Embedding(7 + 1, 128)

        self.prot1_convs = ConvNN(384, 64, 7)
        self.prot2_convs = ConvNN(384, 64, 8)
        self.prot1_fc = nn.Linear(23, 128)
        self.prot2_fc = nn.Linear(23, 128)
        self.global_max_pooling = GlobalMaxPool1d()
        # self.dnns = DNN(config.in_dim,config.d_dim1,config.d_dim2,config.dropout)
        self.dnns = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512))
        # c_dim
        self.output = nn.Linear(512, 1)

    def forward(self, x_prot1_2, x_prot2_2, x_prot1_dense, x_prot2_dense, x_prot1_scannet, x_prot2_scannet):
        x_prot1_2 = x_prot1_2.to(device)
        x_prot2_2 = x_prot2_2.to(device)
        x_prot1_dense = x_prot1_dense.to(device)
        x_prot2_dense = x_prot2_dense.to(device)
        x_prot1_scannet = x_prot1_scannet.to(device)
        x_prot2_scannet = x_prot2_scannet.to(device)
        prot1_2_emb = self.embed_two(x_prot1_2.long())
        prot2_2_emb = self.embed_two(x_prot2_2.long())
        prot1_dense = self.prot1_fc(x_prot1_dense)
        prot2_dense = self.prot2_fc(x_prot2_dense)

        encode_prot1 = torch.cat([prot1_2_emb, x_prot1_scannet, prot1_dense], dim=-1)
        encode_prot2 = torch.cat([prot2_2_emb, x_prot2_scannet, prot2_dense], dim=-1)

        encode_prot1 = encode_prot1.permute(0, 2, 1)
        encode_prot2 = encode_prot2.permute(0, 2, 1)

        encode_prot1 = self.prot1_convs(encode_prot1)
        encode_prot1 = encode_prot1.permute(0, 2, 1)
        encode_prot1_global = self.global_max_pooling(encode_prot1)

        encode_prot2 = self.prot2_convs(encode_prot2)
        encode_prot2 = encode_prot2.permute(0, 2, 1)
        encode_prot2_global = self.global_max_pooling(encode_prot2)


        encode_interaction = torch.cat([encode_prot1_global, encode_prot2_global], axis=-1)
        encode_interaction = self.dnns(encode_interaction)
        predictions = torch.sigmoid(self.output(encode_interaction))

        return predictions.squeeze(dim=1)
