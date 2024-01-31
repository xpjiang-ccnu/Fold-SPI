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


class Fold_SPI(nn.Module):
    def __init__(self):
        super(Fold_SPI, self).__init__()
        # self.config = config
        self.embed_seq = nn.Embedding(65 + 1, 128)  # padding_idx=0, vocab_size = 65/25, embedding_size=128
        self.embed_ss = nn.Embedding(75 + 1, 128)
        self.embed_two = nn.Embedding(7 + 1, 128)

        self.SEP_convs = ConvNN(384, 64, 7)
        self.prot_convs = ConvNN(384, 64, 8)
        self.SEP_fc = nn.Linear(23, 128)
        self.prot_fc = nn.Linear(23, 128)
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

    def forward(self, x_SEP_2, x_prot_2, x_SEP_dense, x_prot_dense, x_SEP_scannet, x_prot_scannet):
        x_SEP_2 = x_SEP_2.to(device)
        x_prot_2 = x_prot_2.to(device)
        x_SEP_dense = x_SEP_dense.to(device)
        x_prot_dense = x_prot_dense.to(device)
        x_SEP_scannet = x_SEP_scannet.to(device)
        x_prot_scannet = x_prot_scannet.to(device)
        SEP_2_emb = self.embed_two(x_SEP_2.long())
        prot_2_emb = self.embed_two(x_prot_2.long())
        SEP_dense = self.SEP_fc(x_SEP_dense)
        prot_dense = self.prot_fc(x_prot_dense)

        encode_SEP = torch.cat([SEP_2_emb, x_SEP_scannet, SEP_dense], dim=-1)
        encode_protein = torch.cat([prot_2_emb, x_prot_scannet, prot_dense], dim=-1)

        encode_SEP = encode_SEP.permute(0, 2, 1)
        encode_protein = encode_protein.permute(0, 2, 1)

        encode_SEP = self.SEP_convs(encode_SEP)
        encode_SEP = encode_SEP.permute(0, 2, 1)
        encode_SEP_global = self.global_max_pooling(encode_SEP)

        encode_protein = self.prot_convs(encode_protein)
        encode_protein = encode_protein.permute(0, 2, 1)
        encode_protein_global = self.global_max_pooling(encode_protein)


        encode_interaction = torch.cat([encode_SEP_global, encode_protein_global], axis=-1)
        encode_interaction = self.dnns(encode_interaction)
        predictions = torch.sigmoid(self.output(encode_interaction))

        return predictions.squeeze(dim=1)
