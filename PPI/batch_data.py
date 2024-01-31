import pickle

import torch
from sklearn.model_selection import KFold

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.cuda("cpu")


class LabelledDataset(Dataset):
    def __init__(self, interact, feature_dir):
        with open(feature_dir + 'protein_2_feature_dict', 'rb') as f:
            self.protein_2_feature_dict = pickle.load(f, encoding="latin1")
        with open(feature_dir + 'protein_dense_feature_dict', 'rb') as f:
            self.protein_dense_feature_dict = pickle.load(f, encoding="latin1")
        with open(feature_dir + 'protein_scannet_feature_dict', 'rb') as f:
            self.protein_scannet_feature_dict = pickle.load(f, encoding="latin1")
        self.prot1 = [x[0] for x in interact]
        self.prot2 = [x[1] for x in interact]
        self.label = [x[2] for x in interact]
        self.n_samples = len(interact)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        prot1_2 = self.protein_2_feature_dict[self.prot1[index]]
        prot2_2 = self.protein_2_feature_dict[self.prot2[index]]
        prot1_dense = self.protein_dense_feature_dict[self.prot1[index]]
        prot2_dense = self.protein_dense_feature_dict[self.prot2[index]]
        prot1_scannet = self.protein_scannet_feature_dict[self.prot1[index]]
        prot2_scannet = self.protein_scannet_feature_dict[self.prot2[index]]
        label = self.label[index]

        prot1_2 = torch.tensor(prot1_2)
        prot2_2 = torch.tensor(prot2_2)
        prot1_dense = torch.tensor(prot1_dense)
        prot2_dense = torch.tensor(prot2_dense)
        prot1_scannet = torch.tensor(prot1_scannet)
        prot2_scannet = torch.tensor(prot2_scannet)
        label = torch.tensor(label)

        return prot1_2, prot2_2, prot1_dense, prot2_dense, prot1_scannet, prot2_scannet, label
