import pickle

import torch

from torch.utils.data import Dataset

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.cuda("cpu")


class LabelledDataset(Dataset):
    def __init__(self, interact, feature_dir):
        with open(feature_dir + 'protein_2_feature_dict', 'rb') as f:
            self.protein_2_dict = pickle.load(f, encoding="latin1")
        with open(feature_dir + 'SEP_2_feature_dict', 'rb') as f:
            self.SEP_2_dict = pickle.load(f, encoding="latin1")
        with open(feature_dir + 'protein_dense_feature_dict', 'rb') as f:
            self.protein_dense_dict = pickle.load(f, encoding="latin1")
        with open(feature_dir + 'SEP_dense_feature_dict', 'rb') as f:
            self.SEP_dense_dict = pickle.load(f, encoding="latin1")
        with open(feature_dir + 'protein_scannet_feature_dict', 'rb') as f:
            self.protein_scannet_dict = pickle.load(f, encoding="latin1")
        with open(feature_dir + 'SEP_scannet_feature_dict', 'rb') as f:
            self.SEP_scannet_dict = pickle.load(f, encoding="latin1")
        self.protein = [x[0] for x in interact]
        self.SEP = [x[1] for x in interact]
        # self.protein_id = [x[4] for x in interact]
        # self.SEP_id = [x[3] for x in interact]
        self.label = [x[2] for x in interact]
        self.n_samples = len(interact)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        protein_2 = self.protein_2_dict[self.protein[index]]
        SEP_2 = self.SEP_2_dict[self.SEP[index]]
        protein_dense = self.protein_dense_dict[self.protein[index]]
        SEP_dense = self.SEP_dense_dict[self.SEP[index]]
        protein_scannet = self.protein_scannet_dict[self.protein[index]]
        SEP_scannet = self.SEP_scannet_dict[self.SEP[index]]
        label = self.label[index]

        protein_2 = torch.tensor(protein_2)
        SEP_2 = torch.tensor(SEP_2)
        protein_dense = torch.tensor(protein_dense)
        SEP_dense = torch.tensor(SEP_dense)
        protein_scannet = torch.tensor(protein_scannet)
        SEP_scannet = torch.tensor(SEP_scannet)
        label = torch.tensor(label)

        return protein_2, SEP_2, protein_dense, SEP_dense, protein_scannet, SEP_scannet, label


# interact = []
# Feature_dir = './preprocessing/PPI_data/PPI_feature/'
# with open('./preprocessing/PPI_data/actions_all.txt') as f:  # change your own data here
#     for line in f.readlines():
#         line_data = line.strip().split(',')
#         line_data[2] = float(line_data[2])
#         interact.append(line_data)
#
# dataset = LabelledDataset(interact, Feature_dir)
#
# #
# kf = KFold(n_splits=5, shuffle=True, random_state=0)
# flag = 0
# for train_index, test_index in kf.split(dataset):
#     flag += 1
#     train_fold = torch.utils.data.dataset.Subset(dataset, train_index)
#     test_fold = torch.utils.data.dataset.Subset(dataset, test_index)
#
#     train_loader = DataLoader(dataset=train_fold, batch_size=128, shuffle=True)
#     test_loader = DataLoader(dataset=test_fold, batch_size=128, shuffle=True)
#
#     train_SEP_ids = []
#     train_protein_ids = []
#     for i, batch_data in enumerate(train_loader):
#         Y = batch_data[6].float()
#         X_pep_2 = batch_data[0].float()
#         X_pep_dense = batch_data[1].float()
#         X_pep_scannet = batch_data[2].float()
#         X_prot_2 = batch_data[3].float()
#         X_prot_dense = batch_data[4].float()
#         X_prot_scannet = batch_data[5].float()
#     #     SEP_id = batch_data[7]
#     #     protein_id = batch_data[8]
#     #     train_SEP_ids.extend(SEP_id)
#     #     train_protein_ids.extend(protein_id)
#     #
#     # test_SEP_ids = []
#     # test_protein_ids = []
#     # for i, batch_data in enumerate(test_loader):
#     #     # Y = batch_data[6].float()
#     #     # X_pep_2 = batch_data[0].float()
#     #     # X_pep_dense = batch_data[1].float()
#     #     # X_pep_scannet = batch_data[2].float()
#     #     # X_prot_2 = batch_data[3].float()
#     #     # X_prot_dense = batch_data[4].float()
#     #     # X_prot_scannet = batch_data[5].float()
#     #     SEP_id = batch_data[7]
#     #     protein_id = batch_data[8]
#     #     test_SEP_ids.extend(SEP_id)
#     #     test_protein_ids.extend(protein_id)
#
#     # with open('./train_interact' + str(flag) + '.txt', 'w') as f_train:
#     #     for sep, protein in zip(train_SEP_ids, train_protein_ids):
#     #         output = sep + ',' + protein + '\n'
#     #         f_train.write(output)
#     # with open('./test_interact' + str(flag) + '.txt', 'w') as f_test:
#     #     for sep, protein in zip(test_SEP_ids, test_protein_ids):
#     #         output = sep + ',' + protein + '\n'
#     #         f_test.write(output)
#     train_index = list(train_index)
#     test_index = list(test_index)
#     for index in test_index:
#         if index in train_index:
#             print('error!!!')

