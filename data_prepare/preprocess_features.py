# 整合全部的特征，将他们的编码方式统一，并写入pickle数据
import glob

import numpy as np
import sys
import pickle
import math

import pandas as pd

import argparse
parser.add_argument('--data_dir', default='./STRING_data', type=str)
data_dir = parser.parse_args().data_dir

amino_acid_set = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                  "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                  "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                  "U": 19, "T": 20, "W": 21,
                  "V": 22, "Y": 23, "X": 24,
                  "Z": 25}  # consider non-standard residues

amino_acid_num = 25

ss_set = {"H": 1, "C": 2, "E": 3}  # revise order, not necessary if training your own model
ss_number = 3

physicochemical_set = {'A': 1, 'C': 3, 'B': 7, 'E': 5, 'D': 5, 'G': 2, 'F': 1,
                       'I': 1, 'H': 6, 'K': 6, 'M': 1, 'L': 1, 'O': 7, 'N': 4,
                       'Q': 4, 'P': 1, 'S': 4, 'R': 6, 'U': 7, 'T': 4, 'W': 2,
                       'V': 1, 'Y': 4, 'X': 7, 'Z': 7}

residue_list = list(amino_acid_set.keys())
ss_list = list(ss_set.keys())

new_key_list = []
for i in residue_list:
    for j in ss_list:
        str_1 = str(i) + str(j)
        new_key_list.append(str_1)

new_value_list = [x + 1 for x in list(range(amino_acid_num * ss_number))]

seq_ss_dict = dict(zip(new_key_list, new_value_list))
seq_ss_number = amino_acid_num * ss_number  # 75


def label_sequence(line, pad_prot_len, res_ind):
    X = np.zeros(pad_prot_len)

    for i, res in enumerate(line[:pad_prot_len]):
        X[i] = res_ind[res]

    return X


def label_seq_ss(line, pad_prot_len, res_ind):
    line = line.strip().split(',')
    X = np.zeros(pad_prot_len)
    for i, res in enumerate(line[:pad_prot_len]):
        X[i] = res_ind[res]
    return X


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


sigmoid_array = np.vectorize(sigmoid)


def padding_sigmoid_pssm(x, N):
    x = sigmoid_array(x)
    padding_array = np.zeros([N, x.shape[1]])
    if x.shape[0] >= N:  # sequence is longer than N
        padding_array[:N, :x.shape[1]] = x[:N, :]
    else:
        padding_array[:x.shape[0], :x.shape[1]] = x
    return padding_array


def padding_ScanNet(x, N):
    padding_array = np.zeros([N, x.shape[1]])
    if x.shape[0] >= N:  # sequence is longer than N
        padding_array[:N, :x.shape[1]] = x[:N, :]
    else:
        padding_array[:x.shape[0], :x.shape[1]] = x
    return padding_array


def padding_intrinsic_disorder(x, N):
    padding_array = np.zeros([N, x.shape[1]])
    if x.shape[0] >= N:  # sequence is longer than N
        padding_array[:N, :x.shape[1]] = x[:N, :]
    else:
        padding_array[:x.shape[0], :x.shape[1]] = x
    return padding_array


def readFa(fa):
    with open(fa, 'r') as FA:
        seqName, seq = '', ''
        while 1:
            line = FA.readline()
            line = line.strip('\n')
            if (line.startswith('>') or not line) and seqName:
                yield ((seqName, seq))
            if line.startswith('>'):
                seqName = line[1:]
                seq = ''
            else:
                seq += line
            if not line:
                break


input_file = data_dir + '/filter_SPI.txt'

if __name__ == '__main__':
    f = open(input_file)
    SEP_set = set()
    prot_set = set()

    # if the file has headers and pay attention to the columns (whether have SEP binding site labels)
    for line in f.readlines()[1:]:
        protein, SEP = line.strip().split(',')[:2]
        SEP_set.add(SEP)
        prot_set.add(protein)


    f.close()

    fasta_dict = []
    for seqName, seq in readFa(data_dir + '/protein.fasta'):
        fasta_dict.append((seqName, seq))
    for seqName, seq in readFa(data_dir + '/SEP.fasta'):
        fasta_dict.append((seqName, seq))
    fasta_dict = dict(fasta_dict)

    SEP_len = [len(fasta_dict[SEP]) for SEP in SEP_set]
    prot_len = [len(fasta_dict[prot]) for prot in prot_set]

    SEP_len.sort()
    prot_len.sort()

    pad_SEP_len = 100
    pad_prot_len = prot_len[int(0.8 * len(prot_len)) - 1]

    scannet_SEP_dict_all = []
    for filename in glob.glob(data_dir + '/SEP_ScanNet/*.csv'):
        ID = filename.split('/')[-1].split('.csv')[0]
        each_scannet = pd.read_csv(filename, sep=',')
        scannet_SEP_dict_all.append((ID, each_scannet.values))
    scannet_SEP_dict_all = dict(scannet_SEP_dict_all)

    scannet_protein_dict_all = []
    for filename in glob.glob(data_dir + '/protein_ScanNet/*.csv'):
        ID = filename.split('/')[-1].split('.csv')[0]
        each_scannet = pd.read_csv(filename, sep=',')
        scannet_protein_dict_all.append((ID, each_scannet.values))
    scannet_protein_dict_all = dict(scannet_protein_dict_all)


    scannet_SEP_dict = {}
    scannet_protein_dict = {}

    for key in list(SEP_set):
        key_seq = fasta_dict[key]
        feature = padding_ScanNet(scannet_SEP_dict_all[key], pad_SEP_len)
        scannet_SEP_dict[key] = feature

    for key in list(prot_set):
        key_seq = fasta_dict[key]
        feature = padding_ScanNet(scannet_protein_dict_all[key], pad_prot_len)
        scannet_protein_dict[key] = feature


    # load raw dense features, the directory dense_feature_dict and proprocessing need to be created first.
    with open(data_dir + '/protein_pssm', 'rb') as f:  # value: (sequence_length, 20) without sigmoid
        protein_pssm_dict = pickle.load(f)

    with open(data_dir + '/protein_intrinsic', 'rb') as f:  # value: (sequence_length, 3): long, short, anchor
        protein_intrinsic_dict = pickle.load(f)

    with open(data_dir + '/SEP_intrinsic', 'rb') as f:  # value: (sequence_length, 3): long, short, anchor
        SEP_intrinsic_dict = pickle.load(f)

    with open(data_dir + '/SEP_pssm', 'rb') as f:  # value: (sequence_length, 20) without sigmoid
        SEP_pssm_dict = pickle.load(f)

    SEP_2_feature_dict = {}
    protein_2_feature_dict = {}

    SEP_dense_feature_dict = {}
    protein_dense_feature_dict = {}

    protein_intrinsic_feature_dict = {}
    SEP_intrinsic_feature_dict = {}

    f = open(input_file)
    for line in f.readlines()[1:]:
        protenin, SEP = line.strip().split(',')[:2]
        if SEP not in SEP_2_feature_dict:
            feature = label_sequence(fasta_dict[SEP], pad_SEP_len, physicochemical_set)
            SEP_2_feature_dict[SEP] = feature
        if protenin not in protein_2_feature_dict:
            feature = label_sequence(fasta_dict[protenin], pad_prot_len, physicochemical_set)
            protein_2_feature_dict[protenin] = feature
        if SEP not in SEP_dense_feature_dict:
            sep_feature_pssm = padding_sigmoid_pssm(SEP_pssm_dict[SEP], pad_SEP_len)
            feature = padding_intrinsic_disorder(SEP_intrinsic_dict[SEP], pad_SEP_len)
            feature_dense = np.concatenate((sep_feature_pssm, feature), axis=1)
            SEP_dense_feature_dict[SEP] = feature_dense
        if protenin not in protein_dense_feature_dict:
            feature_pssm = padding_sigmoid_pssm(protein_pssm_dict[protenin], pad_prot_len)
            feature_intrinsic = padding_intrinsic_disorder(protein_intrinsic_dict[protenin], pad_prot_len)
            feature_dense = np.concatenate((feature_pssm, feature_intrinsic), axis=1)
            protein_dense_feature_dict[protenin] = feature_dense

    f.close()

    with open(data_dir + '/feature/SEP_2_feature_dict', 'wb') as f:
        pickle.dump(SEP_2_feature_dict, f)
    with open(data_dir + '/feature/protein_2_feature_dict', 'wb') as f:
        pickle.dump(protein_2_feature_dict, f)
    with open(data_dir + '/feature/SEP_dense_feature_dict', 'wb') as f:
        pickle.dump(SEP_dense_feature_dict, f)
    with open(data_dir + '/feature/protein_dense_feature_dict', 'wb') as f:
        pickle.dump(protein_dense_feature_dict, f)
    with open(data_dir + '/feature/SEP_scannet_feature_dict', 'wb') as f:
        pickle.dump(scannet_SEP_dict, f)
    with open(data_dir + '/feature/protein_scannet_feature_dict', 'wb') as f:
        pickle.dump(scannet_protein_dict, f)
