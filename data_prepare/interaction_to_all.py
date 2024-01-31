# 创建负样本
import pandas as pd
import numpy as np
import argparse
parser.add_argument('--data_dir', default='./STRING_data', type=str)
data_dir = parser.parse_args().data_dir
data = pd.read_csv(data_dir + '/filter_SPI.txt')

vals_col = np.unique(data[['protein_id']])
vals_row = np.unique(data[['SEP_id']])
df = pd.DataFrame(0, index=vals_row, columns=vals_col)
f_row = df.index.get_indexer
f_col = df.columns.get_indexer
df.values[f_row(data.SEP_id), f_col(data.protein_id)] = 1

data_TN = []
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        if df.values[i, j] == 0:
            data_TN.append([df.columns[j], df.index[i], df.values[i, j]])

data_TP = []
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        if df.values[i, j] == 1:
            data_TP.append([df.columns[j], df.index[i], df.values[i, j]])

data_TN = np.array(data_TN)
np.random.shuffle(data_TN)
data_TN = data_TN[:1 * len(data_TP), :]


data_all = np.concatenate((data_TP, data_TN))
np.random.shuffle(data_all)
np.random.shuffle(data_all)
np.random.shuffle(data_all)
#
np.savetxt(data_dir + '/SPI/SPI_1.txt', data_all, delimiter=',', fmt='%s')

