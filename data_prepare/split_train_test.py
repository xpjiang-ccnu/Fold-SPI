import numpy as np
import argparse
parser.add_argument('--data_dir', default='./STRING_data', type=str)
data_dir = parser.parse_args().data_dir

SPIs = {}
SEP_ids = []
Protein_ids = []

test_SEP_ids = []
test_Protein_ids = []
test_SPIs = []
train_SEP_ids = []
train_Protein_ids = []
train_SPIs = []

SPI_file = data_dir + '/filter_SPI.txt'

SPI_len = 0
# with open(SPI_file, 'r') as f:
#     for SPI in f.readlines():
#         SPI_len += 1
#         SEP = SPI.strip().split('\t')[1]
#         protein = SPI.strip().split('\t')[0]
#         if protein not in Protein_ids:
#             Protein_ids.append(protein)
#             SPIs[protein] = [SEP]
#         else:
#             SPIs[protein].append(SEP)
#         if SEP not in SEP_ids:
#             SEP_ids.append(SEP)
#
#
# len_test = int(0.2 * SPI_len)
# len_train = SPI_len - len_test
#
# Protein_ids = np.array(Protein_ids)
# np.random.shuffle(Protein_ids)
# np.random.shuffle(Protein_ids)
# np.random.shuffle(Protein_ids)
# np.random.shuffle(Protein_ids)
#
# cnt = 0
# for protein in Protein_ids:
#     if cnt < len_test:
#         test_Protein_ids.append(protein)
#         for SEP in SPIs[protein]:
#             test_SPIs.append([SEP, protein])
#             cnt += 1
#     else:
#         train_Protein_ids.append(protein)
#         for SEP in SPIs[protein]:
#             train_SPIs.append([SEP, protein])


with open(SPI_file, 'r') as f:
    for SPI in f.readlines():
        SPI_len += 1
        SEP = SPI.strip().split('\t')[1]
        protein = SPI.strip().split('\t')[0]
        if SEP not in SEP_ids:
            SEP_ids.append(SEP)
            SPIs[SEP] = [protein]
        else:
            SPIs[SEP].append(protein)
        if protein not in Protein_ids:
            Protein_ids.append(protein)


len_test = int(0.2 * SPI_len)
len_train = SPI_len - len_test

SEP_ids = np.array(SEP_ids)
np.random.shuffle(SEP_ids)
np.random.shuffle(SEP_ids)
np.random.shuffle(SEP_ids)
np.random.shuffle(SEP_ids)

cnt = 0
for SEP in SEP_ids:
    if cnt < len_test:
        test_SEP_ids.append(SEP)
        for protein in SPIs[SEP]:
            test_SPIs.append([SEP, protein])
            cnt += 1
    else:
        train_SEP_ids.append(SEP)
        for protein in SPIs[SEP]:
            train_SPIs.append([SEP, protein])

with open(data_dir + '/split_by_SEP/test_SPI.txt', 'w') as f_test:
    f_test.write('SEP_id,protein_id\n')
    for SPI in test_SPIs:
        out_spi = SPI[0] + ',' + SPI[1] + '\n'
        f_test.write(out_spi)

with open(data_dir + '/split_by_SEP/train_SPI.txt', 'w') as f_train:
    f_train.write('SEP_id,protein_id\n')
    for SPI in train_SPIs:
        out_spi = SPI[0] + ',' + SPI[1] + '\n'
        f_train.write(out_spi)

