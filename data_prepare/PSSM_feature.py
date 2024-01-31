### Load Protein PSSM Files (first change the value of protein_number)
# prot_pssm_dict : key is protein sequence, value is protein PSSM Matrix
import pickle
import glob
import numpy as np
import argparse


prot_pssm_dict_all = {}
prot_pssm_dict = {}

parser.add_argument('--seq_file', default='./STRING_data/SEP.fasta', type=str)
parser.add_argument('--PSSM_dir', default='./STRING_data/SEP_PSSM/', type=str)
parser.add_argument('--output_pssm_dict', default='./STRING_data/SEP_pssm', type=str)

# 读取全部的fasta

def readFa(fa):
    with open(fa, 'r') as FA:
        seqName, seq = '', ''
        while 1:
            line = FA.readline()
            line = line.strip('\n')
            if (line.startswith('>') or not line) and seqName:
                yield seqName, seq
            if line.startswith('>'):
                seqName = line[1:]
                seq = ''
            else:
                seq += line
            if not line:
                break


fasta_dic = []
seq_file = parser.parse_args().seq_file
for seqName, seq in readFa(seq_file):
    fasta_dic.append((seqName, seq))
fasta_dic = dict(fasta_dic)

PSSM_dir = parser.parse_args().PSSM_dir
for protein_id in list(fasta_dic.keys()):
    pssm_line_list = []
    # prot_key = p.split('/')[-1].split('.pssm')[0]
    pssm_file = PSSM_dir + protein_id + '.pssm'
    prot_seq = fasta_dic[protein_id]
    with open(pssm_file, 'r') as f:  # directory to store pssm files (single file of each protein)
        for line in f.readlines()[3:-6]:
            line_list = line.strip().split(' ')
            if line_list[1] == 'X':
                continue
            line_list = [x for x in line_list if x != ''][2:22]
            line_list = [int(x) for x in line_list]
            if len(line_list) != 20:
                print('Error line:' + pssm_file)
                print(line)
                print(line_list)
            pssm_line_list.append(line_list)
        pssm_array = np.array(pssm_line_list)
        if pssm_array.shape[1] != 20:
            print('Error!')
            print(pssm_file)
        else:
            prot_pssm_dict_all[protein_id] = (prot_seq, pssm_array)
            prot_pssm_dict[protein_id] = pssm_array

output_pssm_dict = parser.parse_args().output_pssm_dict
with open(output_pssm_dict, 'wb') as f:  # 'output_pssm_dict' is the name of the output dict you like
    pickle.dump(prot_pssm_dict, f)
