import argparse

parser.add_argument('--input_PPI', default='./STRING_data/STRING_PPI_all.txt', type=str)
parser.add_argument('--seq_file', default='./STRING_data/STRING_sequence.fasta', type=str)
parser.add_argument('--output_SPI', default='./STRING_data/filter.txt', type=str)
parser.add_argument('--output_pro_seq', default='./STRING_data/protein.fasta', type=str)
parser.add_argument('--output_SEP_seq', default='./STRING_data/SEP.fasta', type=str)


# 读取从STRING数据库中下载的具有实验证据的人类蛋白质SPI数据
# 读取他们的sequence序列信息

SPI_all = []
protein_list_all = []
input_PPI = parser.parse_args().input_PPI
with open(input_PPI, 'r') as f_SPI:
    for line in f_SPI.readlines()[1:]:
        line = line.strip()
        protein_1 = line.split(' ')[0]
        protein_2 = line.split(' ')[1]
        score = int(line.split(' ')[2])
        if score < 300:
            continue
        if protein_1 not in protein_list_all:
            protein_list_all.append(protein_1)
        if protein_2 not in protein_list_all:
            protein_list_all.append(protein_2)
        # if [protein_2, protein_1] not in SPI_all:
        SPI_all.append([protein_1, protein_2])
print(len(SPI_all))
print(len(protein_list_all))


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


fasta_dict = []
fasta_seqs = []
seq_file = parser.parse_args().seq_file
for seqName, seq in readFa(seq_file):
    if seq not in fasta_seqs:
        fasta_seqs.append(seq)
        fasta_dict.append((seqName, seq))
fasta_dict = dict(fasta_dict)

# 利用长度区分protein和SEP
protein_dic = []
SEP_dic = []
protein_ids = []
SEP_ids = []
for protein_id in protein_list_all:
    if protein_id not in fasta_dict.keys():
        continue
    else:
        protein_seq = fasta_dict[protein_id]
        if 30 < len(protein_seq) <= 100:
            SEP_dic.append((protein_id, protein_seq))
            SEP_ids.append(protein_id)
        if 100 < len(protein_seq) <= 1000:
            protein_dic.append((protein_id, protein_seq))
            protein_ids.append(protein_id)

print(len(protein_dic))
print(len(SEP_dic))

# 过滤那些只有蛋白质或者只有肽的SPI，保留既有蛋白质又有肽的SPI
filter_SPI = []
filter_protein = []
filter_SEP = []
protein_dic = dict(protein_dic)
SEP_dic = dict(SEP_dic)

for i in range(len(SPI_all)):
    if SPI_all[i][0] in protein_ids and SPI_all[i][1] in SEP_ids:
        if SPI_all[i] not in filter_SPI:
            filter_SPI.append(SPI_all[i])
            if SPI_all[i][0] not in filter_protein:
                filter_protein.append(SPI_all[i][0])
            if SPI_all[i][1] not in filter_SEP:
                filter_SEP.append(SPI_all[i][1])
    elif SPI_all[i][0] in SEP_ids and SPI_all[i][1] in protein_ids:
        if [SPI_all[i][1], SPI_all[i][0]] not in filter_SPI:
            filter_SPI.append([SPI_all[i][1], SPI_all[i][0]])
            if SPI_all[i][1] not in filter_protein:
                filter_protein.append(SPI_all[i][1])
            if SPI_all[i][0] not in filter_SEP:
                filter_SEP.append(SPI_all[i][0])

print(len(filter_SPI))
print(len(filter_protein))
print(len(filter_SEP))


output_SPI = parser.parse_args().output_SPI
output_pro_seq = parser.parse_args().output_pro_seq
output_SEP_seq = parser.parse_args().output_SEP_seq

with open(output_SPI, 'w') as f_filter:
    for i in range(len(filter_SPI)):
        SPI = filter_SPI[i][0] + '\t' + filter_SPI[i][1] + '\n'
        f_filter.write(SPI)

with open(output_pro_seq, 'w') as f_protein:
    for i in range(len(filter_protein)):
        fasta = '>' + filter_protein[i] + '\n' + protein_dic[filter_protein[i]] + '\n'
        f_protein.write(fasta)

with open(output_SEP_seq, 'w') as f_SEP:
    for i in range(len(filter_SEP)):
        fasta = '>' + filter_SEP[i] + '\n' + SEP_dic[filter_SEP[i]] + '\n'
        f_SEP.write(fasta)
