# 读取从STRING数据库中下载的具有实验证据的人类蛋白质PPI数据
# 读取他们的sequence序列信息

PPI_all = []
protein_list_all = []

with open('./STRING_PPI_all.txt', 'r') as f_PPI:
    for line in f_PPI.readlines()[1:]:
        line = line.strip()
        protein_1 = line.split(' ')[0]
        protein_2 = line.split(' ')[1]
        score = int(line.split(' ')[2])
        if score < 500:
            continue
        if protein_1 not in protein_list_all:
            protein_list_all.append(protein_1)
        if protein_2 not in protein_list_all:
            protein_list_all.append(protein_2)
        PPI_all.append([protein_1, protein_2])
print(len(PPI_all))
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
# 读取序列信息（可从STRING数据库中下载）
seq_file = './STRING_data/protein.fasta'
for seqName, seq in readFa(seq_file):
    if seq not in fasta_seqs:
        fasta_seqs.append(seq)
        fasta_dict.append((seqName, seq))
fasta_dict = dict(fasta_dict)

filter_PPI = []
filter_protein = []

for i in range(len(PPI_all)):
    if PPI_all[i][0] in list(fasta_dict.keys()) and PPI_all[i][1] in list(fasta_dict.keys()):
        if PPI_all[i] not in filter_PPI:
            filter_PPI.append(PPI_all[i])
            if PPI_all[i][0] not in filter_protein:
                filter_protein.append(PPI_all[i][0])

print(len(filter_PPI))
print(len(filter_protein))

# 将过滤出来的PPI写入
with open('./STRING_data/filter_PPI.txt', 'w') as f_filter:
    for i in range(len(filter_PPI)):
        PPI = filter_PPI[i][0] + '\t' + filter_PPI[i][1] + '\n'
        f_filter.write(PPI)

with open('./STRING_data/protein_PPI.fasta', 'w') as f_protein:
    for i in range(len(filter_protein)):
        fasta = '>' + filter_protein[i] + '\n' + fasta_dict[filter_protein[i]] + '\n'
        f_protein.write(fasta)
