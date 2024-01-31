# 为计算PSSM做准备

import argparse

parser.add_argument('--SEP_file', default='./STRING_data/SEP.fasta', type=str)
parser.add_argument('--protein_file', default='./STRING_data/protein.fasta', type=str)
parser.add_argument('--output_dir', default='./STRING_data', type=str)


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


SEP_file = parser.parse_args().SEP_file

for seqName, seq in readFa(SEP_file):
    fasta_file = parser.parse_args().output_dir + '/SEP_fasta/' + seqName + '.fasta'
    fasta = '>' + seqName + '\n' + seq + '\n'
    with open(fasta_file, 'w') as f:
        f.write(fasta)

protein_file = parser.parse_args().protein_file

for seqName, seq in readFa(protein_file):
    fasta_file = parser.parse_args().output_dir + '/protein_fasta/' + seqName + '.fasta'
    fasta = '>' + seqName + '\n' + seq + '\n'
    with open(fasta_file, 'w') as f:
        f.write(fasta)
