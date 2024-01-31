import pickle
import numpy as np
import argparse
parser.add_argument('--fasta_filename', default='./STRING_data/SEP', type=str)
parser.add_argument('--output_intrisic_dict', default='./STRING_data/SEP_intrinsic', type=str)


def extract_intrinsic_disorder(filename, ind):
    fasta_filename = filename + '.fasta'
    disorder_filename = filename + '_' + ind + '.result'
    raw_fasta_list = []
    with open('./' + fasta_filename, 'r') as f:
        for line in f.readlines():
            line_list = line.strip()
            raw_fasta_list.append(line_list)

    fasta_id_list = [x[1:] for x in raw_fasta_list if x[0] == '>']
    fasta_sequence_list = [x for x in raw_fasta_list if x[0] != '>']
    fasta_seq_len_list = [len(x) for x in fasta_sequence_list]
    print(len(fasta_id_list), len(fasta_sequence_list), len(fasta_seq_len_list))

    fasta_dict = {}
    for i in range(len(fasta_id_list)):
        fasta_dict[fasta_id_list[i]] = (fasta_sequence_list[i], fasta_seq_len_list[i])

    # load protein intrinsic disorder result
    raw_result_list = []
    with open(disorder_filename, 'r') as f:
        for line in f.readlines():
            line_list = line.strip()
            if len(line_list) > 0 and line_list[0] != '#':
                raw_result_list.append(line_list)

    intrinsic_id_list = [x[1:] for x in raw_result_list if x[0] == '>']
    intrinsic_score_list = [x.split('\t') for x in raw_result_list if x[0] != '>']

    start_idx = 0
    raw_score_dict = {}
    for idx in range(len(intrinsic_id_list)):
        prot_id = intrinsic_id_list[idx]
        seq_len = fasta_dict[prot_id][1]
        end_idx = start_idx + seq_len
        individual_score_list = intrinsic_score_list[start_idx:end_idx]
        individual_score_list = [x[2:] for x in individual_score_list]
        individual_score_array = np.array(individual_score_list, dtype='float')
        raw_score_dict[prot_id] = individual_score_array
        start_idx = end_idx
    print(len(fasta_dict.keys()), len(raw_score_dict.keys()))
    return fasta_dict, raw_score_dict


# long & short
fasta_filename = parser.parse_args().fasta_filename

# the input fasta file used in IUPred2A
fasta_dict_long, raw_score_dict_long = extract_intrinsic_disorder(fasta_filename, 'long')
fasta_dict_short, raw_score_dict_short = extract_intrinsic_disorder(fasta_filename, 'short')

Intrinsic_score_long = {}
for key in fasta_dict_long.keys():
    sequence = fasta_dict_long[key][0]
    seq_len = fasta_dict_long[key][1]
    Intrinsic = raw_score_dict_long[key]
    if Intrinsic.shape[0] != seq_len:
        print('Error!')
    Intrinsic_score_long[key] = Intrinsic

Intrinsic_score_short = {}
for key in fasta_dict_short.keys():
    sequence = fasta_dict_short[key][0]
    seq_len = fasta_dict_short[key][1]
    Intrinsic = raw_score_dict_short[key]
    if Intrinsic.shape[0] != seq_len:
        print('Error!')
    Intrinsic_score_short[key] = Intrinsic

Intrinsic_score = {}
for key in Intrinsic_score_short.keys():
    long_Intrinsic = Intrinsic_score_long[key][:, 0]
    short_Intrinsic = Intrinsic_score_short[key]
    concat_Intrinsic = np.column_stack((long_Intrinsic, short_Intrinsic))
    Intrinsic_score[key] = np.column_stack((long_Intrinsic, short_Intrinsic))

output_intrisic_dict = parser.parse_args().output_intrisic_dict

with open(output_intrisic_dict, 'wb') as f:  # 'output_intrisic_dict' is the name of the output dict you like
    pickle.dump(Intrinsic_score, f)
