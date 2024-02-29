import predict_features
import glob

# list_chains = ['2kho',  # Hsp70 protein.
#                '2p6b_AB',  # Calcineurin.
#                '1brs_A',  # Barnase
#                # '/path/to/my/file.pdb' # Local file.
#                ]
pdb_list = glob.glob('./PDB/*.pdb')

list_chains = [x.split('/')[-1].split('.pdb')[0] for x in pdb_list]

list_dictionary_features = predict_features.predict_features(list_chains,
                                                             model='ScanNet_PPI_noMSA',  # PPBS model without evolution.
                                                             layer='SCAN_filter_activity_aa',
                                                             # AA-scale spatio-chemical filters
                                                             output_format='dictionary')

# print('Residue ID', 'Features 1-5')
# for key, item in list(list_dictionary_features[0].items())[:10]:
#     print(key, '%.2f,%.2f,%.2f,%.2f,%.2f' % (item[0], item[1], item[2], item[3], item[4]))

for i, value in enumerate(list_chains):
    file = './SEP_ScanNet/' + value + '.csv'
    with open(file, 'w') as f:
        for key, items in list(list_dictionary_features[i].items()):
            out_data = ''
            for item in items:
                out_data += '%.2f,' % item
            out_data = out_data[:-1] + '\n'
            f.write(out_data)

