import sys

import numpy as np

from model import Fold_SPI
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from batch_data import LabelledDataset
import argparse

parser = argparse.ArgumentParser(description='Fold_SPI predict')
parser.add_argument('--feature_dir', default='./data_prepare/STRING_data/feature/', type=str)
parser.add_argument('--model_dir', default='./ckpts/STRING/model_best.pkl', type=str)
parser.add_argument('--predict_file', default='./data_prepare/STRING_data/split_by_SEP/test_SEP_1.txt', type=str)
parser.add_argument('--preds_file', default='Fold_SPI_preds_SEP_1.txt', type=str)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('load feature dict')

Feature_dir = parser.parse_args().feature_dir
predict_interact = []
predict_file = parser.parse_args().predict_file
with open(predict_file, 'r') as f:
    for line in f.readlines():
        line_data = line.strip().split(',')[:2]
        line_data.append(0.0)
        predict_interact.append(line_data)
predict_dataset = LabelledDataset(predict_interact, Feature_dir)


def predict(model, dataloader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch, batch_data in enumerate(dataloader):
            X_SEP_2 = batch_data[1].float()
            X_SEP_dense = batch_data[3].float()
            X_SEP_scannet = batch_data[5].float()
            X_prot_2 = batch_data[0].float()
            X_prot_dense = batch_data[2].float()
            X_prot_scannet = batch_data[4].float()

            pred, SEP_global, protein_global = model(X_SEP_2, X_prot_2, X_SEP_dense, X_prot_dense, X_SEP_scannet, X_prot_scannet)
            preds.extend(pred.detach().cpu().numpy().tolist())

    preds = np.array(preds)

    return preds


Model = Fold_SPI()
Model.load_state_dict(torch.load(parser.parse_args().model_dir)['model_state_dict'])
Model = Model.to(device)
criterion = nn.BCELoss()

predict_loader = DataLoader(dataset=predict_dataset, batch_size=128, shuffle=False)
predict_preds = predict(Model, predict_loader)

np.savetxt(parser.parse_args().preds_file, predict_preds, delimiter='\n', fmt='%s')



