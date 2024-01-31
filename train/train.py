import numpy as np

# from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, classification_report, \
    average_precision_score, f1_score

import torch.optim as optim
from model import Fold_SPI

from batch_data import LabelledDataset

import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cls_scores(label, pred):
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    # r2_score, mean_squred_error are ignored
    return roc_auc_score(label, pred), average_precision_score(label, pred)


print('load feature dict')

parser = argparse.ArgumentParser(description='Fold_SPI_model_training')
parser.add_argument('--feature_dir', default='../data_prepare/STRING_data/', type=str)
parser.add_argument('--train_file', default='../data_prepare/STRING_data/split_by_SEP/train_SEP_1.txt', type=str)
parser.add_argument('--test_file', default='../data_prepare/STRING_data/split_by_SEP/test_SEP_1.txt', type=str)
parser.add_argument('--label_file', default='Fold_SPI_label_SEP_1.txt', type=str)
parser.add_argument('--preds_file', default='Fold_SPI_preds_SEP_1.txt', type=str)
parser.add_argument('--data_name', default='STRING', type=str)


data_name = parser.parse_args().data_name
Feature_dir = parser.parse_args().feature_dir

train_interact = []
train_file = parser.parse_args().train_file
with open(train_file, 'r') as f:
    for line in f.readlines():
        line_data = line.strip().split(',')[:3]
        line_data[2] = float(line_data[2])
        train_interact.append(line_data)

test_interact = []
test_file = parser.parse_args().test_file
with open(test_file, 'r') as f:
    for line in f.readlines():
        line_data = line.strip().split(',')[:3]
        line_data[2] = float(line_data[2])
        test_interact.append(line_data)

train_dataset = LabelledDataset(train_interact, Feature_dir)
test_dataset = LabelledDataset(test_interact, Feature_dir)


def train(model, dataloader, device, criterion, optimizer):
    model.train()
    preds, labels = [], []
    avg_loss = 0
    criterion.to(device)
    # print('start loading')
    for batch, batch_data in enumerate(dataloader):
        Y = batch_data[6].float()
        X_SEP_2 = batch_data[1].float()
        X_SEP_dense = batch_data[3].float()
        X_SEP_scannet = batch_data[5].float()
        X_prot_2 = batch_data[0].float()
        X_prot_dense = batch_data[2].float()
        X_prot_scannet = batch_data[4].float()

        pred = model(X_SEP_2, X_prot_2, X_SEP_dense, X_prot_dense, X_SEP_scannet, X_prot_scannet)
        preds.extend(pred.detach().cpu().numpy().tolist())
        labels.extend(Y.detach().cpu().numpy().tolist())

        optimizer.zero_grad()
        loss = criterion(pred, Y.to(device))
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)
        avg_loss += loss.item()

    preds = np.array(preds)
    labels = np.array(labels)

    acc = accuracy_score(labels, np.round(preds))
    f1 = f1_score(labels, np.round(preds))
    test_scores = cls_scores(labels, preds)
    AUC = round(test_scores[0], 6)
    AUPR = round(test_scores[1], 6)
    avg_loss /= len(dataloader)

    return avg_loss, AUC, AUPR, acc, f1


def test(model, dataloader, criterion):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        avg_loss = 0
        for batch, batch_data in enumerate(dataloader):
            Y = batch_data[6].float()
            X_SEP_2 = batch_data[1].float()
            X_SEP_dense = batch_data[3].float()
            X_SEP_scannet = batch_data[5].float()
            X_prot_2 = batch_data[0].float()
            X_prot_dense = batch_data[2].float()
            X_prot_scannet = batch_data[4].float()

            pred = model(X_SEP_2, X_prot_2, X_SEP_dense, X_prot_dense, X_SEP_scannet, X_prot_scannet)
            preds.extend(pred.detach().cpu().numpy().tolist())
            labels.extend(Y.detach().cpu().numpy().tolist())

            loss = criterion(pred, Y.to(device))
            avg_loss += loss.item()

        avg_loss /= len(dataloader)

    preds = np.array(preds)
    labels = np.array(labels)

    acc = accuracy_score(labels, np.round(preds))
    f1 = f1_score(labels, np.round(preds))
    test_scores = cls_scores(labels, preds)
    AUC = round(test_scores[0], 6)
    AUPR = round(test_scores[1], 6)

    return avg_loss, AUC, AUPR, acc, f1, preds, labels


test_AUC_list, test_AUPR_list = [], []

colors = ['r', 'g', 'b', 'c', 'm']


best_train_data = []
best_test_data = []


best_test_aupr = 0.0

test_acc_list = []
# train_fold = torch.utils.data.dataset.Subset(train_dataset)
# test_fold = torch.utils.data.dataset.Subset(test_dataset)

train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=True)

print("Start training and device in use:", device)
EPOCHS = 200
model = Fold_SPI()
model = model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
# optimizer = optim.RMSprop(model.parameters(),lr=0.0005) #weight_decay=1e-6
for e in range(EPOCHS):
    train_loss, train_AUC, train_AUPR, train_acc, train_f1 = train(model, train_loader, device, criterion,
                                                                   optimizer)
    test_loss, test_AUC, test_AUPR, test_acc, test_f1, test_preds, labels = test(model, test_loader, criterion)
    print(f"Epoch:{e + 1};")

    print(
        f"Training loss:{train_loss:.4f}, AUC:{train_AUC:.6f}, AUPR:{train_AUPR:.6f}, acc:{train_acc:.4f}, f1:{train_f1:.4f}")
    print(
        f"Test loss:{test_loss:.4f}, AUC:{test_AUC:.6f}, AUPR:{test_AUPR:.6f}, acc:{test_acc:.4f}, f1:{test_f1:.4f}")
    # print('now learning rate: {}'.format(scheduler.optimizer.param_groups[0]['lr']))
    if best_test_aupr < test_AUPR:
        best_test_aupr = test_AUPR
        best_epoch = e
        print(f'Save model of epoch {e}')
        checkpoint = {'model': Fold_SPI(), 'model_state_dict': model.state_dict()}
        torch.save(checkpoint, f'../ckpts/' + data_name + '/model_best.pkl')

        np.savetxt('../result/' + data_name + '/' + parser.parse_args().label_file, labels, delimiter='\n', fmt='%s')
        np.savetxt('../result/' + data_name + '/' + parser.parse_args().preds_file, test_preds, delimiter='\n', fmt='%s')

    test_acc_list.append(test_acc)

test_AUC_list.append(test_AUC)
test_AUPR_list.append(test_AUPR)


print('fold mean auc & aupr', np.mean(test_AUC_list, axis=0), np.mean(test_AUPR_list, axis=0))
# print("best acc & epoch", best_test_acc, best_epoch)
# plt.show()
