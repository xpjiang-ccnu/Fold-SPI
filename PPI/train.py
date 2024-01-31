import numpy as np

# from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, classification_report, \
    average_precision_score, f1_score

import torch.optim as optim
from model import Fold_PPI

from batch_data import LabelledDataset
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cls_scores(label, pred):
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    # r2_score, mean_squred_error are ignored
    return roc_auc_score(label, pred), average_precision_score(label, pred)


print('load feature dict')


parser = argparse.ArgumentParser(description='Fold-PPI train')
parser.add_argument('--feature_dir', default='../data_prepare/STRING_data/', type=str)
parser.add_argument('--data_file', default=None, type=str)
parser.add_argument('--label_file', default=None, type=str)
parser.add_argument('--preds_file', default=None, type=str)

Feature_dir = parser.parse_args().feature_dir
interact = []
with open(parser.parse_args().data_file, 'r') as f:  # change your own data here
    for line in f.readlines():
        line_data = line.strip().split(',')[:3]
        line_data[2] = float(line_data[2])
        interact.append(line_data)

dataset = LabelledDataset(interact, Feature_dir)

kf = KFold(n_splits=5, shuffle=True, random_state=0)


# @torchsnooper.snoop()
def train(model, dataloader, device, criterion, optimizer):
    model.train()
    preds, labels = [], []
    avg_loss = 0
    criterion.to(device)
    # print('start loading')
    for batch, batch_data in enumerate(dataloader):
        Y = batch_data[6].float()
        X_prot1_2 = batch_data[1].float()
        X_prot1_dense = batch_data[3].float()
        X_prot1_scannet = batch_data[5].float()
        X_prot2_2 = batch_data[0].float()
        X_prot2_dense = batch_data[2].float()
        X_prot2_scannet = batch_data[4].float()

        pred = model(X_prot1_2, X_prot2_2, X_prot1_dense, X_prot2_dense, X_prot1_scannet, X_prot2_scannet)
        preds.extend(pred.detach().cpu().numpy().tolist())
        labels.extend(Y.detach().cpu().numpy().tolist())

        optimizer.zero_grad()
        loss = criterion(pred, Y.to(device))
        loss.backward()
        optimizer.step()

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
            X_prot1_2 = batch_data[1].float()
            X_prot1_dense = batch_data[3].float()
            X_prot1_scannet = batch_data[5].float()
            X_prot2_2 = batch_data[0].float()
            X_prot2_dense = batch_data[2].float()
            X_prot2_scannet = batch_data[4].float()

            pred = model(X_prot1_2, X_prot2_2, X_prot1_dense, X_prot2_dense, X_prot1_scannet, X_prot2_scannet)
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


def load_checkpoint(filepath):
    ckpt = torch.load(filepath)
    model = ckpt['model']
    model.load_state_dict(ckpt['model_state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


n_fold = 5

test_AUC_list, test_AUPR_list = [], []

colors = ['r', 'g', 'b', 'c', 'm']
# best_test_acc = 0.0
best_test_aupr = 0.0

best_train_data = []
best_test_data = []

fold = -1
for train_index, test_index in kf.split(dataset):
    fold = fold + 1
    test_acc_list = []
    train_fold = torch.utils.data.dataset.Subset(dataset, train_index)
    test_fold = torch.utils.data.dataset.Subset(dataset, test_index)

    train_loader = DataLoader(dataset=train_fold, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=test_fold, batch_size=128, shuffle=True)

    print("Start training and device in use:", device)
    EPOCHS = 200
    model = Fold_PPI()
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = optim.RMSprop(model.parameters(),lr=0.0005) #weight_decay=1e-6
    for e in range(EPOCHS):
        train_loss, train_AUC, train_AUPR, train_acc, train_f1 = train(model, train_loader, device, criterion, optimizer)
        test_loss, test_AUC, test_AUPR, test_acc, test_f1, test_preds, labels = test(model, test_loader, criterion)
        print(f"Epoch:{e + 1};")

        print(f"Training loss:{train_loss:.4f}, AUC:{train_AUC:.6f}, AUPR:{train_AUPR:.6f}, acc:{train_acc:.4f}, f1:{train_f1:.4f}")
        print(f"Test loss:{test_loss:.4f}, AUC:{test_AUC:.6f}, AUPR:{test_AUPR:.6f}, acc:{test_acc:.4f}, f1:{test_f1:.4f}")
        if best_test_aupr < test_AUPR:
            best_test_aupr = test_AUPR
            best_epoch = e
            print(f'Save model of epoch {e}')
            checkpoint = {'model': Fold_PPI(), 'model_state_dict': model.state_dict()}
            torch.save(checkpoint, f'../ckpts/STRING/model_best.pkl')

            np.savetxt('../result/PPI/' + parser.parse_args().label_file, labels, delimiter='\n', fmt='%s')
            np.savetxt('../result/PPI/' + parser.parse_args().preds_file, test_preds, delimiter='\n', fmt='%s')

        test_acc_list.append(test_acc)

    test_AUC_list.append(test_AUC)
    test_AUPR_list.append(test_AUPR)

    # plt.plot(list(range(EPOCHS)), test_acc_list, colors[fold], label='Fold {0:d}'.format(fold))
    # plt.legend(loc="lower right")
    # plt.title('Test Accuracy Curve')

    print(f'Save model of epoch {e} with {n_fold}-fold cv')
    checkpoint = {'model': Fold_PPI(), 'model_state_dict': model.state_dict()}
    torch.save(checkpoint, f'../ckpts/PPI/model_cv_ckpts_{fold}.pkl')


print('fold mean auc & aupr', np.mean(test_AUC_list, axis=0), np.mean(test_AUPR_list, axis=0))
print("best aupr & epoch", best_test_aupr, best_epoch)
# plt.show()
