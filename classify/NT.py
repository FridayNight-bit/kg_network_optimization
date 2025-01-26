import pandas as pd
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import TensorDataset
from utils.tool import min_max_scaler, train_ch5, classifier_result, print_acc
import numpy as np
from typing import Tuple, Union, Any
import torch
from tqdm import tqdm, trange
from dataGenerate.data_new import col

torch.manual_seed(1)  # reproducible
torch.cuda.manual_seed(1)

# Hyper Parameters
# B_INIT = -0.2
num_cell = 4
num_feature = 17
rate_test = 0.2
batch_size = 256
lr = 0.001
num_epochs = 10
device = 'cpu'

label_dqns = []


def data_process(data_name: str) -> Tuple[TensorDataset, Union[int, Any], Any]:

    tqdm.write('data processing')
    if data_name == 'predict':
        data = pd.read_csv('../data/data_predict_new.csv', header=0, usecols=range(0, num_feature + 1))
    else:
        data = pd.read_csv('../data/data_train_new.csv', header=0, usecols=range(0, num_feature + 1))
    index_sub = [
        (i + 3, i + 4, i + 5, i + 6, i + 7, i + 8, i + 9, i + 10, i + 11) for i in range(0, data.shape[0], 12)]
    index_sub = np.array(index_sub).reshape(-1, 3)
    index_main = [(i, i + 1, i + 2) for i in range(0, data.shape[0], 12)]
    index_main = np.array(index_main).reshape(-1, 1)
    index = np.hstack((index_main, index_sub)).reshape(-1)
    data = data.iloc[index]
    feature = data.iloc[:, :-1]
    label = data.iloc[:, -1]

    num_sample = label.shape[0] // num_cell
    label = torch.Tensor(label.values).squeeze()
    labels = torch.zeros((num_sample, 2), dtype=torch.int64)
    for i in trange(num_sample):
        label_tmp = label[i * num_cell]
        if label_tmp != 0:
            labels[i][0] = 1
            labels[i][1] = label_tmp

    feature_normalize = np.zeros((num_sample * num_cell, num_feature))
    for i in range(num_feature):
        operation_feature = np.array(feature[feature.columns[i]])
        feature_normalize[:, i] = min_max_scaler(operation_feature)
    features = torch.from_numpy(feature_normalize).reshape(num_sample, 1, num_cell, num_feature)

    if data_name == 'predict':
        dataset = Data.TensorDataset(
            features, torch.cat([labels, torch.arange(num_sample).reshape(num_sample, 1)], dim=1))  # 给标签数据附加序号
    elif data_name == 'origin':
        dataset = Data.TensorDataset(features, labels[:, 0])
    else:
        dataset = Data.TensorDataset(features, labels[:, 1])
    return dataset, num_sample, data


def dataset_process(dataset, num_sample: int) -> Tuple[any, any]:
    num_test = int(rate_test * num_sample)
    num_train = num_sample - num_test
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_test], generator=None)



    train_iter = Data.DataLoader(
        dataset=train_dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    test_iter = Data.DataLoader(
        dataset=test_dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    return train_iter, test_iter


def vgg_block(num_convs: int, in_channels: int, out_channels: int, batch_norm: bool):  # vgg卷积层
    blk = []
    for i in range(num_convs):
        if i == 0:
            if batch_norm:
                blk.append(nn.BatchNorm2d(in_channels))
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            # if want same width and length of this image after Conv2d,
            # padding=(kernel_size-1)/2 if stride=1
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        if batch_norm:
            blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        # blk.append(nn.Dropout(p=0.5))
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))  # H,W减半
    return nn.Sequential(*blk)


def fc_block(num_fcs: int, input_size: int, num_classes: int, batch_norm: bool):  # 全连接层
    fcs = []
    hidden_size = input_size // 4
    for i in range(num_fcs):
        if i == 0 and num_fcs != 1:
            fcs.append(nn.Linear(input_size, hidden_size))
            if batch_norm:
                fcs.append(nn.BatchNorm1d(hidden_size))
            fcs.append(nn.ReLU())
            # fcs.append(nn.Dropout(p=0.5))
        elif i == 0:
            fcs.append(nn.Linear(input_size, num_classes))
            fcs.append(nn.Softmax(dim=1))
        elif i != (num_fcs - 1):
            fcs.append(nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                fcs.append(nn.BatchNorm1d(hidden_size))
            fcs.append(nn.ReLU())
            # fcs.append(nn.Dropout(p=0.5))
        else:
            fcs.append(nn.Linear(hidden_size, num_classes))
            fcs.append(nn.Softmax(dim=1))
    return nn.Sequential(*fcs)


class CNN(nn.Module):
    def __init__(self, num_convs=1,
                 in_channels=1,
                 out_channels=1,
                 num_fcs=2,
                 input_size=(4, 17),
                 num_classes=4,
                 batch_norm=True):
        super(CNN, self).__init__()
        self.cnn = vgg_block(num_convs, in_channels, out_channels, batch_norm)
        in_size = out_channels * (input_size[0] // 2 ** num_convs) * (input_size[1] // 2 ** num_convs)
        self.out = fc_block(num_fcs, in_size, num_classes, batch_norm)

    def forward(self, x):
        x = self.cnn(x)
        # flatten the output of conv2 to (batch_size, out_channels * H * W)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


def train_and_save(model_name: str, train_iter, test_iter):
    if model_name == "origin":
        cnn = CNN(out_channels=17)
    elif model_name == "coverage":
        cnn = CNN(out_channels=11, num_classes=5)
    elif model_name == "capacity":
        cnn = CNN(out_channels=6)
    elif model_name == 'interference':
        cnn = CNN(out_channels=5)
    else:
        cnn = CNN(out_channels=17, num_classes=11)

    # print(cnn)  # net architecture

    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)  # optimize all cnn parameters
    train_ch5(model_name, cnn, train_iter, test_iter, optimizer, device, num_epochs)
    torch.save(cnn, 'model/cnn_' + model_name + '.pt')  # save the model
    # torch.save(cnn.state_dict(), 'cnn_' + str + '.pkl')  # save only the parameters


def load_and_predict(class_num: Tuple[int, int, int, int], nrCellID: int, modeType: bool, label: int):
    dataset, _, data = data_process('predict')
    data_iter = Data.DataLoader(
        dataset=dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    tqdm.write('predicting')
    cnn, cnn_coverage, cnn_capacity, cnn_interference = \
        torch.load('../model/cnn_origin.pt', map_location=torch.device('cpu')).to('cpu'), \
        torch.load('../model/cnn_coverage.pt', map_location=torch.device('cpu')).to('cpu'), \
        torch.load('../model/cnn_capacity.pt', map_location=torch.device('cpu')).to('cpu'), \
        torch.load('../model/cnn_interference.pt', map_location=torch.device('cpu')).to('cpu')
    cnn.eval()
    cnn_coverage.eval()
    cnn_capacity.eval()
    cnn_interference.eval()
    correct, correct_coverage, correct_capacity, correct_interference = torch.zeros(class_num[0]), \
                                                                        torch.zeros(class_num[1]), torch.zeros(
        class_num[2]), torch.zeros(class_num[3])
    total, total_coverage, total_capacity, total_interference = torch.zeros(class_num[0]), torch.zeros(class_num[1]), \
                                                                torch.zeros(class_num[2]), torch.zeros(class_num[3])
    acc, n = torch.zeros(11), torch.zeros(11)
    with torch.no_grad():
        t = trange(len(data_iter))
        for X, y in data_iter:
            t.update(1)
            X = X.float()
            y_hat1 = cnn(X)
            y_hat1_max = y_hat1.argmax(dim=1)
            res1 = (y_hat1_max == y[:, 0] + 0).float()
            for index, (x_sample, y_hat1_max_sample) in enumerate(zip(X, y_hat1_max)):
                label_single1, label_single2, label_single3 = y[index, 0], y[index, 1], y[index, 2]
                correct[label_single1] += res1[index].item()
                total[label_single1] += 1
                if y_hat1_max_sample == 1:
                    is_cover_p = classifier_result('coverage', cnn_coverage, x_sample, label_single2, correct_coverage,
                                                   total_coverage, acc, n, device, num_cell, num_feature)
                    label_dqns.append(label_single3) if is_cover_p else None

                else:
                    if label_single2 == 0:
                        acc[label_single2] += 1
                    n[label_single2] += 1
        t.close()

        index_p = [(i * num_cell, i * num_cell + 1, i * num_cell + 2, i * num_cell + 3) for i in label_dqns]
        index_p = np.array(index_p).reshape(-1)
        data = data.iloc[index_p]
        pd_data = pd.DataFrame(data, columns=col)
        pd_data.to_csv('../data/data_cover_for_dqn.csv', mode='w', header=True, columns=col, index=False)

        if modeType == 1:
            if nrCellID in np.arange(1, 4, 1):
                predict_res = 'weak coverage problem '
                opt_mode_select = 'A2C'
                mappingLabel = 1
            elif nrCellID in np.arange(4, 7, 1):
                predict_res = 'cross-regioin coverage problem '
                opt_mode_select = 'A2C'
                mappingLabel = 2
            elif nrCellID in np.arange(7, 10, 1):
                predict_res = 'overlapping coverage problem '
                opt_mode_select = 'Dueling-DDQN'
                mappingLabel = 3
            else:
                predict_res = 'uneven coverage problem '
                opt_mode_select = 'Dueling-DDQN'
                mappingLabel = 4
        # 意图优化
        else:
            if label == 1:
                predict_res = 'weak coverage problem '
                opt_mode_select = 'A2C'
            elif label == 2:
                predict_res = 'cross-regioin coverage problem '
                opt_mode_select = 'A2C'
            elif label == 3:
                predict_res = 'overlapping coverage problem '
                opt_mode_select = 'Dueling-DDQN'
            else:
                predict_res = 'uneven coverage problem '
                opt_mode_select = 'Dueling-DDQN'
            mappingLabel = label



        confidence = (correct_coverage / total_coverage)[mappingLabel]
        return mappingLabel, predict_res + 'credibility：' + str(confidence.item()), opt_mode_select


def train_model(model_name: str):
    dataset, num_sample, _ = data_process(data_name=model_name)
    train_iter, test_iter = dataset_process(dataset=dataset, num_sample=num_sample)
    train_and_save(model_name=model_name, train_iter=train_iter, test_iter=test_iter)



def classifier(nrCellID: int, modeType: bool, label: int):
    # data_process('predict')
    # train_model(model_name='all')
    return load_and_predict((4, 5, 4, 4), nrCellID, modeType, label)

