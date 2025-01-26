"""
general tool
"""
import torch
import numpy as np
import time
from matplotlib import pyplot as plt
from math import pi, asin, sqrt, sin, cos, tan, log10, log, log2
from typing import List, Union, Tuple
from numpy.random import choice, normal
from random import randint, uniform
from unused.data import EARTH_RADIUS, Jwd, plmn, dus, ran


def getId(i: int) -> str:
    return "0" * (3 - len(str(i + 1))) + str(i + 1)



def getSupportedType() -> str:
    st = ("4G", "5G", "6G")
    return choice(st, p=(0.6, 0.3, 0.1))


def getDistance(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
    radLat1 = lat1 * pi / 180.0
    radLat2 = lat2 * pi / 180.0
    a = radLat1 - radLat2
    b = lng1 * pi / 180.0 - lng2 * pi / 180.0
    dst = 2 * asin((sqrt(pow(sin(a / 2), 2) + cos(radLat1) * cos(radLat2) * pow(sin(b / 2), 2))))
    dst = dst * EARTH_RADIUS
    dst = round(dst * 10000) / 10000
    return dst


def getjw(v: List[Jwd], x: float, y: float, dx1: float, dx2: float, dis1: float) -> (float, float):
    i = randint(0, 3)
    if i == 0:
        x1 = x + uniform(dx1, dx2)
        y1 = y + uniform(dx1, dx2)
    elif i == 1:
        x1 = x - uniform(dx1, dx2)
        y1 = y + uniform(dx1, dx2)
    elif i == 2:
        x1 = x - uniform(dx1, dx2)
        y1 = y - uniform(dx1, dx2)
    else:
        x1 = x + uniform(dx1, dx2)
        y1 = y - uniform(dx1, dx2)
    if dis1 > 0:
        if len(v) > 0:
            for e in v:
                if getDistance(x1, y1, e.lon, e.lat) < dis1:
                    x1, y1 = getjw(v, x1, y1, dx1, dx2, dis1)
        temp = Jwd()
        temp.lon = x
        temp.lat = y
        v.append(temp)
    return x1, y1


def getTransRate(s: str) -> int:
    if s == "4G":
        return round(normal(150, 10))
    elif s == "5G":
        return round(normal(500, 20))
    else:
        return round(normal(1000, 100))


def Mean(res: int, val: int, count: int, cpe_size: int) -> Union[float, int]:
    if count == (cpe_size - 1):
        return (res + val) / cpe_size
    else:
        return res + val


def getplmn() -> str:
    i = randint(1, 3)
    if i == 1:
        return plmn[randint(0, 3)]
    elif i == 2:
        return plmn[randint(4, 6)]
    else:
        return plmn[randint(7, 9)]


def getPlmnList(s: str) -> Tuple[str]:
    i = plmn.index_main(s)
    if i <= 3:
        return plmn[:4]
    elif i <= 6:
        return plmn[4:7]
    else:
        return plmn[7:]


def jw2xy(l: float, B: float) -> (float, float):
    l = l * pi / 180
    B = B * pi / 180
    B0 = 30 * pi / 180
    a = 6378137
    b = 6356752.3142
    e = sqrt(1 - (b / a) * (b / a))
    e2 = sqrt((a / b) * (a / b) - 1)
    CosB0 = cos(B0)
    N = (a * a / b) / sqrt(1 + e2 * e2 * CosB0 * CosB0)
    K = N * CosB0
    SinB = sin(B)
    tans = tan(pi / 4 + B / 2)
    E2 = pow((1 - e * SinB) / (1 + e * SinB), e / 2)
    xx = tans * E2
    xc = K * log(xx)
    yc = K * l
    return xc, yc


def getSnaList(i: int) -> Tuple[str]:
    s = getplmn()
    return tuple([s + str(randint(0, 255)) + str(randint(0, 16777215)) for _ in range(i)])


def getRelatedCellDuList(i: int) -> Tuple[str]:
    return tuple([str(x.CellDuId) for x in dus[i].celldus])



def getTxPower(i: int, j: int, s: str) -> int:
    if s == "mean":
        return ran.gnbs[i].nrcells[j * 3].CellMeanTxPower + ran.gnbs[i].nrcells[j * 3 + 1].CellMeanTxPower + \
               ran.gnbs[i].nrcells[j * 3 + 2].CellMeanTxPower
    else:
        return ran.gnbs[i].nrcells[j * 3].CellMaxTxPower + ran.gnbs[i].nrcells[j * 3 + 1].CellMaxTxPower + \
               ran.gnbs[i].nrcells[j * 3 + 2].CellMaxTxPower


def shannon(bw: int, power: int) -> float:
    s = 10 * log10(power * normal(0.5, 0.03) * 1E3)
    return bw * 1E6 * log2(1 + (s + 174 - 10 * log10(bw * 1E6)))


def evaluate_accuracy_2(data_iter, net, device: str) -> float:
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval()
            acc_sum += (net(X.float().to(device)).argmax(dim=1) == y.to(device) + 0).float().sum().to(device).item()
            net.train()
            n += y.shape[0]
    return acc_sum / n


def train_ch5(model_name: str, net, train_iter, test_iter, optimizer, device: str, num_epochs: int):
    net = net.to(device)
    print("training on", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    train_acc, test_acc, train_loss = [], [], []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.float().to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.to(device).item()
            train_acc_sum += (y_hat.argmax(dim=1) == y + 0).float().sum().to(device).item()
            n += y.shape[0]
            batch_count += 1
        test_acc.append(evaluate_accuracy_2(test_iter, net, device))
        train_acc.append(train_acc_sum / n)
        train_loss.append(train_l_sum / batch_count)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_loss[epoch], train_acc[epoch], test_acc[epoch], time.time() - start))

    plt.plot(range(1, num_epochs + 1), test_acc, linewidth=2, color='olivedrab', label='test data')
    plt.plot(range(1, num_epochs + 1), train_acc, linewidth=2, color='chocolate', linestyle='--', label='train data')
    plt.legend(loc='lower right')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    test_max_index = np.argmax(test_acc).item()
    show_max = '[' + str(test_max_index + 1) + ', ' + str(round(test_acc[test_max_index], 3)) + ']'

    plt.plot(test_max_index + 1, test_acc[test_max_index], 'ko')
    plt.annotate(show_max, xy=(test_max_index + 1, test_acc[test_max_index]),
                 xytext=(test_max_index + 1, test_acc[test_max_index]))
    plt.grid()
    plt.title(model_name)
    plt.savefig('figure/' + model_name)
    plt.show()
    # plt.plot(range(num_epochs), train_loss, linewidth=2, label='loss')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.show()


def min_max_scaler(data):
    mins = np.amin(data)
    maxs = np.amax(data)
    return (data - mins) / (maxs - mins)


def evaluate(model_name: str, cnn, x_sample, y_sample, device: str, num_cell: int, num_feature: int):
    x_sample = x_sample.view(1, 1, num_cell, num_feature)
    y_hat2_sample = cnn(x_sample)
    y_hat2_max_sample = y_hat2_sample.argmax(dim=1)
    if (model_name == 'coverage' and y_sample in range(5)) or y_sample == 0:
        return (y_hat2_max_sample == y_sample + 0).float().to(device).item(), y_sample, \
               True if y_hat2_max_sample != 0 else False
    elif model_name == 'capacity' and y_sample in range(5, 8):
        return (y_hat2_max_sample == y_sample - 4).float().to(device).item(), y_sample - 4
    elif model_name == 'interference' and y_sample in range(8, 11):
        return (y_hat2_max_sample == y_sample - 7).float().to(device).item(), y_sample - 7
    else:
        return -1, -1


def classifier_result(model_name: str, cnn, x_sample, y_sample, correct, total, acc, n, device,
                      num_cell: int, num_feature: int):
    res, label_single, is_cover_p = evaluate(model_name, cnn, x_sample, y_sample, device, num_cell, num_feature)
    if res != -1:
        correct[label_single] += res
        total[label_single] += 1
        acc[y_sample] += res
    n[y_sample] += 1
    return is_cover_p


def print_acc(name: str, correct, total, class_num: int):
    acc_str = name + ' accuracy: %.3f %d / %d \n' % (sum(correct) / sum(total), sum(correct), sum(total))
    for acc_idx in range(class_num):
        if total[acc_idx] == 0:
            acc_str += 'class:%d acc:NA\n' % acc_idx
            continue
        acc = correct[acc_idx] / total[acc_idx]
        acc_str += 'class:%d acc:%.3f %d / %d \n' % (acc_idx, acc, correct[acc_idx], total[acc_idx])
    # tqdm.write(acc_str)
    return acc_str
