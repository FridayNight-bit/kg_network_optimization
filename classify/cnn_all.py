import torch
import torch.utils.data as Data
from tqdm import tqdm, trange
from NT import data_process, batch_size, print_acc, num_cell, col, CNN
import numpy as np
import pandas as pd

label_dqns = []


def predict():
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
    cnn = torch.load('model/cnn_all.pt')
    cnn.to('cpu')
    cnn.eval()
    correct, total = torch.zeros(11), torch.zeros(11)
    with torch.no_grad():
        t = trange(len(data_iter))
        for X, y in data_iter:
            t.update(1)
            X = X.float()
            y_hat1 = cnn(X)
            y_hat1_max = y_hat1.argmax(dim=1)
            res = (y_hat1_max == y[:, 1] + 0).float()
            for index in range(len(y)):
                label_single, label_index = y[index, 1], y[index, 2]
                if y_hat1_max[index] != 0:
                    label_dqns.append(label_index)
                correct[label_single] += res[index].item()
                total[label_single] += 1
    t.close()

    index_p = [(i * num_cell, i * num_cell + 1, i * num_cell + 2, i * num_cell + 3) for i in label_dqns]
    index_p = np.array(index_p).reshape(-1)
    data = data.iloc[index_p]
    pd_data = pd.DataFrame(data, columns=col)
    pd_data.to_csv('data/data_cover_for_dqn_all.csv', mode='w', header=True, columns=col, index=False)

    print_acc('cnn', correct, total, 11)


if __name__ == '__main__':
    predict()
