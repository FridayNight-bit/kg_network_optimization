"""
show figure
"""
from math import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib as mpl

from torch import nn
num_tes = 10
num_cell = 4
f = 2.4

class Net(nn.Module):
    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3_adv = nn.Linear(n_mid, n_out)
        self.fc3_v = nn.Linear(n_mid, 1)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        adv = self.fc3_adv(h2)
        val = self.fc3_v(h2).expand(-1, adv.size(1))
        output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))
        return output

def data_plot3D(mode):
    plt.rcParams.update({'font.size': 14})
    x_value_labelpad = 8
    y_value_labelpad = 10

    data = pd.read_csv('../data/data_plot_' + mode + '.csv', header=0, usecols=('reward', 'RSRP', 'throughput', 'snr'))
    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    X = [i for i in range(data.shape[0])]
    Y = data['RSRP']
    Z = data['throughput']
    plt.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['legend.fontsize'] = 10
    # plt.title('KG-Driven Network Optimization')
    sc = ax.plot_trisurf(X, Y, Z, cmap='coolwarm')
    plt.colorbar(sc, label='Throughput (bps)')
    ax.set_xlabel('Epochs',labelpad=x_value_labelpad)
    ax.set_ylabel('RSRP (dBm)',labelpad=y_value_labelpad)
    plt.tight_layout()
    # plt.savefig('figure/fig_scatter_cover1' + mode)
    plt.savefig('../figure/fig_throughput_kg' + '.pdf', bbox_inches='tight')
    plt.show()



    data = pd.read_csv('../data/data_plot_all' + '1' + '.csv', header=0, usecols=('reward', 'RSRP', 'throughput', 'snr'))


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = [i for i in range(data.shape[0])]
    Y = data['RSRP']
    Z = data['throughput']
    plt.rcParams['axes.unicode_minus'] = False

    sc = ax.plot_trisurf(X, Y, Z, cmap='coolwarm')
    plt.colorbar(sc, label='Throughput (bps)')
    ax.set_xlabel('Epochs', labelpad=x_value_labelpad)
    ax.set_ylabel('RSRP (dBm)', labelpad=y_value_labelpad)

    plt.savefig('../figure/fig_throughput_direct' + '.pdf', bbox_inches='tight')
    plt.show()

    data1 = pd.read_csv('../data/data_plot_' + mode + '.csv', header=0, usecols=('reward', 'RSRP', 'throughput', 'snr'))
    data2 = pd.read_csv('../data/data_plot_all' + '1' + '.csv', header=0, usecols=('reward', 'RSRP', 'throughput', 'snr'))

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    X1 = np.arange(data1.shape[0])
    Y1 = data1['RSRP']
    Z1 = data1['throughput']
    surf1 = ax.plot_trisurf(X1, Y1, Z1, cmap='coolwarm', alpha=0.7, label='algorithm 1')

    X2 = np.arange(data2.shape[0])
    Y2 = data2['RSRP']
    Z2 = data2['throughput']
    surf2 = ax.plot_trisurf(X2, Y2, Z2, cmap='viridis', alpha=0.7, label='algorithm 2')


    # ax.set_ylim(-115, -114)
    fig.colorbar(surf1, ax=ax, shrink=0.6, aspect=15, label='Throughput (bps) (Proposed)', pad=0.02)
    fig.colorbar(surf2, ax=ax, shrink=0.6, aspect=15, label='Throughput (bps) (DDQN)', pad=0.04)

    ax.set_xlabel('Epochs',labelpad=8)
    ax.set_ylabel('RSRP (dBm)',labelpad=10)
    # ax.set_zlabel('Throughput (bps)',labelpad=8)

    plt.tight_layout()

    plt.savefig('../figure/fig_combined_throughput_v1.pdf', bbox_inches='tight')

    plt.show()



    num_tiles = 50

    Ret = np.linspace(np.radians(1), np.radians(30), num=num_tiles).repeat(num_tiles)
    Ptx = np.array([np.linspace(30, 10 * np.log10(3e4), num=num_tiles)] * num_tiles).reshape(num_tiles ** 2)
    Rsrp = get_RSRP(Ptx, Ret)
    state_cover1 = torch.tensor(np.vstack((Ret, Ptx, Rsrp)).transpose(), dtype=torch.float)
    model_cover1 = torch.load('../model/main_q_net_' + mode + '.pt').eval()
    value_cover1 = model_cover1(state_cover1).max(1)[0].detach().numpy()


    f_dl = np.repeat(399000 * 5e-6, num_cell * num_tiles ** 2)
    f_ul = np.repeat(399000 * 5e-6, num_tiles ** 2)
    Rsrp_main = Rsrp
    Rsrp_sub = get_RSRP(Ptx, Ret, 500)
    state_cover_all = torch.tensor(
        np.hstack((f_dl, f_ul, Ptx, Ret, Rsrp_main, Rsrp_sub)).reshape(9, -1).transpose(), dtype=torch.float)
    model_cover_all = torch.load('../model/main_q_net_all.pt')
    value_cover_all = model_cover_all(state_cover_all).max(1)[0].detach().numpy()

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf1 = ax.plot_trisurf(Ret * 180 / np.pi, Ptx, value_cover1, cmap='coolwarm', alpha=0.7, label='algorithm1')

    surf2 = ax.plot_trisurf(Ret * 180 / np.pi, Ptx, value_cover_all, cmap='viridis', alpha=0.7, label='algorithm2')


    fig.colorbar(surf1, ax=ax, shrink=0.6, aspect=15, label='NN Value Function (Proposed)', pad=0.01)
    fig.colorbar(surf2, ax=ax, shrink=0.6, aspect=15, label='NN Value Function (DDQN)', pad=0.05)


    ax.set_xlabel('RET (°)', labelpad=8)
    ax.set_ylabel('PTx (dBm)', labelpad=10)


    plt.tight_layout()


    plt.savefig('../figure/fig_combined_value_function_v1.pdf', bbox_inches='tight')

    plt.show()
def get_RSRP(Ptx, RetTilt, distance=300):
    return Ptx - 10 * log10(num_tes * 1e4) - \
           22 * np.log10(np.repeat(distance, np.array(RetTilt).size) / np.cos(RetTilt)) - 20 * log10(f) - 32.0
data_plot3D("weak_coverage")