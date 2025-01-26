"""
Proposed KG-driven algorithm
"""
import random
from collections import namedtuple
from math import *
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.random import normal
from torch import nn
from torch import optim
from tqdm import trange, tqdm

from classify.NT import num_cell


torch.manual_seed(1)
torch.cuda.manual_seed(1)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
GAMMA = 0.99
MAX_STEPS = 50
NUM_EPISODES = 100
BATCH_SIZE = 32
CAPACITY = 10000
device = 'cpu'
num_samples = 0

num_tes = 10
distances = np.random.normal(300, 30, num_tes)
f = 2.4
n0 = -174
lamuda = 0.7
bw = 1e7


class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, state, action, state_next, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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


class Brain:
    def __init__(self, num_states, num_actions, is_train, mode):
        self.mode = mode
        self.num_actions = num_actions
        self.memory = ReplayMemory(CAPACITY)
        n_in, n_mid, n_out = num_states, 32, num_actions
        if is_train:
            self.main_q_network = Net(n_in, n_mid, n_out).to(device)
            self.target_q_network = Net(n_in, n_mid, n_out).to(device)
        else:
            self.main_q_network = torch.load('../model/main_q_net_' + mode + '.pt').to(device)
            self.target_q_network = torch.load('../model/target_q_net_' + mode + '.pt').to(device)
        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=0.0001)

    def save_net(self):
        torch.save(self.main_q_network, '../model/main_q_net_' + self.mode + '.pt')
        torch.save(self.target_q_network, '../model/target_q_net_' + self.mode + '.pt')

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = \
            self.make_minibatch()
        self.expected_state_action_values = self.get_expected_state_action_values()
        self.update_main_q_network()

    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))
        if epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval()
            with torch.no_grad():
                action = self.main_q_network(state).max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[random.randrange(self.num_actions)]])
        return action

    def make_minibatch(self):
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        self.main_q_network.eval()
        self.target_q_network.eval()
        self.state_action_values = self.main_q_network(self.state_batch).gather(1, self.action_batch)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, self.batch.next_state)))
        next_state_values = torch.zeros(BATCH_SIZE)
        a_m = torch.zeros(BATCH_SIZE).type(torch.int64)
        a_m[non_final_mask] = self.main_q_network(self.non_final_next_states).detach().max(1)[1]
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)
        next_state_values[non_final_mask] = self.target_q_network(
            self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()
        expected_state_action_values = self.reward_batch + GAMMA * next_state_values
        return expected_state_action_values

    def update_main_q_network(self):
        self.main_q_network.train()
        loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())


class Agent:
    def __init__(self, num_states, num_actions, is_train, mode):
        self.brain = Brain(num_states, num_actions, is_train, mode)

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        self.brain.update_target_q_network()


def get_RSRP(Ptx, RetTilt, distance=300):
    return Ptx - 10 * log10(num_tes * 1e4) - \
           22 * np.log10(np.repeat(distance, np.array(RetTilt).size) / np.cos(RetTilt)) - 20 * log10(f) - 32.0
    # Ptx:dBm; distance:m; f:GHz;


class Env:
    def __init__(self, is_train, mode):
        if not is_train:
            data = pd.read_csv("../data/data_cover_for_dqn_all.csv", header=0, usecols=['MaxTxPower', 'RetTilt', 'Label'])
            if mode == "weak_coverage":
                data = data[data['Label'] == 1]
            else:
                data = data[data['Label'] == 2]
            global num_samples
            # data = data[:100]
            num_samples = data.shape[0]
            self.Rets = np.array(data['RetTilt'] / 10)
            self.Ptxs = np.array(data['MaxTxPower'])  # W
        self.reset(is_train, mode, 0)

    def reward(self, Ret, Ptx):
        self.RSRP = get_RSRP(Ptx, Ret)
        snr = np.power(10, (self.RSRP - n0 - 10 * log10(bw / num_tes)) / 10)
        R = (bw / num_tes) * np.log2(1 + snr)
        R_avg = R.mean()
        R_edge = np.array(sorted(R)[: int(num_tes * 0.3)]).mean()
        reward = lamuda * R_edge + (1 - lamuda) * R_avg
        return reward, R_avg, snr.mean()

    def step(self, action):
        Ptx_max = 10 * log10(3e4)
        Ptx_min = 30
        Ret_max = radians(30)
        Ret_min = radians(1)
        if action == 0 and self.Ptx < Ptx_max:
            self.Ptx += 1
        elif action == 1 and self.Ptx > Ptx_min:
            self.Ptx -= 1
        elif action == 2 and self.Ret < Ret_max:
            self.Ret += pi / 180
        elif action == 3 and self.Ret > Ret_min:
            self.Ret -= pi / 180
        reward, R_avg, snr = self.reward(self.Ret, self.Ptx)
        reward = reward / self.base
        RSRP_mean = self.RSRP.mean()
        state_next = np.hstack((self.Ret, self.Ptx, RSRP_mean))
        return state_next, reward, R_avg, snr, RSRP_mean

    def reset(self, is_train, mode, i):
        if is_train:
            if mode == 'weak_coverage':
                MaxRet = random.randint(20, 30)
                MinRet = random.randint(1, MaxRet - 1)
                Ptx = normal(1, 0.02, 1)  # w
            else:
                MaxRet = random.randint(3, 10)
                MinRet = random.randint(1, MaxRet - 1)
                Ptx = normal(30, 3, 1)  # w
            Ret = random.randint(MinRet, MaxRet)
        else:
            Ret = self.Rets[i * num_cell]
            Ptx = self.Ptxs[i * num_cell]
        self.Ret = radians(Ret)
        self.Ptx = 10 * log10(Ptx * 1e3)  # dBm
        self.base, _, _ = self.reward(self.Ret, self.Ptx)
        RSRP_mean = self.RSRP.mean()
        self.state_space = np.hstack((self.Ret, self.Ptx, RSRP_mean))
        self.action_space = np.hstack((self.Ret, self.Ptx))
        return self.state_space


class Environment:
    def __init__(self, is_train: bool, mode: str):
        self.is_train = is_train
        self.mode = mode
        self.env = Env(is_train, mode)
        num_states = self.env.state_space.shape[0]
        num_actions = self.env.action_space.shape[0] * 2
        self.agent = Agent(num_states, num_actions, is_train, mode)

    def run(self):
        reward, R, snr, RSRP = 0.0, 0.0, 0.0, 0.0
        rewards = []
        RSRPs = []
        Rs = []
        snrs = []

        for episode in trange(NUM_EPISODES if self.is_train else num_samples // num_cell):
            state = self.env.reset(self.is_train, self.mode, episode)
            state = torch.from_numpy(state).type(torch.float32)
            state = torch.unsqueeze(state, 0).to(device)


            for step in range(MAX_STEPS):
                action = self.agent.get_action(state, episode).to(device)
                state_next, reward, R, snr, RSRP = self.env.step(action)
                state_next = torch.from_numpy(state_next).type(torch.float32).to(device)
                state_next = torch.unsqueeze(state_next, 0)
                if self.is_train:
                    self.agent.memorize(state, action, state_next, torch.tensor([reward], dtype=torch.float32))
                    self.agent.update_q_function()
                state = state_next
                # if step == MAX_STEPS - 1:
                #     print('%d Episode | Stopped after %d steps | r = %f' % (episode + 1, step + 1, reward))

            if self.is_train and episode % 2 == 0:
                self.agent.update_target_q_function()

            rewards.append(reward)
            RSRPs.append(RSRP)
            Rs.append(R)
            snrs.append(snr)

        else:
            # rewards_avg = np.mean(rewards).item()
            # RSRPs_avg = np.mean(RSRPs).item()
            # Rs_avg = np.mean(Rs).item()
            # snrs_avg = np.mean(snrs).item()
            # print('reward: %f, RSRP: %f, R: %f, snr: %f' % (rewards_avg, RSRPs_avg, Rs_avg, snrs_avg))

            # save the data
            data_save = np.array((rewards, RSRPs, Rs, snrs)).transpose()
            col = ('reward', 'RSRP', 'throughput', 'snr')
            pd_data = pd.DataFrame(data_save, columns=col)
            pd_data.to_csv('../data/data_plot_' + self.mode + '.csv', header=True, columns=col, index=False)

            return np.mean(Rs), np.mean(RSRPs), 1.0


if __name__ == '__main__':
    Environment(is_train=False, mode='weak_coverage').run()