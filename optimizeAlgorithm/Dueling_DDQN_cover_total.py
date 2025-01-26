"""
DDQN algorithm
"""
import random
from collections import namedtuple
from math import *
from random import randrange, choices
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.random import normal, randint
from torch import nn
from torch import optim
from classify.NT import num_cell

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
GAMMA = 0.99
MAX_STEPS = 200
NUM_EPISODES = 100
BATCH_SIZE = 32
CAPACITY = 10000
torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)
num_samples = 0
device = 'cpu'

lamuda = 0.7
C1 = 1
C2 = 0.1
bw = 1e7
n0 = pow(10, -17.4)  # mW
num_cells = 4
num_tes = 10
distance_main = np.random.normal(300, 30, num_tes)
distance_sub1 = np.random.normal(500, 30, num_tes)
distance_sub2 = np.random.normal(500, 30, num_tes)
distance_sub3 = np.random.normal(500, 30, num_tes)
f = 2.4
dl = [randrange(422000, 434000, 20), randrange(386000, 398000, 20), randrange(361000, 376000, 20),
      randrange(173800, 178800, 20), randrange(524000, 538000, 20), randrange(185000, 192000, 20),
      randrange(145800, 149200, 20), randrange(151600, 153600, 20), randrange(172000, 175000, 20),
      randrange(158200, 164200, 20), randrange(386000, 399000, 20), randrange(171800, 178800, 20),
      randrange(151600, 160600, 20), randrange(470000, 472000, 20), randrange(402000, 405000, 20),
      randrange(514000, 524000, 20), randrange(376000, 384000, 20), randrange(460000, 480000, 20),
      randrange(286400, 303400, 20), randrange(285400, 286400, 20), randrange(496700, 499000, 20),
      randrange(422000, 440000, 20), randrange(399000, 404000, 20), randrange(123400, 130400, 20),
      randrange(295000, 303600, 20), randrange(499200, 538000, 20)]
ul = [randrange(384000, 396000, 20), randrange(370000, 382000, 20), randrange(342000, 357000, 20),
      randrange(164800, 169800, 20), randrange(500000, 514000, 20), randrange(176000, 178300, 20),
      randrange(139800, 143200, 20), randrange(157600, 159600, 20), randrange(163000, 166000, 20),
      randrange(166400, 172400, 20), randrange(370000, 383000, 20), randrange(162800, 169800, 20),
      randrange(140600, 149600, 20), randrange(461000, 463000, 20), randrange(402000, 405000, 20),
      randrange(514000, 524000, 20), randrange(376000, 384000, 20), randrange(460000, 480000, 20),
      randrange(286400, 303400, 20), randrange(285400, 286400, 20), randrange(496700, 499000, 20),
      randrange(384000, 402000, 20), randrange(342000, 356000, 20), randrange(339000, 342000, 20),
      randrange(132600, 139600, 20), randrange(285400, 294000, 20), randrange(499200, 538000, 20),
      randrange(176000, 183000, 20)]


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
    def __init__(self, num_states, num_actions, is_train):
        self.num_actions = num_actions
        self.num_states = num_states
        self.memory = ReplayMemory(CAPACITY)
        n_in, n_mid, n_out = num_states, 32, num_actions
        if is_train:
            self.main_q_network = Net(n_in, n_mid, n_out)
            self.target_q_network = Net(n_in, n_mid, n_out)
        else:
            self.main_q_network = torch.load('../model/main_q_net_all.pt').to(device)
            self.target_q_network = torch.load('../model/target_q_net_all.pt').to(device)
        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=0.0001)

    def save_net(self):
        torch.save(self.main_q_network, '../model/main_q_net_all.pt')
        torch.save(self.target_q_network, '../model/target_q_net_all.pt')

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
    def __init__(self, num_states, num_actions, is_train):
        self.brain = Brain(num_states, num_actions, is_train)

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        self.brain.update_target_q_network()


class Env:
    def __init__(self, is_train, label: int):
        if not is_train:
            data = pd.read_csv("../data/data_cover_for_dqn_all.csv", header=0,
                               usecols=['MaxTxPower', 'RetTilt', 'ArfcnDL', 'ArfcnUL', 'Label'])
            data = data[data['Label'] == label]
            global num_samples
            num_samples = data.shape[0]
            self.Rets = np.array(data['RetTilt'] / 10)  #
            self.Ptxs = np.array(data['MaxTxPower'])  # W
            self.Arfcn_DLs = np.array(data['ArfcnDL'])
            self.Arfcn_ULs = np.array(data['ArfcnUL'])
        self.reset(is_train, 0)

    def Afrcn_to_Freq(self):
        self.f_dl = 5e-3 * self.Arfcn_DL * 1e-3  # Ghz    arfcn < 599999 / f < 3GHz
        self.f_ul = 5e-3 * self.Arfcn_UL * 1e-3

    def get_RSRP(self, Ptx, RetTilt, distance):
        RSRP_dBm = Ptx - (10 * log10(num_tes * 1e4) + 22 * np.log10(distance / cos(RetTilt)) + 20 * log10(f) + 32.0)
        return np.power(10, RSRP_dBm / 10)  # Ptx:mW; distance:m; f:GHz;

    def get_inter(self):
        self.inter = self.f_dl[0] / (self.f_dl[0] + 1e2 * abs(self.f_dl[0] - self.f_dl[1:])) + \
                     self.f_ul / (self.f_ul + 1e2 * abs(self.f_ul - self.f_dl[0]))

    def get_reward(self, Ret, Ptx):
        self.RSRP_main = self.get_RSRP(Ptx, Ret, distance_main)
        self.RSRP_sub1 = self.get_RSRP(Ptx, Ret, distance_sub1)
        self.RSRP_sub2 = self.get_RSRP(Ptx, Ret, distance_sub2)
        self.RSRP_sub3 = self.get_RSRP(Ptx, Ret, distance_sub3)
        self.RSRP_sub_sum = self.RSRP_sub1 * self.inter[0] + self.RSRP_sub2 * self.inter[1] + self.RSRP_sub3 * self.inter[2]
        self.RSRP_main_avg = self.RSRP_main.mean()
        self.RSRP_sub_sum_avg = self.RSRP_sub_sum.mean()
        self.snr = self.RSRP_main / (n0 * (bw / num_tes) + self.RSRP_sub_sum)
        R = bw / num_tes * np.log2(1 + self.snr)

        R_avg = R.mean()
        R_edge = np.array(sorted(R)[: int(num_tes * 0.3)]).sum()
        reward = lamuda * R_edge + (1 - lamuda) * R_avg
        return reward, R_avg, self.snr.mean()

    def step(self, action):
        Ptx_max = 10 * log10(3e4)
        Ptx_min = 30
        Ret_max = radians(30)
        Ret_min = radians(1)
        if action == 0 and self.Arfcn_DL[0] > 362280:
            self.Arfcn_DL[0] -= 20
        elif action == 1 and self.Arfcn_DL[0] < 499302:
            self.Arfcn_DL[0] += 20
        elif action == 2 and self.Arfcn_UL > 132600:
            self.Arfcn_UL -= 20
        elif action == 3 and self.Arfcn_UL < 733333:
            self.Arfcn_UL += 20
        elif action == 4 and self.Ptx < Ptx_max:
            self.Ptx += 1
        elif action == 5 and self.Ptx > Ptx_min:
            self.Ptx -= 1
        elif action == 6 and self.Ret < Ret_max:
            self.Ret += pi / 180
        elif action == 7 and self.Ret > Ret_min:
            self.Ret -= pi / 180
        self.Afrcn_to_Freq()
        self.get_inter()
        reward, R_avg, snr = self.get_reward(self.Ret, self.Ptx)
        reward = reward / self.base_R
        state_next = np.hstack((self.f_dl, self.f_ul, self.Ptx, self.Ret, self.RSRP_main_avg, self.RSRP_sub_sum_avg))
        #   (9,)
        return state_next, reward, R_avg, snr

    def reset(self, is_train, i):
        if is_train:
            self.Arfcn_DL = np.array(choices(
                dl[len(dl) // 2 - randint(1, 6): len(dl) // 2 + randint(1, 6)], k=12))[:num_cells]
            self.Arfcn_UL = np.array(ul[dl.index(self.Arfcn_DL[0])])
            mode = random.randint(0, 1)
            if mode == 0:
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
            self.Arfcn_DL = self.Arfcn_DLs[i * num_cells: (i + 1) * num_cells]
            self.Arfcn_UL = self.Arfcn_ULs[i * num_cells]
        self.Ret = radians(Ret)
        self.Ptx = 10 * log10(Ptx * 1e3)  # dBm
        self.Afrcn_to_Freq()
        self.get_inter()
        self.base_R, _, _ = self.get_reward(self.Ret, self.Ptx)
        self.state_space = np.hstack((
            self.f_dl, self.f_ul, self.Ptx, self.Ret, self.RSRP_main_avg, self.RSRP_sub_sum_avg))  # squeeze to (9,)
        return self.state_space


class Environment:
    def __init__(self, is_train=False, label=1):
        self.env = Env(is_train, label)
        self.is_train = is_train
        self.label = label
        num_states = self.env.state_space.shape[0]
        num_actions = 4 * 2  # (arfcn_dl, arfcn_ul, ptx, ret) * 2
        self.agent = Agent(num_states, num_actions, is_train)

    def run(self):
        reward, R, snr = 0, 0, 0
        rewards = []
        RSRPs = []
        Rs = []
        snrs = []
        for episode in range(NUM_EPISODES if self.is_train else num_samples // num_cell):
            state = self.env.reset(self.is_train, episode)
            state = torch.from_numpy(state).type(torch.float32)
            state = torch.unsqueeze(state, 0)
            for step in range(MAX_STEPS):
                action = self.agent.get_action(state, episode)
                state_next, reward, R, snr = self.env.step(action)
                state_next = torch.from_numpy(state_next).type(torch.float32)
                state_next = torch.unsqueeze(state_next, 0)
                if self.is_train:
                    self.agent.memorize(state, action, state_next, torch.tensor([reward], dtype=torch.float32))
                    self.agent.update_q_function()
                state = state_next
                if step == MAX_STEPS - 1:
                    print('%d Episode | Finished after %d steps | r = %f' % (episode + 1, step + 1, reward))
            if self.is_train and episode % 2 == 0:
                self.agent.update_target_q_function()
            rewards.append(reward)
            RSRPs.append(10 * log10(state[:, 7]))
            Rs.append(R)
            snrs.append(snr)
        if self.is_train:
            self.agent.brain.save_net()

        else:

            # rewards_avg = np.mean(rewards).item()
            # rewards_max = max(rewards)
            # rewards_min = min(rewards)
            # RSRPs_avg = np.mean(RSRPs).item()
            # RSRPs_max = max(RSRPs)
            # RSRPs_min = min(RSRPs)
            # Rs_avg = np.mean(Rs).item()
            # Rs_max = max(Rs)
            # Rs_min = min(Rs)
            # snrs_avg = np.mean(snrs).item()
            # snr_max = max(snrs)
            # snr_min = min(snrs)
            # print('reward: %f~%f mean: %f, \n RSRP: %f~%f mean: %f, \n R: %f~%f mean: %f, \n snr: %f~%f mean: %f'
            #       % (rewards_min, rewards_max, rewards_avg, RSRPs_min, RSRPs_max, RSRPs_avg, Rs_min, Rs_max, Rs_avg,
            #          snr_min, snr_max, snrs_avg))

            # save the data
            data_save = np.array((rewards, RSRPs, Rs, snrs)).transpose()
            col = ('reward', 'RSRP', 'throughput', 'snr')
            pd_data = pd.DataFrame(data_save, columns=col)
            pd_data.to_csv('../data/data_plot_all' + str(self.label) + '.csv', header=True, columns=col, index=False)



if __name__ == '__main__':
    Environment(is_train=False, label=1).run()