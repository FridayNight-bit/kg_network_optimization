"""
Network issue data generates core code
"""
from math import *
from random import sample, randint, randrange, choices, choice
from string import ascii_letters, digits
from typing import Tuple, Union
import numpy as np
from numpy.random import normal
import pandas as pd


mode = ''
label = 0
cpe_te_num = 0
cell_num = 12

du_num = 6
cpe_num = 5
rru_num = du_num
nrcell_num = 2
gnb_num = 3
f = 2.4
col = ("ULMeanNL", "ConnMean", "AttOutExecInterXn", "SuccOutInterXn", "ArfcnUL", "ArfcnDL", "NbrPktDL",
       "NbrPktLossDL", "UpOctUL", "UpOctDL", "BsChannelBwUL", "BsChannelBwDL", "MaxTxPower", "RetTilt",
       "TransRatePeak_mean", "RSRP_mean", "RSRQ_mean", "Label")


class Jwd:
    def __init__(self, jwd_num):
        self.jwd_num = jwd_num

    def get_jw(self, x_center, y_center, d1: float, d2: float, interval=1):
        flag = np.random.choice((-1, 1), self.jwd_num)
        x_edge = x_center.repeat(interval) + flag * np.random.uniform(d1, d2, self.jwd_num)
        y_edge = y_center.repeat(interval) + flag * np.random.uniform(d1, d2, self.jwd_num)
        return x_edge, y_edge


class Terminal:
    def __init__(self, terminal_num=0):
        self.TerminalId = np.array(['' for _ in range(terminal_num)])
        self.Longitude, self.Latitude = np.zeros(terminal_num), np.zeros(terminal_num)

        self.terminal_num = terminal_num

    def get_Terminal(self, x, y, is_cpe=True, interval=3):
        if mode == 'cap' and label == 2:  # 切换类
            self.Longitude, self.Latitude = Jwd(self.terminal_num).get_jw(x, y, 0.0015, 0.002, interval)
        else:
            self.Longitude, self.Latitude = Jwd(self.terminal_num).get_jw(x, y, 0.001, 0.002, interval)
        if is_cpe:
            self.TerminalId = get_Id("Terminal_cpe", self.terminal_num)
        else:
            self.TerminalId = get_Id("Terminal_cpn", self.terminal_num)
        return self

    def save_Terminal(self, is_cpe=True):
        col = ("Id", "Longitude", "Latitude")
        data_save = np.vstack((self.TerminalId, self.Longitude, self.Latitude)).transpose()
        pd_data = pd.DataFrame(data_save, columns=col)
        if is_cpe:
            pd_data.to_csv('data/terminal_new.csv', columns=col, index=False)
        else:
            pd_data.to_csv('data/terminal_new.csv', columns=col, index=False, mode='a', header=False)


class Cpe:
    def __init__(self, cpe_num=0):
        self.CpeId, self.CpeName, self.SupportedType = np.array(['' for _ in range(cpe_num)]), \
                                                       np.array(['' for _ in range(cpe_num)]), \
                                                       np.array(['' for _ in range(cpe_num)])
        self.MaxDistance, self.TransRateMean, self.TransRatePeak, self.RSRP, self.RSRQ = np.zeros(cpe_num, dtype=int), \
                                                                                         np.zeros(cpe_num, dtype=int), \
                                                                                         np.zeros(cpe_num, dtype=int), \
                                                                                         np.zeros(cpe_num, dtype=int), \
                                                                                         np.zeros(cpe_num, dtype=int)
        self.Longitude, self.Latitude = np.zeros(cpe_num), np.zeros(cpe_num)
        self.terminals = Terminal()

        self.cpe_num = cpe_num
        self.terminal_num = 3

    def get_Cpe_config_cover_and_cap(self, x_center, y_center, Power, RetTilt, terminal_num):
        self.CpeId = get_Id("Cpe", self.cpe_num)
        self.CpeName = get_Name("Cpe", self.cpe_num)
        self.MaxDistance = np.round(np.random.normal(100, 10, self.cpe_num))
        self.SupportedType = get_Supported_Type(self.cpe_num)
        if mode == "cap" and label == 2:  # 切换类
            self.Longitude, self.Latitude = Jwd(self.cpe_num).get_jw(x_center, y_center, 0.002, 0.003, 10)
        else:
            self.Longitude, self.Latitude = Jwd(self.cpe_num).get_jw(x_center, y_center, 0.001, 0.003, 10)
        lon = x_center.repeat(10)
        lat = y_center.repeat(10)
        pw = (Power / (cpe_num + terminal_num)).repeat(5)
        bw = (nrs.BsChannelBwDL / (cpe_num + terminal_num)).repeat(5)
        distance = Tools().get_Distance(lon, lat, self.Longitude, self.Latitude)
        if mode == "cover":
            self.RSRP = np.round(10 * np.log10(pw * 1e3) - path_loss(distance, RetTilt.repeat(5)))
            self.TransRatePeak = shannon(bw, self.RSRP, True)
            # if label == 0 or label == 4:
            #
            # elif label == 1:
            #     self.RSRP = np.round(np.random.normal(-105, 1.5, self.cpe_num) * (1 + 0.1 * (lamuda - 1)))
            # else:
            #     self.RSRP = np.round(np.random.normal(-85, 1.5, self.cpe_num) * (1 + 0.1 * (lamuda - 1)))
            self.RSRQ = np.round(np.random.normal(-11.25, 1, self.cpe_num))
        # elif mode == "cap":
        #     bw = nrs.BsChannelBwDL.repeat(5)
        #     self.TransRatePeak = \
        #         np.round(np.random.normal(500, 20, self.cpe_num) * (1 + 0.2 * (1 - lamuda)) * (bw / 40))
        #     if label == 1:
        #         self.RSRP = np.round(np.random.normal(-85, 1.5, self.cpe_num) * (1 + 0.1 * (lamuda - 1)))
        #         self.RSRQ = np.round(-10 * (1 + 0.4 * (lamuda - 1)))
        #     elif label == 2:
        #         self.RSRP = np.round(np.random.normal(-95, 1.5, self.cpe_num) * (1 + 0.1 * (lamuda - 1)))
        #         self.RSRQ = np.round(-15 * (1 + 0.4 * (lamuda - 1)))
        #     else:
        #         self.RSRP = np.round(np.random.normal(-95, 1.5, self.cpe_num) * (1 + 0.1 * (lamuda - 1)))
        #         self.RSRQ = np.round(-10 * (1 + 0.4 * (lamuda - 1)))
        self.TransRateMean = np.round(self.TransRatePeak * np.random.normal(0.5, 0.03, self.cpe_num))
        TransRatePeak_mean = self.TransRatePeak.reshape(cell_num, 5).mean(axis=1)
        RSRP_mean = self.RSRP.reshape(cell_num, 5).mean(axis=1)
        RSRQ_mean = self.RSRQ.reshape(cell_num, 5).mean(axis=1)
        self.terminals = Terminal(self.terminal_num * self.cpe_num).get_Terminal(self.Longitude, self.Latitude)
        return self, TransRatePeak_mean, RSRP_mean, RSRQ_mean

    def get_Cpe_config_inter(self, x_center, y_center):
        self.CpeId = get_Id("Cpe", self.cpe_num)
        self.CpeName = get_Name("Cpe", self.cpe_num)
        self.MaxDistance = np.round(np.random.normal(100, 10, self.cpe_num))
        self.SupportedType = get_Supported_Type(self.cpe_num)
        self.Longitude, self.Latitude = Jwd(self.cpe_num).get_jw(x_center, y_center, 0.001, 0.003, 10)
        self.terminals = Terminal(self.terminal_num * self.cpe_num).get_Terminal(self.Longitude, self.Latitude)
        return self

    def get_Cpe_perform_inter(self, lamuda_interference, MaxTxPower):
        lon = dus.Longitude.repeat(10)
        lat = dus.Latitude.repeat(10)
        pw = MaxTxPower.repeat(5)

        lamuda_coverage = (Tools().get_Distance(lon, lat, cpns.cpes.Longitude, cpns.cpes.Latitude) / 0.28) * (240.0 / pw)
        cpns.cpes.TransRatePeak = np.round(np.random.normal(500, 10, self.cpe_num) * (1 + 0.01 * (1 - lamuda_coverage)))
        cpns.cpes.TransRateMean = np.round(cpns.cpes.TransRatePeak * np.random.normal(0.5, 0.03, self.cpe_num))
        cpns.cpes.RSRP = np.round(np.random.normal(-95, 1.5, self.cpe_num) * (1 + 0.01 * (lamuda_coverage - 1)))
        lamuda = lamuda_interference.repeat(5)
        if label == 3:
            cpns.cpes.RSRQ = np.round(-18 * (1 - 3 * lamuda))
        else:
            cpns.cpes.RSRQ = np.round(-12 * (1 - 3 * lamuda))
        TransRatePeak_mean = cpns.cpes.TransRatePeak.reshape(cell_num, 5).mean(axis=1)
        RSRP_mean = cpns.cpes.RSRP.reshape(cell_num, 5).mean(axis=1)
        RSRQ_mean = cpns.cpes.RSRQ.reshape(cell_num, 5).mean(axis=1)
        return TransRatePeak_mean, RSRP_mean, RSRQ_mean

    def save_Cpe(self):
        col = ("Id", "CpeName", "MaxDistance", "SupportedType", "Longitude", "Latitude", "TransRateMean",
               "TransRatePeak", "RSRP", "RSRQ")
        data_save = np.vstack((self.CpeId, self.CpeName, self.MaxDistance, self.SupportedType, self.Longitude,
                               self.Latitude, self.TransRateMean, self.TransRatePeak, self.RSRP, self.RSRQ)).transpose()
        pd_data = pd.DataFrame(data_save, columns=col)
        pd_data.to_csv('data/cpe_new.csv', columns=col, index=False)
        self.terminals.save_Terminal()


class CpnSubNetwork:
    def __init__(self, cpn_num=0):
        self.CpnSNId, self.CpnSNName = np.array(['' for _ in range(cpn_num)]), np.array(['' for _ in range(cpn_num)])
        self.cpes, self.terminals = Cpe(), Terminal()

        self.cpn_num = cpn_num
        self.cpe_num = 5

        self.RSRP_mean, self.RSRQ_mean, self.TransRate_mean = np.zeros(cpn_num, dtype=int), \
                                                              np.zeros(cpn_num, dtype=int), \
                                                              np.zeros(cpn_num, dtype=int),

    def get_Cpn(self, x_center, y_center, Power, RetTilt):
        self.CpnSNId = get_Id("Cpn", self.cpn_num)
        self.CpnSNName = get_Name("Cpn", self.cpn_num)
        if mode == "cover":
            if label == 1:
                terminal_num = randint(2, 5)
            elif label == 2 or label == 3:
                terminal_num = randint(10, 15)
            else:
                terminal_num = randint(5, 10)
            cpes, self.TransRate_mean, self.RSRP_mean, self.RSRQ_mean = Cpe(self.cpn_num * self.cpe_num).\
                get_Cpe_config_cover_and_cap(x_center, y_center, Power, RetTilt, terminal_num)
        elif mode == "cap":
            if label == 1:
                terminal_num = 15
            else:
                terminal_num = 7
            cpes, self.TransRate_mean, self.RSRP_mean, self.RSRQ_mean = \
                Cpe(self.cpn_num * self.cpe_num).get_Cpe_config_cover_and_cap(x_center, y_center, Power, RetTilt)
        else:
            terminal_num = 7
            cpes = Cpe(self.cpn_num * self.cpe_num).get_Cpe_config_inter(x_center, y_center)
        self.cpes = cpes
        global cpe_te_num
        cpe_te_num = self.cpes.terminals.terminal_num
        self.terminals = Terminal(terminal_num * self.cpn_num).get_Terminal(x_center, y_center, False, terminal_num * 2)
        global cpns
        cpns = self

    def get_Cpn_perform_inter(self, lamuda, MaxTxPower):
        cpns.TransRate_mean, cpns.RSRP_mean, cpns.RSRQ_mean = \
            Cpe(self.cpn_num * self.cpe_num).get_Cpe_perform_inter(lamuda, MaxTxPower)

    def save_Cpn(self):
        col = ("Id", "CpnSNName")
        data_save = np.vstack((self.CpnSNId, self.CpnSNName)).transpose()
        pd_data = pd.DataFrame(data_save, columns=col)
        pd_data.to_csv('data/cpn_new.csv', columns=col, index=False)
        self.cpes.save_Cpe()
        self.terminals.save_Terminal(False)


class NrCell:
    def __init__(self, nrcell_num=0):
        self.NrCellId = np.array(['' for _ in range(nrcell_num)])
        self.ArfcnDL, self.ArfcnUL, self.ArfcnSUL, self.BsChannelBwDL, self.BsChannelBwUL, self.BsChannelBwSUL, \
        self.ULMeanNL, self.ULMaxNL, self.NbrPktUL, self.NbrPktDL, self.NbrPktLossDL, self.ConnMean, self.ConnMax, \
        self.AttOutExecInterXn, self.SuccOutInterXn = np.zeros(nrcell_num, dtype=int), \
                                                      np.zeros(nrcell_num, dtype=int), np.zeros(nrcell_num, dtype=int), \
                                                      np.zeros(nrcell_num, dtype=int), np.zeros(nrcell_num, dtype=int), \
                                                      np.zeros(nrcell_num, dtype=int), np.zeros(nrcell_num, dtype=int), \
                                                      np.zeros(nrcell_num, dtype=int), np.zeros(nrcell_num, dtype=int), \
                                                      np.zeros(nrcell_num, dtype=int), np.zeros(nrcell_num, dtype=int), \
                                                      np.zeros(nrcell_num, dtype=int), np.zeros(nrcell_num, dtype=int), \
                                                      np.zeros(nrcell_num, dtype=int), np.zeros(nrcell_num, dtype=int)
        self.UpOctUL, self.UpOctDL, self.CellMeanTxPower, self.CellMaxTxPower = np.zeros(nrcell_num), \
                                                                                np.zeros(nrcell_num), \
                                                                                np.zeros(nrcell_num), \
                                                                                np.zeros(nrcell_num)

        self.nrcell_num = nrcell_num
        self.nrcell_main_num = 3
        self.nrcell_sub_num = self.nrcell_num - self.nrcell_main_num

        self.bw1 = (5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100)
        self.dl1 = [randrange(422000, 434000, 20), randrange(386000, 398000, 20), randrange(361000, 376000, 20),
                    randrange(173800, 178800, 20), randrange(524000, 538000, 20), randrange(185000, 192000, 20),
                    randrange(145800, 149200, 20), randrange(151600, 153600, 20), randrange(172000, 175000, 20),
                    randrange(158200, 164200, 20), randrange(386000, 399000, 20), randrange(171800, 178800, 20),
                    randrange(151600, 160600, 20), randrange(470000, 472000, 20), randrange(402000, 405000, 20),
                    randrange(514000, 524000, 20), randrange(376000, 384000, 20), randrange(460000, 480000, 20),
                    randrange(499200, 537999, 3), randrange(499200, 537999, 6), randrange(636667, 646666, 1),
                    randrange(636668, 646666, 2), randrange(286400, 303400, 20), randrange(285400, 286400, 20),
                    randrange(496700, 499000, 20), randrange(422000, 440000, 20), randrange(399000, 404000, 20),
                    randrange(123400, 130400, 20), randrange(295000, 303600, 20), randrange(620000, 680000, 1),
                    randrange(620000, 680000, 2), randrange(620000, 653333, 1), randrange(620000, 653332, 2),
                    randrange(693334, 733333, 1), randrange(693334, 733332, 2), randrange(499200, 538000, 20)]
        self.ul1 = [randrange(384000, 396000, 20), randrange(370000, 382000, 20), randrange(342000, 357000, 20),
                    randrange(164800, 169800, 20), randrange(500000, 514000, 20), randrange(176000, 178300, 20),
                    randrange(139800, 143200, 20), randrange(157600, 159600, 20), randrange(163000, 166000, 20),
                    randrange(166400, 172400, 20), randrange(370000, 383000, 20), randrange(162800, 169800, 20),
                    randrange(140600, 149600, 20), randrange(461000, 463000, 20), randrange(402000, 405000, 20),
                    randrange(514000, 524000, 20), randrange(376000, 384000, 20), randrange(460000, 480000, 20),
                    randrange(499200, 537999, 3), randrange(499200, 537999, 6), randrange(636667, 646666, 1),
                    randrange(636668, 646666, 2), randrange(286400, 303400, 20), randrange(285400, 286400, 20),
                    randrange(496700, 499000, 20), randrange(384000, 402000, 20), randrange(342000, 356000, 20),
                    randrange(339000, 342000, 20), randrange(132600, 139600, 20), randrange(285400, 294000, 20),
                    randrange(620000, 680000, 1), randrange(620000, 680000, 2), randrange(620000, 653333, 1),
                    randrange(620000, 653332, 2), randrange(693334, 733333, 1), randrange(693334, 733332, 2),
                    randrange(499200, 537996, 6), randrange(499200, 538000, 20), randrange(176000, 183000, 20)]
        self.dl1.sort()
        self.ul1.sort()

    def get_Nrcell_config(self):
        self.NrCellId = get_Id("NrCell", self.nrcell_num)
        if mode == "cover":
            if label == 3:
                self.ArfcnDL = np.array(choices(self.dl1[len(self.dl1)//2 - randint(1, self.nrcell_num//2):
                                                len(self.dl1)//2 + randint(1, self.nrcell_num//2)], k=self.nrcell_num))
                self.ArfcnUL = np.array(sample(self.ul1, k=self.nrcell_num))
            elif label == 4:
                self.ArfcnDL = np.array(sample(self.dl1, k=self.nrcell_num))
                self.ArfcnUL = np.hstack(([self.ul1[self.dl1.index(i)] for i in self.ArfcnDL[:self.nrcell_main_num]],
                                          sample(self.ul1, k=self.nrcell_sub_num)))
            else:
                self.ArfcnDL = np.array(sample(self.dl1, k=self.nrcell_num))
                self.ArfcnUL = np.array(sample(self.ul1, k=self.nrcell_num))
            self.BsChannelBwDL = np.array(choices(self.bw1, k=self.nrcell_num))
            self.BsChannelBwUL = np.array(choices(self.bw1, k=self.nrcell_num))
        elif mode == "cap":
            self.ArfcnDL = np.array(sample(self.dl1, k=self.nrcell_num))
            self.ArfcnUL = np.array(sample(self.ul1, k=self.nrcell_num))
            if label == 3:
                self.BsChannelBwDL = np.hstack((
                    choices(self.bw1[:len(self.bw1)//3], k=self.nrcell_main_num),
                    choices(self.bw1, k=self.nrcell_sub_num)))
                self.BsChannelBwUL = np.hstack((
                    choices(self.bw1[:len(self.bw1)//3], k=self.nrcell_main_num),
                    choices(self.bw1, k=self.nrcell_sub_num)))
            else:
                self.BsChannelBwDL = np.array(choices(self.bw1, k=self.nrcell_num))
                self.BsChannelBwUL = np.array(choices(self.bw1, k=self.nrcell_num))
        else:
            if label == 2:
                self.ArfcnUL = np.array(choice(self.ul1)).repeat(self.nrcell_num)
            else:
                self.ArfcnUL = np.array(sample(self.ul1, k=self.nrcell_num))
            self.ArfcnDL = np.array(sample(self.dl1, k=self.nrcell_num))
            self.BsChannelBwDL = np.array(choices(self.bw1, k=self.nrcell_num))
            self.BsChannelBwUL = np.array(choices(self.bw1, k=self.nrcell_num))
        return self

    def get_Nrcell_perform(self):
        te_num = cpe_te_num + cpns.terminals.terminal_num + cpns.cpes.cpe_num
        if mode == "cover":
            nrs.ConnMax = np.round(te_num * np.random.uniform(0.9, 1, self.nrcell_num))
            nrs.ConnMean = np.round(nrs.ConnMax / np.random.normal(2, 0.1, self.nrcell_num))
            nrs.AttOutExecInterXn = np.round(np.random.normal(15, 1, self.nrcell_num) * nrs.ConnMean)
            nrs.UpOctDL = (shannon(nrs.BsChannelBwDL, rrus.antennas.MaxTxPower * 0.1, False) / 8E3) * nrs.ConnMean
            nrs.NbrPktDL = np.round(nrs.UpOctDL / np.random.normal(1, 0.1, self.nrcell_num))
            lamuda_cover_1_first = 5e-7 * np.array(abs(nrs.ArfcnDL[0] - nrs.ArfcnDL[1]))
            lamuda_cover_1_last = 5e-7 * np.array(abs(nrs.ArfcnDL[-1] - nrs.ArfcnDL[-2]))
            lamuda_cover_1 = 5e-7 * np.array([(abs(nrs.ArfcnDL[i] - nrs.ArfcnDL[i + 1]) +
                                               abs(nrs.ArfcnDL[i] - nrs.ArfcnDL[i - 1])) / 2
                                              for i in range(1, self.nrcell_num - 1)])
            lamuda_cover_1 = np.hstack((lamuda_cover_1_first, lamuda_cover_1, lamuda_cover_1_last))
            if label == 1:
                nrs.ULMeanNL = np.hstack((np.round(
                    np.random.normal(-95, 3, self.nrcell_main_num) * (1 + lamuda_cover_1[: self.nrcell_main_num])),
                    np.round(-105 * (1 + lamuda_cover_1[self.nrcell_main_num:]))))
                nrs.UpOctUL = nrs.UpOctDL / np.random.normal(8, 1, self.nrcell_num)
                nrs.SuccOutInterXn = np.round(nrs.AttOutExecInterXn * np.random.uniform(0.9, 1, self.nrcell_num))
                nrs.NbrPktLossDL = np.hstack((
                    np.round(np.random.uniform(0.1, 0.2, self.nrcell_main_num) * nrs.NbrPktDL[:self.nrcell_main_num]),
                    np.round(np.random.uniform(0.05, 0.1, self.nrcell_sub_num) * nrs.NbrPktDL[self.nrcell_main_num:])))
            if label == 3:
                nrs.ULMeanNL = np.round(np.random.normal(-95, 3, self.nrcell_num) * (1 + lamuda_cover_1))
                nrs.UpOctUL = nrs.UpOctDL / np.random.normal(8, 1, self.nrcell_num)
                nrs.SuccOutInterXn = np.round(nrs.AttOutExecInterXn * np.random.uniform(0.9, 1, self.nrcell_num))
                nrs.NbrPktLossDL = np.round(np.random.uniform(0.05, 0.1, self.nrcell_num) * nrs.NbrPktDL)
            elif label == 4:
                lamuda_cover_2 = 5e-7 * abs(nrs.ArfcnUL[:self.nrcell_main_num] - nrs.ArfcnDL[:self.nrcell_main_num])
                nrs.ULMeanNL = np.hstack((
                    np.round(np.random.normal(-95, 3, self.nrcell_main_num) * (1 + lamuda_cover_2)),
                    np.round(-105 * (1 + lamuda_cover_1[self.nrcell_main_num:]))))
                nrs.UpOctUL = np.hstack((
                    nrs.UpOctDL[:self.nrcell_main_num] / np.random.normal(11, 1, self.nrcell_main_num),
                    nrs.UpOctDL[self.nrcell_main_num:] / np.random.normal(8, 1, self.nrcell_sub_num)))
                nrs.SuccOutInterXn = np.hstack((np.round(
                    nrs.AttOutExecInterXn[:self.nrcell_main_num] * np.random.uniform(0.5, 0.9, self.nrcell_main_num)),
                                                np.round(
                    nrs.AttOutExecInterXn[self.nrcell_main_num:] * np.random.uniform(0.9, 1, self.nrcell_sub_num))))
                nrs.NbrPktLossDL = np.round(np.random.uniform(0.05, 0.1, self.nrcell_num) * nrs.NbrPktDL)
            else:
                nrs.ULMeanNL = np.round(-105 * (1 + lamuda_cover_1))
                nrs.UpOctUL = nrs.UpOctDL / np.random.normal(8, 1, self.nrcell_num)
                nrs.SuccOutInterXn = np.round(nrs.AttOutExecInterXn * np.random.uniform(0.9, 1, self.nrcell_num))
                nrs.NbrPktLossDL = np.round(np.random.uniform(0.05, 0.1, self.nrcell_num) * nrs.NbrPktDL)
        elif mode == "cap":
            nrs.ConnMax = np.round(te_num * np.random.uniform(0.95, 1, self.nrcell_num))
            nrs.ConnMean = np.round(nrs.ConnMax / np.random.normal(2, 0.05, self.nrcell_num))
            nrs.UpOctDL = shannon(nrs.BsChannelBwDL, rrus.antennas.MaxTxPower) / \
                          8E3 * np.random.normal(0.5, 0.03, self.nrcell_num) * nrs.ConnMean
            nrs.NbrPktDL = np.round(nrs.UpOctDL / np.random.normal(1, 0.1, self.nrcell_num))
            if label == 2:
                nrs.AttOutExecInterXn = np.round(np.random.normal(30, 1, self.nrcell_num) * nrs.ConnMean)
            else:
                nrs.AttOutExecInterXn = np.round(np.random.normal(15, 1, self.nrcell_num) * nrs.ConnMean)
            nrs.SuccOutInterXn = np.round(nrs.AttOutExecInterXn * np.random.uniform(0.9, 1, self.nrcell_num))
            nrs.UpOctUL = nrs.UpOctDL / np.random.normal(8, 1, self.nrcell_num)
            nrs.ULMeanNL = np.round(np.random.normal(-110, 3, self.nrcell_num))
            nrs.NbrPktLossDL = np.round(np.random.uniform(0, 0.1, self.nrcell_num) * nrs.NbrPktDL)
        else:
            nrs.ConnMax = np.round(te_num * np.random.uniform(0.9, 1, self.nrcell_num))
            nrs.ConnMean = np.round(nrs.ConnMax / np.random.normal(2, 0.1, self.nrcell_num))
            nrs.UpOctDL = shannon(nrs.BsChannelBwDL, rrus.antennas.MaxTxPower) \
                          / 8E3 * np.random.normal(0.5, 0.03, self.nrcell_num) * nrs.ConnMean
            nrs.AttOutExecInterXn = np.round(np.random.normal(15, 1, self.nrcell_num) * nrs.ConnMean)
            nrs.NbrPktDL = np.round(nrs.UpOctDL / np.random.normal(1, 0.1, self.nrcell_num))
            nrs.UpOctUL = nrs.UpOctDL / np.random.normal(8, 1, self.nrcell_num)

            lamuda_inter_first = 2e-7 * np.array(abs(nrs.ArfcnUL[0] - nrs.ArfcnUL[1]))
            lamuda_inter_last = 2e-7 * np.array(abs(nrs.ArfcnUL[-1] - nrs.ArfcnUL[-2]))
            lamuda_inter = 2e-7 * np.array([abs(nrs.ArfcnUL[i] - nrs.ArfcnUL[i + 1]) +
                                            abs(nrs.ArfcnUL[i] - nrs.ArfcnUL[i - 1]) / 2 for i in
                                            range(1, self.nrcell_num - 1)])
            lamuda_inter = np.hstack((lamuda_inter_first, lamuda_inter, lamuda_inter_last))
            if label == 1:
                nrs.ULMeanNL = np.hstack((
                    np.round(np.random.normal(-95, 1, self.nrcell_main_num) * (1 + lamuda_inter[:self.nrcell_main_num])),
                    np.round(np.random.normal(-105, 1, self.nrcell_sub_num) * (1 + lamuda_inter[self.nrcell_main_num:]))))
                nrs.SuccOutInterXn = np.hstack((np.round(
                    nrs.AttOutExecInterXn[:self.nrcell_main_num] * np.random.normal(0.7, 0.03, self.nrcell_main_num)),
                    np.round(
                    nrs.AttOutExecInterXn[self.nrcell_main_num:] * np.random.uniform(0.9, 1, self.nrcell_sub_num))))
                nrs.NbrPktLossDL = np.hstack((
                    np.round(np.random.normal(0.3, 0.03, self.nrcell_main_num) * nrs.NbrPktDL[:self.nrcell_main_num]),
                    np.round(np.random.uniform(0, 0.1, self.nrcell_sub_num) * nrs.NbrPktDL[self.nrcell_main_num:])))
            elif label == 2:
                nrs.ULMeanNL = np.round(np.random.normal(-90, 1, self.nrcell_num) * (1 + lamuda_inter))
                nrs.SuccOutInterXn = np.round(nrs.AttOutExecInterXn * np.random.normal(0.5, 0.03, self.nrcell_num))
                nrs.NbrPktLossDL = np.round(np.random.normal(0.5, 0.03, self.nrcell_num) * nrs.NbrPktDL)
            elif label == 3:
                nrs.ULMeanNL = np.hstack((
                    np.round(np.random.normal(-85, 1, self.nrcell_main_num) * (1 + lamuda_inter[:self.nrcell_main_num])),
                    np.round(np.random.normal(-105, 1, self.nrcell_sub_num) * (1 + lamuda_inter[self.nrcell_main_num:]))))
                nrs.SuccOutInterXn = np.hstack((np.round(
                    nrs.AttOutExecInterXn[:self.nrcell_main_num] * np.random.normal(0.3, 0.03, self.nrcell_main_num)),
                    np.round(
                    nrs.AttOutExecInterXn[self.nrcell_main_num:] * np.random.uniform(0.9, 1, self.nrcell_sub_num))))
                nrs.NbrPktLossDL = np.hstack((
                    np.round(np.random.normal(0.7, 0.03, self.nrcell_main_num) * nrs.NbrPktDL[:self.nrcell_main_num]),
                    np.round(np.random.uniform(0, 0.1, self.nrcell_sub_num) * nrs.NbrPktDL[self.nrcell_main_num:])))
            else:
                nrs.ULMeanNL = np.round(np.random.normal(-105, 1, self.nrcell_num) * (1 + lamuda_inter))
                nrs.SuccOutInterXn = np.round(nrs.AttOutExecInterXn * np.random.uniform(0.9, 1, self.nrcell_num))
                nrs.NbrPktLossDL = np.round(np.random.uniform(0, 0.1, self.nrcell_num) * nrs.NbrPktDL)
            CpnSubNetwork(self.nrcell_num).get_Cpn_perform_inter(lamuda_inter, rrus.antennas.MaxTxPower)
        nrs.ULMaxNL = np.round(nrs.ULMeanNL / np.random.normal(1.2, 0.1, self.nrcell_num))
        nrs.NbrPktUL = np.round(nrs.UpOctUL / np.random.normal(1, 0.1, self.nrcell_num))
        nrs.CellMaxTxPower = rrus.antennas.MaxTxPower / np.random.normal(20, 3, self.nrcell_num)
        nrs.CellMeanTxPower = nrs.CellMaxTxPower * np.random.normal(0.5, 0.03, self.nrcell_num)

    def save_Nrcell(self):
        col = ("Id", "ArfcnDL", "ArfcnUL", "BsChannelBwDL", "BsChannelBwUL", "ULMeanNL",
               "ULMaxNL", "UpOctUL", "UpOctDL", "NbrPktUL", "NbrPktDL", "NbrPktLossDL", "CellMeanTxPower",
               "CellMaxTxPower", "ConnMean", "ConnMax", "AttOutExecInterXn", "SuccOutInterXn")
        data_save = np.vstack((self.NrCellId, self.ArfcnDL, self.ArfcnUL, self.BsChannelBwDL, self.BsChannelBwUL,
                               self.ULMeanNL, self.ULMaxNL, self.UpOctUL, self.UpOctDL, self.NbrPktDL, self.NbrPktUL,
                               self.NbrPktLossDL, self.CellMeanTxPower, self.CellMaxTxPower, self.ConnMean,
                               self.ConnMax, self.AttOutExecInterXn, self.SuccOutInterXn)).transpose()
        pd_data = pd.DataFrame(data_save, columns=col)
        pd_data.to_csv('data/nrcell_new.csv', columns=col, index=False)


class CuFunction:
    def __init__(self, cu_num=0):
        self.CuId, self.CuName = np.array(['' for _ in range(cu_num)]), np.array(['' for _ in range(cu_num)])
        self.Longitude, self.Latitude = np.zeros(cu_num), np.zeros(cu_num)

        self.cu_num = cu_num

    def get_Cu(self):
        self.CuId = get_Id("CU", self.cu_num)
        self.CuName = get_Name("CU", self.cu_num)
        self.Longitude = gnbs.Longitude
        self.Latitude = gnbs.Latitude
        return self

    def save_Cu(self):
        col = ("Id", "CuName", "Longitude", "Latitude")
        data_save = np.vstack((self.CuId, self.CuName, self.Longitude, self.Latitude)).transpose()
        pd_data = pd.DataFrame(data_save, columns=col)
        pd_data.to_csv('data/cu_new.csv', columns=col, index=False)


class CellDu:
    def __init__(self, celldu_num=0):
        self.CellDuId = np.array(['' for _ in range(celldu_num)])
        self.ArfcnDL, self.ArfcnUL, self.ArfcnSUL, self.BsChannelBwDL, self.BsChannelBwUL, self.BsChannelBwSUL = \
            np.zeros(celldu_num, dtype=int), np.zeros(celldu_num, dtype=int), np.zeros(celldu_num, dtype=int), \
            np.zeros(celldu_num, dtype=int), np.zeros(celldu_num, dtype=int), np.zeros(celldu_num, dtype=int)

        self.celldu_num = celldu_num

    def get_Celldu(self):
        self.CellDuId = get_Id("CellDu", self.celldu_num)
        self.ArfcnDL = nrs.ArfcnDL
        self.ArfcnUL = nrs.ArfcnUL
        self.BsChannelBwDL = nrs.BsChannelBwDL
        self.BsChannelBwUL = nrs.BsChannelBwUL
        return self

    def save_Celldu(self):
        col = ("Id", "ArfcnDL", "ArfcnUL", "BsChannelBwDL", "BsChannelBwUL")
        data_save = np.vstack((
            self.CellDuId, self.ArfcnDL, self.ArfcnUL, self.BsChannelBwDL, self.BsChannelBwUL)).transpose()
        pd_data = pd.DataFrame(data_save, columns=col)
        pd_data.to_csv('data/celldu_new.csv', columns=col, index=False)


class DuFunction:
    def __init__(self, du_num=0):
        self.DuId, self.DuName = np.array(['' for _ in range(du_num)]), np.array(['' for _ in range(du_num)])
        self.Longitude, self.Latitude = np.zeros(du_num), np.zeros(du_num)
        self.celldus = CellDu()

        self.du_num = du_num

    def get_Du(self):
        celldu_num = 2
        self.DuId = get_Id("DU", self.du_num)
        self.DuName = get_Name("DU", self.du_num)
        self.Longitude, self.Latitude = Jwd(self.du_num).get_jw(gnbs.Longitude, gnbs.Latitude, 0.004, 0.005, 2)
        self.celldus = CellDu(celldu_num * self.du_num).get_Celldu()
        return self

    def save_Du(self):
        col = ("Id", "DuName", "Longitude", "Latitude")
        data_save = np.vstack((self.DuId, self.DuName, self.Longitude, self.Latitude)).transpose()
        pd_data = pd.DataFrame(data_save, columns=col)
        pd_data.to_csv('data/du_new.csv', columns=col, index=False)
        self.celldus.save_Celldu()
        cpns.save_Cpn()


class GNBFunction:
    def __init__(self, gnb_num=0):
        self.GNBId, self.GNBName = np.array(['' for _ in range(gnb_num)]), np.array(['' for _ in range(gnb_num)])
        self.Longitude, self.Latitude = np.zeros(gnb_num), np.zeros(gnb_num)
        self.nrcells = NrCell()

        self.gnb_num = gnb_num

    def get_Gnb(self, nrcell_num):
        self.GNBId = get_Id("GNB", self.gnb_num)
        self.GNBName = get_Name("GNB", self.gnb_num)
        self.Longitude, self.Latitude = Jwd(self.gnb_num).get_jw(
            np.array(Tools().lon), np.array(Tools().lan), 0.005, 0.006)
        global nrs
        nrs = NrCell(nrcell_num * self.gnb_num).get_Nrcell_config()
        return self

    def save_Gnb(self):
        col = ("Id", "GNBName", "Longitude", "Latitude")
        data_save = np.vstack((self.GNBId, self.GNBName, self.Longitude, self.Latitude)).transpose()
        pd_data = pd.DataFrame(data_save, columns=col)
        pd_data.to_csv('data/gnb_new.csv', columns=col, index=False)
        self.nrcells.save_Nrcell()


class Antenna:
    def __init__(self, antenna_num=0):
        self.AntennaId, self.AntennaName = np.array(['' for _ in range(antenna_num)]), \
                                           np.array(['' for _ in range(antenna_num)])
        self.RetTilt, self.MaxTiltValue, self.MinTiltValue, self.MaxTxPower = np.zeros(antenna_num, dtype=int), \
                                                                              np.zeros(antenna_num, dtype=int), \
                                                                              np.zeros(antenna_num, dtype=int), \
                                                                              np.zeros(antenna_num, dtype=int)

        self.antenna_num = antenna_num
        self.cpe_te_num = 0

    def get_Antenna(self):
        self.AntennaId = get_Id("Antenna", self.antenna_num)
        self.AntennaName = get_Name("Antenna", self.antenna_num)
        if mode == "cover":
            if label == 1:
                self.MaxTxPower = np.round(np.random.normal(1, 0.02, self.antenna_num))     # W
                self.MaxTiltValue = np.random.randint(200, 300, self.antenna_num)  # 0-3600
            elif label == 2:
                self.MaxTxPower = np.round(np.random.normal(30, 3, self.antenna_num))
                self.MaxTiltValue = np.random.randint(30, 100, self.antenna_num)
            elif label == 3:
                self.MaxTxPower = np.round(np.random.normal(30, 3, self.antenna_num))
                self.MaxTiltValue = np.random.randint(30, 100, self.antenna_num)
            else:
                self.MaxTxPower = np.round(np.random.normal(5, 0.1, self.antenna_num))
                self.MaxTiltValue = np.random.randint(100, 200, self.antenna_num)
        elif mode == "cap":
            if label == 1:
                self.MaxTxPower = self.MaxTxPower = np.round(np.random.normal(330, 15, self.antenna_num))
                self.MaxTiltValue = np.random.randint(100, 1800, self.antenna_num)
            else:
                self.MaxTxPower = np.round(np.random.normal(240, 15, self.antenna_num))
                self.MaxTiltValue = np.random.randint(1800, 3600, self.antenna_num)
        else:
            self.MaxTxPower = np.round(np.random.normal(240, 15, self.antenna_num))
            self.MaxTiltValue = np.random.randint(1800, 3600, self.antenna_num)
        self.MinTiltValue = np.random.randint(10, self.MaxTiltValue - 10, self.antenna_num)
        self.RetTilt = np.random.randint(self.MinTiltValue, self.MaxTiltValue, self.antenna_num)
        CpnSubNetwork(self.antenna_num).get_Cpn(dus.Longitude, dus.Latitude, self.MaxTxPower * 1e-4, self.RetTilt)
        return self

    def save_Antenna(self):
        col = ("Id", "AntennaName", "RetTilt", "MaxTilt", "MinTiltValue", "MaxTxPower")
        data_save = np.vstack((self.AntennaId, self.AntennaName, self.RetTilt, self.MaxTiltValue, self.MinTiltValue,
                               self.MaxTxPower)).transpose()
        pd_data = pd.DataFrame(data_save, columns=col)
        pd_data.to_csv('data/antenna_new.csv', columns=col, index=False)


class Rru:
    def __init__(self, rru_num=0, celldu_num=2):
        self.RruId, self.RruName = np.array(['' for _ in range(rru_num)]), np.array(['' for _ in range(rru_num)])
        self.relatedCellDuList = np.array(['' for _ in range(rru_num * celldu_num)]).reshape(rru_num, celldu_num)
        self.MeanTxPower, self.MaxTxPower, self.MeanPower = np.zeros(rru_num), np.zeros(rru_num), np.zeros(rru_num)
        self.antennas = Antenna()

        self.rru_num = rru_num
        self.celldu_num = celldu_num

        self.VendorName, self.SerialNumber, self.VersionNumber, self.DateOfLastService = '', '', '', ''
        self.FreqBand = ()

    def get_Rru_config(self):
        antenna_num = 2
        self.RruId = get_Id("Rru", self.rru_num)
        self.RruName = get_Name("Rru", self.rru_num)
        self.relatedCellDuList = dus.celldus.CellDuId.reshape(self.rru_num, self.celldu_num)
        antennas = Antenna(self.rru_num * antenna_num).get_Antenna()
        self.antennas = antennas
        return self

    def get_Rru_perform(self):
        rrus.MeanTxPower = getTxPower("mean")
        rrus.MaxTxPower = getTxPower("max")
        rrus.MeanPower = rrus.MeanTxPower * np.random.normal(10, 0.1, self.rru_num)

    def save_Rru(self):
        col = ("Id", "RruName", "relatedCellDuList1", "relatedCellDuList2", "MeanTxPower", "MaxTxPower", "MeanPower")
        data_save = np.vstack((self.RruId, self.RruName, self.relatedCellDuList[:, 0].transpose(),
                               self.relatedCellDuList[:, 1].transpose(), self.MeanTxPower, self.MaxTxPower,
                               self.MeanPower)).transpose()
        pd_data = pd.DataFrame(data_save, columns=col)
        pd_data["relatedCellDuList"] = pd_data["relatedCellDuList1"] + ' ' + pd_data["relatedCellDuList2"]
        col = ("Id", "RruName", "relatedCellDuList", "MeanTxPower", "MaxTxPower", "MeanPower")
        pd_data.to_csv('data/rru_new.csv', columns=col, index=False)
        self.antennas.save_Antenna()


class RanSubNetwork:
    def __init__(self, ran_num):
        self.RanSNId, self.RanSNName = np.array(['' for _ in range(ran_num)]), np.array(['' for _ in range(ran_num)])
        self.gnbs, self.dus, self.cus, self.rrus = GNBFunction(), DuFunction(), CuFunction(), Rru()

        self.ran_num = ran_num

    def get_Ran(self, mode_, label_):
        global mode
        mode = mode_
        global label
        label = label_
        du_num = 2
        gnb_num = 3
        rru_num = du_num
        cu_num = gnb_num
        nrcell_num = 4
        self.RanSNId = get_Id("Ran", self.ran_num)
        self.RanSNName = get_Name("Ran", self.ran_num)
        global gnbs
        gnbs = GNBFunction(gnb_num * self.ran_num).get_Gnb(nrcell_num)
        self.gnbs = gnbs
        global dus
        dus = DuFunction(self.ran_num * du_num * gnb_num).get_Du()
        self.dus = dus
        self.cus = CuFunction(self.ran_num * cu_num).get_Cu()
        global rrus
        rrus = Rru(self.ran_num * rru_num * gnb_num).get_Rru_config()
        NrCell(nrcell_num * gnb_num).get_Nrcell_perform()
        self.gnbs.nrcells = nrs
        Rru(self.ran_num * rru_num * gnb_num).get_Rru_perform()
        self.rrus = rrus
        return self

    def save_Ran(self):
        col = ("Id", "RanSNName")
        data_save = np.vstack((self.RanSNId, self.RanSNName)).transpose()
        pd_data = pd.DataFrame(data_save, columns=col)
        pd_data.to_csv('data/ran_new.csv', columns=col, index=False)
        self.gnbs.save_Gnb()
        self.dus.save_Du()
        self.cus.save_Cu()
        self.rrus.save_Rru()


gnbs = GNBFunction()
dus = DuFunction()
rrus = Rru()
cpns = CpnSubNetwork()
nrs = NrCell()


def get_Id(name: str, num: int):
    return np.array([name + "-" + "0" * (3 - len(str(i + 1))) + str(i + 1) for i in range(num)])


def get_Name(name: str, num: int):
    return np.array([name + "-" + ''.join(sample(ascii_letters + digits, 3)) for _ in range(num)])


def get_Supported_Type(num: int):
    st = ("4G", "5G", "6G")
    return np.array(choices(st, k=num))


def Mean(res: int, val: int, count: int, cpe_num: int) -> Union[float, int]:
    if count == (cpe_num - 1):
        return (res + val) / cpe_num
    else:
        return res + val


def shannon(bw, power, is_db: bool):
    if not is_db:
        power = 10 * np.log10(power * 1e3)  # dBm
    return bw * 1E6 * np.log2(1 + np.power(10, (power + 174 - 10 * np.log10(bw * 1E6)) / 10))


def getTxPower(s: str):
    if s == "mean":
        return nrs.CellMeanTxPower.reshape(6, 2).sum(axis=1)
    else:
        return nrs.CellMaxTxPower.reshape(6, 2).sum(axis=1)


def path_loss(distance, RetTilt):
    return 22 * np.log10(distance * 1e3 / (np.cos(RetTilt * pi / 1800))) + 20 * log10(f) + 32.0   # m, GHz, dB


def save_Data(is_head: bool, train: str):
    nr = nrs
    an = rrus.antennas
    cpn = cpns
    data_save = np.vstack((nr.ULMeanNL, nr.ConnMean, nr.AttOutExecInterXn, nr.SuccOutInterXn, nr.ArfcnUL,
                           nr.ArfcnDL, nr.NbrPktDL, nr.NbrPktLossDL, nr.UpOctUL, nr.UpOctDL, nr.BsChannelBwUL,
                           nr.BsChannelBwDL, an.MaxTxPower, an.RetTilt, cpn.TransRate_mean, cpn.RSRP_mean,
                           cpns.RSRQ_mean, np.array(label).repeat(cell_num))).transpose()
    pd_data = pd.DataFrame(data_save, columns=col)
    pd_mode = 'w' if is_head else 'a'
    pd_data.to_csv('data/data_' + train + 'new.csv', mode=pd_mode, header=is_head, columns=col, index=False)


def save_Relation(ran: RanSubNetwork):
    col = ('node1', 'relation', 'node2')
    rel1 = np.array(['Contain'])
    rel2 = np.array(['Access (Mid-transmission)'])
    rel3 = np.array(['Cover'])
    rel4 = np.array(['Access (Pre-transmission)'])
    rel5 = np.array(['D2D cooperate'])
    ran_rel1 = np.vstack((ran.RanSNId.repeat(gnbs.gnb_num), rel1.repeat(gnbs.gnb_num), gnbs.GNBId))
    ran_rel2 = np.vstack((ran.RanSNId.repeat(dus.du_num), rel1.repeat(dus.du_num), dus.DuId))
    ran_rel3 = np.vstack((ran.RanSNId.repeat(ran.cus.cu_num), rel1.repeat(ran.cus.cu_num), ran.cus.CuId))
    ran_rel4 = np.vstack((ran.RanSNId.repeat(rrus.rru_num), rel1.repeat(rrus.rru_num), rrus.RruId))
    gnb_rel = np.vstack((gnbs.GNBId.repeat(nrs.nrcell_num//gnbs.gnb_num), rel1.repeat(nrs.nrcell_num), nrs.NrCellId))
    du_rel1 = np.vstack((
        dus.DuId.repeat(dus.celldus.celldu_num//dus.du_num), rel1.repeat(dus.celldus.celldu_num), dus.celldus.CellDuId))
    du_rel2 = np.vstack((dus.DuId, rel2.repeat(dus.du_num), ran.cus.CuId.repeat(dus.du_num//ran.cus.cu_num)))
    rru_rel1 = np.vstack((rrus.RruId.repeat(nrs.nrcell_num//rrus.rru_num), rel3.repeat(nrs.nrcell_num), nrs.NrCellId))
    rru_rel2 = np.vstack((rrus.RruId, rel4.repeat(rrus.rru_num), dus.DuId))
    rru_rel3 = np.vstack((rrus.RruId.repeat(rrus.antennas.antenna_num//rrus.rru_num),
                          rel1.repeat(rrus.antennas.antenna_num), rrus.antennas.AntennaId))
    nr_rel = np.vstack((nrs.NrCellId, rel1.repeat(nrs.nrcell_num), cpns.CpnSNId))
    cpn_rel1 = np.vstack((cpns.CpnSNId.repeat(cpns.terminals.terminal_num//cpns.cpn_num),
                         rel1.repeat(cpns.terminals.terminal_num), cpns.terminals.TerminalId))
    cpn_rel2 = np.vstack((
        cpns.CpnSNId.repeat(cpns.cpes.cpe_num//cpns.cpn_num), rel1.repeat(cpns.cpes.cpe_num), cpns.cpes.CpeId))
    te_rel = np.vstack((cpns.terminals.TerminalId[:cpns.terminals.terminal_num//2],
                        rel5.repeat(cpns.terminals.terminal_num//2),
                        cpns.terminals.TerminalId[cpns.terminals.terminal_num//2:]))
    cpe_rel = np.vstack((cpns.cpes.CpeId.repeat(cpns.cpes.terminals.terminal_num//cpns.cpes.cpe_num),
                         rel1.repeat(cpns.cpes.terminals.terminal_num), cpns.cpes.terminals.TerminalId))

    # data_save = np.hstack((ran_rel1, ran_rel2, ran_rel3, ran_rel4, gnb_rel, du_rel1, du_rel2, rru_rel1, rru_rel2,
    #                        rru_rel3, nr_rel, cpn_rel1, cpn_rel2, te_rel, cpe_rel)).transpose()
    # pd_data = pd.DataFrame(data_save, columns=col)
    # pd_data.to_csv('relation.csv', columns=col, index=False, encoding='utf_8_sig')


    data_save1 = np.hstack((ran_rel1, ran_rel2, ran_rel3, ran_rel4, gnb_rel, du_rel1,
                            rru_rel3, nr_rel, cpn_rel1, cpn_rel2, cpe_rel)).transpose()
    pd_data1 = pd.DataFrame(data_save1, columns=col)
    pd_data1.to_csv('data/relation1.csv', columns=col, index=False, encoding='utf_8_sig')
    data_save2 = du_rel2.transpose()
    pd_data2 = pd.DataFrame(data_save2, columns=col)
    pd_data2.to_csv('data/relation2.csv', columns=col, index=False, encoding='utf_8_sig')
    data_save3 = rru_rel1.transpose()
    pd_data3 = pd.DataFrame(data_save3, columns=col)
    pd_data3.to_csv('data/relation3.csv', columns=col, index=False, encoding='utf_8_sig')
    data_save4 = rru_rel2.transpose()
    pd_data4 = pd.DataFrame(data_save4, columns=col)
    pd_data4.to_csv('data/relation4.csv', columns=col, index=False, encoding='utf_8_sig')
    data_save5 = te_rel.transpose()
    pd_data5 = pd.DataFrame(data_save5, columns=col)
    pd_data5.to_csv('data/relation5.csv', columns=col, index=False, encoding='utf_8_sig')


class Tools:
    def __init__(self):
        self.EARTH_RADIUS, self.lon, self.lan = 6378.137, 116.39, 39.9
        self.plmn = ("46000", "46002", "46004", "46007", "46001", "46006", "46009", "46003", "46005", "46011")

    def get_Distance(self, lng1, lat1, lng2, lat2):
        radLat1 = lat1 * pi / 180.0
        radLat2 = lat2 * pi / 180.0
        a = radLat1 - radLat2
        b = lng1 * pi / 180.0 - lng2 * pi / 180.0
        dst = 2 * np.arcsin(
            (np.sqrt(np.power(np.sin(a / 2), 2) + np.cos(radLat1) * np.cos(radLat2) * np.power(np.sin(b / 2), 2))))
        dst = dst * self.EARTH_RADIUS
        dst = np.round(dst * 10000) / 10000
        return dst

    def get_Plmn(self) -> str:
        i = randint(1, 3)
        if i == 1:
            return self.plmn[randint(0, 3)]
        elif i == 2:
            return self.plmn[randint(4, 6)]
        else:
            return self.plmn[randint(7, 9)]

    def get_Sna_List(self, i: int) -> Tuple[str]:
        s = self.get_Plmn()
        return tuple([s + str(randint(0, 255)) + str(randint(0, 16777215)) for _ in range(i)])

    def get_Plmn_List(self, s: str) -> Tuple[str]:
        i = self.plmn.index(s)
        if i <= 3:
            return self.plmn[:4]
        elif i <= 6:
            return self.plmn[4:7]
        else:
            return self.plmn[7:]
