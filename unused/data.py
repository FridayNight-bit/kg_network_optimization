"""
Definition and initialization of NE classes and data(unused)
"""
from random import randrange


class Jwd:
    def __init__(self):
        self.lon, self.lat = 0.0, 0.0


class Temp:
    def __init__(self):
        self.nrs, self.ans = [], []


class Terminal:
    def __init__(self):
        self.TerminalId, self.TerminalType, self.TerminalBrand = '', '', ''
        self.Storage, self.Computing,  = 0, 0
        self.Longitude, self.Latitude = 0.0, 0.0


class Cpe:
    def __init__(self):
        self.CpeId, self.CpeName, self.SupportedType = '', '', ''
        self.MaxDistance, self.TransRateMean, self.TransRatePeak, self.RSRP, self.RSRQ = 0, 0, 0, 0, 0
        self.Longitude, self.Latitude = 0.0, 0.0
        self.terminals = []


class CpnSubNetwork:
    def __init__(self):
        self.CpnSNId, self.CpnSNName = '', ''
        self.cpes, self.terminals = [], []
        self.RSRP_mean, self.RSRQ_mean, self.TransRate_mean = 0, 0, 0


class NrCell:
    def __init__(self):
        self.NrCellId, self.NCGI, self.CellState, self.NrTAC, self.relatedBwp = '', '', 'Unknow', '', ''
        self.S_NSSAIList = ()  # Mcc+Mnc+SST+SD
        self.ArfcnDL, self.ArfcnUL, self.ArfcnSUL, self.BsChannelBwDL, self.BsChannelBwUL, self.BsChannelBwSUL, \
            self.ULMeanNL, self.ULMaxNL, self.NbrPktUL, self.NbrPktDL, self.NbrPktLossDL, self.ConnMean, self.ConnMax, \
            self.AttOutExecInterXn, self.SuccOutInterXn = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        self.UpOctUL, self.UpOctDL, self.CellMeanTxPower, self.CellMaxTxPower = 0.0, 0.0, 0.0, 0.0


class CuFunction:
    def __init__(self):
        self.CuId, self.CuName = '', ''
        self.PLMNIDList = ()  #Mcc+Mnc
        self.Longitude, self.Latitude = 0.0, 0.0
        self.cucps, self.cuups = [], []


class CellDu:
    def __init__(self):
        self.CellDuId, NCGI, CellState, relatedBwp = '', '', 'Unknow', ''
        self.S_NSSAIList = ()
        self.ArfcnDL, self.ArfcnUL, self.ArfcnSUL, self.BsChannelBwDL, self.BsChannelBwUL, \
            self.BsChannelBwSUL = 0, 0, 0, 0, 0, 0


class DuFunction:
    def __init__(self):
        self.DuId, self.DuName = '', ''
        self.Longitude, self.Latitude = 0.0, 0.0
        self.bwps, self.celldus = [], []


class GNBFunction:
    def __init__(self):
        self.GNBId, self.GNBName, self.GNBGId = '', '', ''
        self.Longitude, self.Latitude = 0.0, 0.0
        self.nrcells, self.bwps = [], []


class Antenna:
    def __init__(self):
        self.AntennaId, self.AntennaName, self.SupportedSeq = '', '', ''
        self.RetTilt, self.MaxTiltValue, self.MinTiltValue, self.MaxTxPower = 0, 0, 0, 0
        self.ChannelInfo = ()
        self.beams = []


class Rru:
    def __init__(self):
        self.RruId, self.RruName, self.VendorName, self.SerialNumber, self.VersionNumber, \
            self.DateOfLastService = '', '', '', '', '', ''
        self.relatedCellDuList, self.FreqBand = (), ()
        self.antennas = []
        self.MeanTxPower, self.MaxTxPower, self.MeanPower = 0.0, 0.0, 0.0


class RanSubNetwork:
    def __init__(self):
        self.RanSNId, self.RanSNName = '', ''
        self.gnbs, self.dus, self.cus, self.rrus = [], [], [], []


EARTH_RADIUS, lon, lan = 6378.137, 116.39, 39.9
CellState = ("Unknown", "Idle", "InActive", "Active")
OsType = ("Linux", "windows", "solaris")
VendorName = ("Huawei", "ZTE", "Nokia", "Ericsson")
FreqBand1 = ("n1", "n2", "n3", "n5", "n7", "n8", "n12", "n14", "n18", "n20", "n25", "n26", "n28", "n29", "n30", "n34",
             "n38", "n39", "n40", "n41", "n48", "n50", "n51", "n53", "n65", "n66", "n70", "n71", "n74", "n75", "n76",
             "n77", "n78", "n79", "n80", "n81", "n82", "n83", "n84", "n86", "n89", "n90", "n91", "n92", "n93", "n94",
             "n95")
FreqBand2 = ("n257", "n258", "n260", "n261")
plmn = ("46000", "46002", "46004", "46007", "46001", "46006", "46009", "46003", "46005", "46011")
qci = (1, 2, 3, 4, 5, 6, 7, 8, 9, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 80, 82, 83, 84, 85, 86)
bw1 = (5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100)
# bw2 = (50, 100, 200, 400)
# bw_sul = (5, 10, 15, 20, 25, 30, 40)
dl1 = [randrange(422000, 434000, 20), randrange(386000, 398000, 20), randrange(361000, 376000, 20),
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
# dl2{ getStep(2054166,2104165,1),getStep(2054167,2104165,2),getStep(2016667,2070832,1),
# getStep(2016667,2070831,2),getStep(2229166,2279165,1), getStep(2229167,2279165,2),
# getStep(2070833,2084999,1),getStep(2070833,2084999,2) },
ul1 = [randrange(384000, 396000, 20), randrange(370000, 382000, 20), randrange(342000, 357000, 20),
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
# ul2 = dl2,
# sul{ getStep(342000,357000),getStep(176000,183000),getStep(166400,172400),getStep(140600,149600),
# getStep(384000,396000),getStep(342000,356000),getStep(164800,169800),getStep(402000,405000) }
dl1.sort()
ul1.sort()
cpns, cpes, terminals, gnbs, dus, cus, nrs, celldus, rrus, antennas = [], [], [], [], [], [], [], [], [], []
jws, cpe_te_sizes, labels = [], [], []
ran = RanSubNetwork()
