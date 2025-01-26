"""
main function for generating network problem data
"""
from tqdm import trange
from dataGenerate.data_new import *


ran_num = 1
loops = 1
mode = "cover"
train = "predict_"

print("data generating")
for i in trange(int(loops)):
    label = randint(0, 4) if mode == "cover" else randint(0, 3)
    # label = 2   # for test
    ran = RanSubNetwork(ran_num).get_Ran(mode, label)
    ran.save_Ran()
    # save_Data(i == 0, train)
    save_Relation(ran)
