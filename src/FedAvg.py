import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def AvgW(w,len):
    w_avg = copy.deepcopy(w)
    for k in w_avg.keys():
        w_avg[k] = torch.div(w_avg[k], len)
    return w_avg

def AddW(w,w0):
    w_avg = copy.deepcopy(w)
    for k in w_avg.keys():
        w_avg[k] += w0[k]
    return w_avg