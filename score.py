import numpy as np

paired_sc = -1
unpaired_sc = .1

def paired(x, i, j):
    return np.exp(-paired_sc)

def unpaired(x, i):
    return np.exp(-unpaired_sc)

def paired_log(x, i, j):
    return -paired_sc

def unpaired_log(x, i):
    return -unpaired_sc
