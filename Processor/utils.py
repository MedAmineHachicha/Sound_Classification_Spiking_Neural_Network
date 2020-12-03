import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch

def txt2list(filename):
    lines_list = []
    with open(filename, 'r') as txt:
        for line in txt:
            lines_list.append(line.rstrip('\n'))
    return lines_list

def plot_spk_rec(spk_rec, idx):

    nb_plt = len(idx)
    d = int(np.sqrt(nb_plt))
    gs = GridSpec(d,d)
    fig= plt.figure(figsize=(30,20),dpi=150)
    for i in range(nb_plt):
        plt.subplot(gs[i])
        plt.imshow(spk_rec[idx[i]].T.astype(int),cmap=plt.cm.gray_r, origin="lower", aspect='auto')
        if i==0:
            plt.xlabel("Time")
            plt.ylabel("Units")


def plot_mem_rec(mem, idx):

    nb_plt = len(idx)
    d = int(np.sqrt(nb_plt))
    dim = (d, d)

    gs=GridSpec(*dim)
    plt.figure(figsize=(30,20))
    dat = mem[idx]

    for i in range(nb_plt):
        if i==0: a0=ax=plt.subplot(gs[i])
        else: ax=plt.subplot(gs[i],sharey=a0)
        ax.plot(dat[i])
