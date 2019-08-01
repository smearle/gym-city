import math
import sys

import numpy as np
import torch

import matplotlib.pyplot as plt

EPSILON = np.finfo(float).eps


def entropy(distribution, base):
    if base == 2:
        return -distribution @ torch.log2(distribution)
    elif base == math.e:
        return -distribution @ torch.log(distribution)
    elif base == 10:
        return -distribution @ torch.log10(distribution)
    sys.exit("Unsupported base. Please choose {}, {}, or {}."
             .format(2, "e", 10))


def plot_entropies(entropies):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    for axis in axes:
        axis.set_xlabel('Steps')
        axis.set_ylabel('Value')

    axes[0].plot(entropies)
    axes[0].set_title('H')
    axes[1].plot(np.diff(entropies))
    axes[1].set_title('d/dH(H)')
    fig.tight_layout()
    plt.show()
