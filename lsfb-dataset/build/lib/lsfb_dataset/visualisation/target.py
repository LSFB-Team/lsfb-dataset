import matplotlib.pyplot as plt
from ..utils.helpers import get_gaps


def plot_binary_segmentation(vector, x_lim: tuple[int, int] = None):
    if x_lim is None:
        x_lim = 0, len(vector)

    fig, ax = plt.subplots(1, 1, figsize=(30, 4))
    ax.set_xlim(*x_lim)

    for start, end in get_gaps(vector, 0, 1):
        plt.axvspan(start, end)

    return fig, ax


def plot_binary_segmentation_with_coarticulation(vector, x_lim: tuple[int, int] = None):
    if x_lim is None:
        x_lim = 0, len(vector)

    fig, ax = plt.subplots(1, 1, figsize=(30, 4))
    ax.set_xlim(*x_lim)

    for start, end in get_gaps(vector, 0, 1):
        plt.axvspan(start, end)

    for start, end in get_gaps(vector, 1, 2):
        plt.axvspan(start, end, color='red')
