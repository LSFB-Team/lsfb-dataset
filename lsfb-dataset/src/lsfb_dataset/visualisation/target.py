import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from ..utils.helpers import get_gaps
from ..utils.target import get_segments


def plot_binary_segmentation(vector, x_lim: tuple[int, int] = None, alpha=1.0, ax=None):
    if x_lim is None:
        x_lim = 0, len(vector)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(30, 4))
        output = fig, ax
    else:
        output = ax

    ax.set_xlim(*x_lim)
    for start, end in get_gaps(vector, 0, 1):
        ax.axvspan(start, end, alpha=alpha)

    return output


def plot_binary_segmentation_with_coarticulation(vector, x_lim: tuple[int, int] = None):
    if x_lim is None:
        x_lim = 0, len(vector)

    fig, ax = plt.subplots(1, 1, figsize=(30, 4))
    ax.set_xlim(*x_lim)

    for start, end in get_gaps(vector, 0, 1):
        plt.axvspan(start, end)

    for start, end in get_gaps(vector, 1, 2):
        plt.axvspan(start, end, color='red')


def plot_lemme_segmentation(
        segmentation: np.ndarray,
        x_lim: tuple[int, int] = None,
        lemmes_nb: int = 100,
        cmap: str = 'twilight_shifted',
):
    if x_lim is None:
        x_lim = 0, len(segmentation)

    norm = mpl.colors.Normalize(vmin=0, vmax=2+lemmes_nb)
    cmap = cm.get_cmap(cmap)

    fig, ax = plt.subplots(1, 1, figsize=(30, 4))
    ax.set_xlim(*x_lim)

    for value, start, end in get_segments(segmentation):
        if value == 0:
            continue
        plt.axvspan(start, end, color=cmap(norm(value)))
