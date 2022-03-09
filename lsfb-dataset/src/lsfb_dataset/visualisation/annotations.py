import matplotlib.pyplot as plt
import numpy as np
from enum import Enum


class Color(Enum):
    ANNOT_GREEN = ('#6ed15a', 1.0)
    ANNOT_GRAY = ('0.4', 0.4)
    PRED_BLUE = ('#4e7ede', 1.0)
    COERC_RED = ('#d4502f', 1.0)


def plot_annotation_vector(annot_vec: list[bool], y_min=0.0, y_max=1.0, ax=None, color=Color.ANNOT_GREEN, label=None):
    frames_nb = len(annot_vec)
    color = color.value

    if ax is None:
        ax = plt.gca()

    for idx in range(frames_nb):
        if annot_vec[idx]:
            ax.axvspan(xmin=idx, xmax=idx + 1,
                       ymin=y_min, ymax=y_max,
                       facecolor=color[0], alpha=color[1], label=label)


def plot_annot_transition_vector(vec: list[int], y_min=0.0, y_max=1.0, ax=None,
                                 talking_color=Color.ANNOT_GREEN, coercion_color=Color.COERC_RED, label=None):
    frames_nb = len(vec)
    talking_color = talking_color.value
    coerc_color = coercion_color.value

    if ax is None:
        ax = plt.gca()

    for idx in range(frames_nb):
        if vec[idx] == 1:
            ax.axvspan(xmin=idx, xmax=idx + 1,
                       ymin=y_min, ymax=y_max,
                       facecolor=talking_color[0], alpha=talking_color[1], label=label)
        elif vec[idx] == 2:
            ax.axvspan(xmin=idx, xmax=idx + 1,
                       ymin=y_min, ymax=y_max,
                       facecolor=coerc_color[0], alpha=coerc_color[1])


def plot_probabilities(proba: list[float], ax=None, color=Color.PRED_BLUE):
    if ax is None:
        ax = plt.gca()

    frames_nb = len(proba)
    color = color.value
    x = np.linspace(0, frames_nb, frames_nb)

    ax.plot(proba, color=color[0], alpha=color[1])
    ax.fill_between(x, proba, step='pre', color=color[0], alpha=0.4 * color[1])


def compare_with_prediction(true_vec, pred_vec, with_transitions=False):
    fig, ax = plt.subplots(figsize=(30, 4))

    if with_transitions:
        plot_annot_transition_vector(true_vec, y_min=0.5, ax=ax, label='Ground truth')
        plot_annot_transition_vector(pred_vec, y_max=0.5, ax=ax, label='Prediction')
    else:
        plot_annotation_vector(true_vec == 1, y_min=0.5, ax=ax, label='Ground truth')
        plot_annotation_vector(pred_vec == 1, y_max=0.5, ax=ax, label='Prediction', color=Color.PRED_BLUE)

    return fig, ax
