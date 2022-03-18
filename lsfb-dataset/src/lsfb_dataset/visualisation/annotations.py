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

def plot_annotations_prediction(title, y_true, y_pred, likelihood=None):
    seq_len = len(y_true)

    fig, (ax0, ax1) = plt.subplots(2, figsize=(40, 8))
    fig.patch.set_facecolor('#cccaca')
    fig.suptitle(title, fontsize=16)

    talking = np.where(y_true == 1)[0]
    pred_talking = np.where(y_pred == 1)[0]

    ax0.set_title('Ground truth')
    ax0.set_ylim(0, 1)
    ax0.set_xlim(0, seq_len)
    ax0.vlines(talking, ymin=0.0, ymax=1.0, label='Annotations')
    ax0.margins(x=0, y=0)
    ax0.get_yaxis().set_visible(False)
    ax0.legend()

    ax1.set_title('Result')
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0, seq_len)
    ax1.vlines(pred_talking, ymin=0.0, ymax=1.0, color='#ffccab', label='Prediction')
    if likelihood is not None:
        ax1.plot(likelihood, label='Likelihood')
    ax1.margins(x=0, y=0)
    ax1.legend()

    return fig, (ax0, ax1)
