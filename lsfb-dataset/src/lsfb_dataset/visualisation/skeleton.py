import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from typing import Tuple, List, Optional, Literal
import torch

from ..datasets.lsfb_cont.landmarks import POSE_CONNECTIONS, HAND_CONNECTIONS


def _compute_refocus(landmarks, focus_pad):
    """
    Compute the bounds of the frame that contains all the landmarks with a padding.

    Parameters
    ----------
    landmarks (Tensor): Features at an instant t that contains a flat representation
        of the landmarks that are contained in the refocused frame.
    focus_pad (float): The padding used for the refocus.

    Returns
    -------
    x_lim (float, float), y_lim (float, float): Boundaries of the refocused frame
    """
    skeleton = landmarks.view(-1, 2)
    sk_x = skeleton[:, 0]
    sk_y = skeleton[:, 1]
    x_min = torch.min(sk_x)
    x_size = torch.max(sk_x) - x_min
    y_min = torch.min(sk_y)
    y_size = torch.max(sk_y) - y_min
    size = max(x_size, y_size)

    x_lim = x_min - focus_pad, x_min + size + focus_pad
    y_lim = y_min + size + focus_pad, y_min - focus_pad

    return x_lim, y_lim


def plot_skeleton(
        landmarks: torch.Tensor,
        connections: List[Tuple[int, int]],
        x_lim: Optional[float] = None,
        y_lim: Optional[float] = None,
        refocus: bool = False,
        focus_pad: float = 0.02,
        vertex_size: float = 0.01,
        vertex_color='lime',
        edge_color='white',
        background_color='black',
        figsize: Tuple[int, int] = (4, 4),
        show_axis: bool = False,
        ax=None,
):
    """
    Plot a skeleton according to its landmarks (vertices) and its connections (edges).

    Parameters
    ----------
    landmarks: Features at an instant t that contains a flat representation
    connections: Undirected edges between landmarks that represent connections in the skeleton.
        Each value is the index of a landmark.
    x_lim: Minimum and maximum value on the x-axis.
    y_lim: Minimum and maximum value on the y-axis.
    refocus: If True, compute x_lim and y_lim to fit the landmarks.
        Does nothing if x_lim and y_lim are manually set.
    focus_pad: Padding use to calculate the refocused frame.
        Does nothing if refocus is False or if x_lim and y_lim are manually set.
    vertex_size: Size (width and height) of the landmarks (vertices) on the figure.
    vertex_color: Color of the landmarks (vertices) (see matplotlib).
    edge_color: Color of the connections (edges) (see matplotlib).
    background_color: Color of the background in the figure (see matplotlib).
    figsize: Size of the figure (see matplotlib).
    show_axis: If True, shows values on the x and y-axis. Otherwise, hides them.
    ax: If specified, draw on it. Otherwise, create a new one (see matplotlib).

    Returns
    -------
    fig, ax: If the ax is not specified
    ax: Otherwise
    """

    skeleton = landmarks.view(-1, 2)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_facecolor(background_color)
    ax.axes.xaxis.set_visible(show_axis)
    ax.axes.yaxis.set_visible(show_axis)

    sk_x = skeleton[:, 0]
    sk_y = skeleton[:, 1]

    if refocus:
        refocus_x, refocus_y = _compute_refocus(landmarks, focus_pad)
    else:
        refocus_x, refocus_y = (0, 1), (1, 0)

    if x_lim is None:
        x_lim = refocus_x
    if y_lim is None:
        y_lim = refocus_y

    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)

    for v0, v1 in connections:
        line = Line2D(
            [sk_x[v0], sk_x[v1]],
            [sk_y[v0], sk_y[v1]],
            color=edge_color,
            zorder=1,
        )
        ax.add_line(line)

    for x, y in skeleton:
        ax.add_patch(Circle(
            (x, y),
            radius=vertex_size/2,
            facecolor=vertex_color,
            zorder=2,
        ))

    if fig is not None:
        return fig, ax

    return ax


def plot(landmarks, skeleton_list, refocus=False, focus_pad=0.02, ax=None, **kwargs):
    """
    Plot skeletons for a given frame.

    Parameters
    ----------
    landmarks: Features of the frame. May contains coordinates of multiple skeletons.
    skeleton_list: List of skeletons.
    refocus: If True, compute x_lim and y_lim to fit the landmarks of all the skeletons.
    focus_pad: Padding use to calculate the refocused frame.
        Does nothing if refocus is False or if x_lim and y_lim are manually set.
    ax: If specified, draw on it. Otherwise, create a new one (see matplotlib).
    **kwargs: other parameters (see plot_skeleton).
    """

    connections = {
        'hand': HAND_CONNECTIONS,
        'skeleton': POSE_CONNECTIONS,
    }
    num_lm = {
        'hand': 21,
        'skeleton': 23,
    }

    fig = None
    ax = ax

    x_lim, y_lim = None, None
    if refocus:
        x_lim, y_lim = _compute_refocus(landmarks, focus_pad)

    index = 0
    for skeleton in skeleton_list:
        start = index
        index += 2 * num_lm[skeleton]

        if ax is None:
            fig, ax = plot_skeleton(
                landmarks[start:index],
                connections[skeleton],
                x_lim=x_lim,
                y_lim=y_lim,
                refocus=False,
                **kwargs,
            )
        else:
            plot_skeleton(
                landmarks[start:index],
                connections[skeleton],
                x_lim=x_lim,
                y_lim=y_lim,
                refocus=False,
                ax=ax,
                **kwargs,
            )

    if fig is not None:
        return fig, ax

    return ax


def plot_grid(
        landmarks: torch.Tensor,
        skeleton_list: List[Literal['skeleton', 'hand']],
        n_rows: int = 2,
        n_cols: int = 3,
        figsize: Tuple[int, int] = (12, 8),
        **kwargs
):
    """
    Plot a grid of skeletons for randomly selected frames.

    Parameters
    ----------
    landmarks: Sequence of features of size (N, D).
        N is the length of the sequence (number of frames for a video).
        D is the size of features (number of coordinates for one-or-many skeletons).
    skeleton_list: List of skeletons.
    n_rows: Number of rows in the grid
    n_cols: Number of columns in the grid.
    figsize: Size of the figure (see matplotlib).
    **kwargs: other parameters (see plot).
    """
    num_lm = len(landmarks)
    indices = torch.randint(num_lm, (n_rows, n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    for m in range(n_rows):
        for n in range(n_cols):
            plot(landmarks[indices[m, n]], skeleton_list, ax=axes[m, n], **kwargs)


def plot_pose(landmarks, **kwargs):
    return plot_skeleton(landmarks, POSE_CONNECTIONS, **kwargs)


def plot_pose_grid(landmarks, **kwargs):
    return plot_grid(landmarks, ['skeleton'], **kwargs)


def plot_hand(landmarks, **kwargs):
    return plot_skeleton(landmarks, HAND_CONNECTIONS, **kwargs)


def plot_hand_grid(landmarks, **kwargs):
    return plot_grid(landmarks, ['hand'], **kwargs)
