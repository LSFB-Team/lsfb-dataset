import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from typing import Optional
import numpy as np

from ..datasets.landmark_connections import POSE_CONNECTIONS, HAND_CONNECTIONS


def __compute_refocus(landmarks, focus_pad):
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
    landmarks = np.reshape(landmarks, (-1, 2))
    x = landmarks[:, 0]
    y = landmarks[:, 1]

    x_min = np.min(x)
    x_max = np.max(x)

    y_min = np.min(y)
    y_max = np.max(y)

    x_lim = (x_min - focus_pad, x_max + focus_pad)
    y_lim = (y_max + focus_pad, y_min - focus_pad)

    return x_lim, y_lim


def __compute_xy_lim(
        landmarks,
        aspect_ratio: float,
        *,
        x_lim: Optional[float] = None,
        y_lim: Optional[float] = None,
        refocus: bool = False,
        focus_pad: float = 0.02,
):
    if refocus:
        refocus_x, refocus_y = __compute_refocus(landmarks, focus_pad)
    else:
        refocus_x, refocus_y = (0, 1), (1, 0)

    if x_lim is None:
        x_lim = refocus_x
    if y_lim is None:
        y_lim = refocus_y

    x_size = x_lim[1] - x_lim[0]
    y_size = y_lim[0] - y_lim[1]
    aspect_ratio = (x_size * aspect_ratio / y_size)

    return x_lim, y_lim, aspect_ratio


def __draw_connections(x, y, *, ax, connections, edge_color):
    for v0, v1 in connections:
        line = Line2D(
            [x[v0], x[v1]],
            [y[v0], y[v1]],
            color=edge_color,
            zorder=1,
        )
        ax.add_line(line)


def __draw_vertices(landmarks, *, ax, vertex_size, vertex_color):
    for x, y in landmarks:
        ax.add_patch(Circle(
            (x, y),
            radius=vertex_size/2,
            facecolor=vertex_color,
            zorder=2,
        ))


def __plot_landmarks(
        landmarks,
        connections,
        *,
        ax=None,
        vertex_size: float = 0.01,
        vertex_color='lime',
        edge_color='white',
        background_color='black',
        size: int = 4,
        aspect_ratio=16/9,
        show_axis: bool = False,
        x_lim: Optional[tuple[float, float]] = None,
        y_lim: Optional[tuple[float, float]] = None,
        refocus: bool = False,
        focus_pad: float = 0.02,
):

    landmarks = np.reshape(landmarks, (-1, 2))
    x = landmarks[:, 0]
    y = landmarks[:, 1]

    x_lim, y_lim, aspect_ratio = __compute_xy_lim(
        landmarks,
        aspect_ratio=aspect_ratio,
        x_lim=x_lim,
        y_lim=y_lim,
        refocus=refocus,
        focus_pad=focus_pad,
    )

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(size * aspect_ratio, size))

    ax.set_facecolor(background_color)
    ax.axes.xaxis.set_visible(show_axis)
    ax.axes.yaxis.set_visible(show_axis)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_box_aspect(1/aspect_ratio)

    __draw_vertices(landmarks, ax=ax, vertex_size=vertex_size, vertex_color=vertex_color)
    __draw_connections(x, y, ax=ax, connections=connections, edge_color=edge_color)

    if fig is not None:
        return fig, ax

    return ax


def plot_landmarks(
        landmarks,
        landmark_types: list[str],
        size: int = 4,
        aspect_ratio: float = 16/9,
        refocus: bool = False,
        focus_pad: float = 0.02,
        x_lim: Optional[tuple[float, float]] = None,
        y_lim: Optional[tuple[float, float]] = None,
        ax=None,
        **kwargs
):
    x_lim, y_lim, aspect_ratio = __compute_xy_lim(
        landmarks,
        aspect_ratio=aspect_ratio,
        x_lim=x_lim,
        y_lim=y_lim,
        refocus=refocus,
        focus_pad=focus_pad,
    )

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(size * aspect_ratio, size))

    last_index = 0
    for lm_type in landmark_types:
        if lm_type == 'pose':
            width = 46
            connections = POSE_CONNECTIONS
            vertex_color = 'lime'
            edge_color = 'white'
        elif lm_type == 'hand_left':
            width = 42
            connections = HAND_CONNECTIONS
            vertex_color = 'red'
            edge_color = 'red'
        elif lm_type == 'hand_right':
            width = 42
            connections = HAND_CONNECTIONS
            vertex_color = 'blue'
            edge_color = 'blue'
        else:
            raise ValueError(f'Unknown landmark type [{lm_type}].')

        __plot_landmarks(
            landmarks[last_index:last_index+width],
            connections,
            vertex_color=vertex_color,
            edge_color=edge_color,
            refocus=False,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            **kwargs
        )
        last_index += width

    if fig is not None:
        return fig, ax

    return ax


def plot_grid(
        landmarks,
        landmark_types: list[str],
        grid_size: tuple[int, int],
        indices=None,
        box_ratio: Optional[float] = None,
        figsize=(16, 9),
        **kwargs
):
    n_rows, n_cols = grid_size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)

    if indices is None:
        indices = np.random.randint(landmarks.shape[0], size=(n_rows, n_cols))
        indices = np.sort(indices, axis=None).reshape(n_rows, n_cols)
    elif isinstance(indices, list):
        indices = np.array(indices)

    for row in range(n_rows):
        for col in range(n_cols):
            ax = plot_landmarks(
                landmarks[indices[row, col]],
                landmark_types,
                ax=axes[row, col],
                **kwargs
            )
            if box_ratio is not None:
                ax.set_box_aspect(box_ratio)

    return fig, axes


def plot_pose(pose_landmarks, **kwargs):
    __plot_landmarks(pose_landmarks, POSE_CONNECTIONS, **kwargs)


def plot_hand(hand_landmarks, **kwargs):
    __plot_landmarks(hand_landmarks, HAND_CONNECTIONS, **kwargs)
