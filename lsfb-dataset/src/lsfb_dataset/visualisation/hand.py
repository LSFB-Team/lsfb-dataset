import matplotlib.pyplot as plt


def plot_hands_1d_position(positions, dim='X', ax=None):
    if ax is None:
        ax = plt.gca()

    ax.set_title(f'Hand position ({dim} axis)')
    ax.plot(positions[:, 0], label='right hand')
    ax.plot(positions[:, 1], label='left hand')
    ax.legend()


def plot_hands_2d_position(positions, axes=None):
    if axes is None:
        _, axes = plt.subplots(2)
    plot_hands_1d_position(positions[:, [0, 2]], dim='X', ax=axes[0])
    plot_hands_1d_position(positions[:, [1, 3]], dim='Y', ax=axes[1])
