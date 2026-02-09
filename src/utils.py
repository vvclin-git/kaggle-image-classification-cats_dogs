"""Utility helpers."""

import matplotlib.pyplot as plt


def plot_hist(hist_list, figsize=(12, 4), show=True):
    # Normalize input
    if isinstance(hist_list, dict):
        hist_list = [hist_list]

    if not hist_list:
        print("No data to plot.")
        return None, []

    # Keys to plot (list values with data)
    first = hist_list[0]
    keys = [k for k, v in first.items() if isinstance(v, list) and len(v) > 0]
    if not keys:
        print("No data to plot.")
        return None, []

    epochs = range(1, len(first[keys[0]]) + 1)

    fig, axes = plt.subplots(1, len(keys), figsize=figsize)
    if len(keys) == 1:
        axes = [axes]

    multi = len(hist_list) > 1

    for ax, key in zip(axes, keys):
        for i, h in enumerate(hist_list):
            series = h.get(key)
            if isinstance(series, list) and len(series) > 0:
                ax.plot(epochs, series, label=f"Fold {i+1}" if multi else key)
        ax.set_xlabel("Epochs")
        ax.set_ylabel(key)
        ax.set_title(key)
        ax.legend()

    fig.tight_layout()
    if show:
        plt.show()

    return fig, axes
