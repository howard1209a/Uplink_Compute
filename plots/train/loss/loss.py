import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]


def smooth_data(data, window_size):
    if len(data) < window_size:
        return data
    smoothed = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    padded = np.concatenate([data[:window_size - 1], smoothed])
    return padded


def plot_loss_curve(
        file_path,
        x_label_name,
        y_label_name,
        y_lim,
        window_size,
        save_path,
        color,
        line_width=1.2,
        figsize=(4, 2.5),
        font_size=10.5,
        dpi=300
):
    data = np.load(file_path)

    smoothed_data = smooth_data(data, window_size)

    fig, ax = plt.subplots(figsize=figsize)

    steps = np.arange(len(smoothed_data))
    ax.plot(steps, smoothed_data,
            linewidth=line_width,
            color=color,
            label='')

    ax.set_xlabel(x_label_name, fontsize=font_size)
    ax.set_ylabel(y_label_name, fontsize=font_size)

    ax.set_ylim(y_lim)

    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_loss_curve(
        file_path="./actor_loss.npy",
        x_label_name="训练轮次",
        y_label_name="Loss",
        y_lim=(-13, -8),
        window_size=10,
        save_path='actor_loss.png',
        color=plt.cm.tab10(np.linspace(0, 1, 3))[0]
    )

    plot_loss_curve(
        file_path="./critic_loss.npy",
        x_label_name="训练轮次",
        y_label_name="Loss",
        y_lim=(-5, 75),
        window_size=10,
        save_path='critic_loss.png',
        color=plt.cm.tab10(np.linspace(0, 1, 3))[1]
    )
