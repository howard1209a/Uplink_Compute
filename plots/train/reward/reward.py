import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]


def smooth_data(data, window_size=10):
    if len(data) < window_size:
        return data
    smoothed = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    padded = np.concatenate([data[:window_size - 1], smoothed])
    return padded


def load_and_group_files(folder_path, param_types):
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

    grouped_data = {param_type: {} for param_type in param_types}

    for filename in all_files:
        name_without_ext = os.path.splitext(filename)[0]

        for param_type in param_types:
            if filename.startswith(f"{param_type}_"):
                param_value = name_without_ext.split('_')[1]
                file_path = os.path.join(folder_path, filename)
                data = np.load(file_path)

                if data.ndim > 1:
                    data = data.flatten()

                grouped_data[param_type][param_value] = data
                break

    return grouped_data


def plot_individual_param_plots(
        folder_path,
        window_size,
        save_folder,
        colors,
        legend_loc,
        y_limits,
        font_size=10.5,
        figsize=(4, 2.5),
        line_width=1.5,
        legend_fontsize=8.5,
        dpi=300
):
    param_types = ['lr', 'gamma', 'clip', 'dim']

    grouped_data = load_and_group_files(folder_path, param_types)

    os.makedirs(save_folder, exist_ok=True)

    param_labels = {
        'lr': {
            '0.00002': '学习率=2e-5',
            '0.0001': '学习率=1e-4',
            '0.0005': '学习率=5e-4'
        },
        'gamma': {
            '0.8': '奖励衰减=0.8',
            '0.95': '奖励衰减=0.95',
            '0.99': '奖励衰减=0.99'
        },
        'clip': {
            '0.05': '剪裁系数=0.05',
            '0.2': '剪裁系数=0.2',
            '0.5': '剪裁系数=0.5'
        },
        'dim': {
            '256': '神经元数=256',
            '512': '神经元数=512',
            '1024': '神经元数=1024'
        }
    }

    for param_type in param_types:
        param_data = grouped_data[param_type]

        fig, ax = plt.subplots(figsize=figsize)

        sorted_values = sorted(param_data.keys(), key=lambda x: float(x) if param_type != 'dim' else int(x))

        for i, param_value in enumerate(sorted_values):
            data = smooth_data(param_data[param_value], window_size)
            episodes = np.arange(len(data))
            label = param_labels.get(param_type, {}).get(param_value, param_value)

            ax.plot(episodes, data,
                    color=colors[i],
                    linewidth=line_width,
                    label=label)

        ax.set_xlabel('训练轮次', fontsize=font_size)
        ax.set_ylabel('奖励值', fontsize=font_size)

        ax.set_ylim(y_limits[param_type])

        ax.grid(True, alpha=0.2, linestyle='--')

        ax.legend(fontsize=legend_fontsize, loc=legend_loc, frameon=False)

        plt.tight_layout()
        filename = f"{param_type}.png"
        filepath = os.path.join(save_folder, filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    custom_y_limits = {
        'lr': (7, 30),
        'gamma': (7, 30),
        'clip': (7, 40),
        'dim': (7, 30)
    }

    saved_files = plot_individual_param_plots(
        folder_path="./data",
        window_size=30,
        save_folder='reward',
        colors=plt.cm.tab10(np.linspace(0, 1, 3)),
        legend_loc='upper left',
        y_limits=custom_y_limits,
    )
