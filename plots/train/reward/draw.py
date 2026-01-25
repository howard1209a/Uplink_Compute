import matplotlib.pyplot as plt
import numpy as np
import os

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def smooth_data(data, window_size=10):
    """使用滑动窗口平均平滑数据"""
    if len(data) < window_size:
        return data
    smoothed = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    padded = np.concatenate([data[:window_size - 1], smoothed])
    return padded


def load_and_group_files(folder_path, param_types):
    """加载并分组npy文件"""
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


def plot_param_comparison_grid(folder_path, param_types=None, window_size=30,
                               line_width=2, figure_ratio=1.2, save_path=None,
                               dpi=300, y_limits=None):
    """
    绘制四参数对比网格图

    参数:
    ----------
    folder_path : str
        包含npy文件的文件夹路径

    param_types : list, 可选
        参数类型列表，如 ['lr', 'gamma', 'clip', 'dim']

    window_size : int, 默认 30
        平滑窗口大小

    line_width : float, 默认 2
        线条宽度

    figure_ratio : float, 默认 1.2
        图形高宽比

    save_path : str, 可选
        保存路径

    dpi : int, 默认 300
        图片分辨率

    y_limits : dict, 可选
        每个子图的y轴范围，格式为 {'lr': (min, max), ...}
    """

    if param_types is None:
        param_types = ['lr', 'gamma', 'clip', 'dim']

    # 加载并分组数据
    grouped_data = load_and_group_files(folder_path, param_types)

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 12 / figure_ratio))
    axes = axes.flatten()

    # 颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, 3))

    # 每个参数值的中文标签
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

    # 绘制每个子图
    for idx, param_type in enumerate(param_types):
        ax = axes[idx]
        param_data = grouped_data[param_type]

        # 如果没有该类型的数据，跳过
        if not param_data:
            continue

        # 按参数值排序
        sorted_values = sorted(param_data.keys(),
                               key=lambda x: float(x) if param_type != 'dim' else int(x))

        # 绘制每条曲线
        for i, param_value in enumerate(sorted_values):
            if i >= len(colors):  # 最多3条线
                break

            data = smooth_data(param_data[param_value], window_size)
            episodes = np.arange(len(data))
            label = param_labels.get(param_type, {}).get(param_value, param_value)

            ax.plot(episodes, data,
                    color=colors[i],
                    linewidth=line_width,
                    label=label,
                    alpha=0.9)

        # 设置坐标轴标签
        ax.set_xlabel('训练轮次', fontsize=10, fontname='SimHei')
        ax.set_ylabel('奖励值', fontsize=10, fontname='SimHei')

        # 设置y轴范围
        if y_limits and param_type in y_limits:
            y_min, y_max = y_limits[param_type]
            ax.set_ylim(y_min, y_max)
        else:
            # 自动计算
            all_values = []
            for param_value in sorted_values:
                if param_value in param_data:
                    data = smooth_data(param_data[param_value], window_size)
                    all_values.extend(data)
            if all_values:
                y_min, y_max = np.min(all_values), np.max(all_values)
                y_padding = (y_max - y_min) * 0.1
                ax.set_ylim(y_min - y_padding, y_max + y_padding)

        # 添加网格
        ax.grid(True, alpha=0.2, linestyle='--')

        # 添加图例
        ax.legend(fontsize=9, loc='upper left')

        # 设置刻度
        ax.minorticks_on()
        ax.tick_params(axis='both', which='minor', length=2)
        ax.tick_params(axis='both', which='major', length=4)

        # 设置刻度标签字体
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('SimHei')
            label.set_fontsize(9)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, axes


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 指定包含npy文件的文件夹路径
    folder_path = "./data"  # 修改为实际路径

    # 自定义y轴范围（可选）
    custom_y_limits = {
        'lr': (7, 30),
        'gamma': (7, 30),
        'clip': (7, 30),
        'dim': (7, 30)
    }

    # 绘制四参数对比网格图
    fig, axes = plot_param_comparison_grid(
        folder_path=folder_path,
        param_types=['lr', 'gamma', 'clip', 'dim'],
        window_size=30,
        line_width=2.5,
        figure_ratio=1.5,
        save_path='param_comparison_grid.png',
        dpi=300,
        y_limits=custom_y_limits  # 可选，不提供则自动计算
    )

    plt.show()