import matplotlib.pyplot as plt
import numpy as np
import os

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def smooth_data(data, window_size=30):
    """平滑数据"""
    if len(data) < window_size:
        return data
    smoothed = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    padded = np.concatenate([data[:window_size - 1], smoothed])
    return padded


def load_npy_file(file_path):
    """
    加载npy文件

    参数:
    file_path: npy文件完整路径
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    data = np.load(file_path)
    if data.ndim > 1:
        data = data.flatten()

    return data


def plot_loss_curve(file_path, title="训练损失曲线",
                    xlabel="训练步数", ylabel="",
                    window_size=50, line_width=2, color='#D72638',
                    figsize=(10, 6), save_path=None, dpi=300):
    """
    绘制单条损失曲线图

    参数:
    file_path: npy文件完整路径
    title: 图表标题
    xlabel: x轴标签
    ylabel: y轴标签
    window_size: 平滑窗口大小
    line_width: 线条宽度
    color: 线条颜色
    figsize: 图形大小
    save_path: 保存路径
    dpi: 图片分辨率
    """
    # 加载数据
    data = load_npy_file(file_path)

    # 平滑数据
    smoothed_data = smooth_data(data, window_size)

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制曲线
    steps = np.arange(len(smoothed_data))
    ax.plot(steps, smoothed_data,
            linewidth=line_width,
            color=color,
            alpha=0.9,
            label='')

    # 设置坐标轴
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')

    # 设置刻度
    ax.minorticks_on()
    ax.tick_params(axis='both', which='minor', length=2)
    ax.tick_params(axis='both', which='major', length=4)

    # 添加图例
    ax.legend(fontsize=11)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    # 显示统计信息
    print(f"文件: {os.path.basename(file_path)}")
    print(f"数据长度: {len(data)}")
    print(f"平均损失: {np.mean(data):.6f}")
    print(f"最小损失: {np.min(data):.6f}")
    print(f"最大损失: {np.max(data):.6f}")

    return fig, ax, data


# ========== 使用示例 ==========
if __name__ == "__main__":
    actor_loss_file = "./actor_loss.npy"

    # 绘制损失曲线
    fig, ax, data = plot_loss_curve(
        file_path=actor_loss_file,
        title="",
        xlabel="训练轮次",
        ylabel="Actor网络损失函数",
        window_size=10,
        line_width=2.5,
        color=plt.cm.tab10(np.linspace(0, 1, 3))[0],
        figsize=(10, 6),
        save_path='actor_loss.png',
        dpi=300
    )

    plt.show()

    critic_loss_file = "./critic_loss.npy"

    # 绘制损失曲线
    fig, ax, data = plot_loss_curve(
        file_path=critic_loss_file,
        title="",
        xlabel="训练轮次",
        ylabel="Critic网络损失函数",
        window_size=10,
        line_width=2.5,
        color=plt.cm.tab10(np.linspace(0, 1, 3))[1],
        figsize=(10, 6),
        save_path='critic_loss.png',  # 可选
        dpi=300
    )

    plt.show()