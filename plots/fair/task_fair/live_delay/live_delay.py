import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_bandwidth_cdf(data_list, colors=None, markers=None, labels=None,
                       title='', show_grid=True, save_path=None, **kwargs):
    """
    绘制带宽数据的CDF曲线图

    参数:
    ----------
    data_list : list of lists
        带宽数据列表，每个子列表代表一条线的数据

    colors : list
        颜色列表（长度应与数据列表相同）

    markers : list
        标记形状列表

    labels : list
        图例标签列表

    title : str
        图表标题

    show_grid : bool
        是否显示网格

    save_path : str
        保存路径

    **kwargs :
        其他图形参数，如figure_size, line_width, alpha等
    """

    # 获取线条数量
    n_lines = len(data_list)

    # 设置默认值
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n_lines))

    if markers is None:
        markers = ['o', 'v', 's', '^', 'D', 'p', '*', 'X']

    if labels is None:
        labels = [f'线路{i + 1}' for i in range(n_lines)]

    # 创建图形
    fig_size = kwargs.get('figure_size', (12, 8))
    fig, ax = plt.subplots(figsize=fig_size)

    # 绘制每条线的CDF
    for i in range(n_lines):
        values = np.array(data_list[i])
        values_sorted = np.sort(values)
        cdf = np.arange(1, len(values_sorted) + 1) / len(values_sorted)

        # 采样点
        step = max(1, len(values_sorted) // 15)

        # 绘制线条
        ax.plot(values_sorted, cdf,
                linestyle='-',
                color=colors[i] if i < len(colors) else None,
                linewidth=kwargs.get('line_width', 2),
                alpha=kwargs.get('alpha', 0.9),
                marker=markers[i % len(markers)],
                markersize=kwargs.get('marker_size', 8),
                markevery=step,
                label=labels[i] if i < len(labels) else f'Line {i + 1}')

    # 设置图形属性
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20, fontname='SimHei')
    ax.set_xlabel('直播延迟(s)', fontsize=12, fontname='SimHei')
    ax.set_ylabel('CDF', fontsize=12, fontname='SimHei')

    # 设置坐标轴范围
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)

    # 添加网格
    if show_grid:
        ax.grid(True, alpha=0.2, linestyle='--')

    # 添加图例
    ax.legend(fontsize=11, loc='lower right',
              borderaxespad=0.5, prop={'family': 'SimHei', 'size': 10})

    # 调整布局
    plt.tight_layout()

    # 保存图片
    if save_path:
        dpi = kwargs.get('dpi', 300)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")

    return fig, ax


# ========== 使用示例（六条线） ==========
if __name__ == "__main__":
    ProCES360_live_delay_list = np.load('ProCES-360_live_delay_list.npy').tolist()
    BASELINE_live_delay_list = np.load('BASELINE_live_delay_list.npy').tolist()
    EPRO_live_delay_list = np.load('EPRO_live_delay_list.npy').tolist()
    MFQAS_live_delay_list = np.load('MFQAS_live_delay_list.npy').tolist()
    ACKKT_live_delay_list = np.load('AC-KKT_live_delay_list.npy').tolist()
    TPMOA_live_delay_list = np.load('TPMOA_live_delay_list.npy').tolist()


    # 组合所有数据
    all_data = [ProCES360_live_delay_list, BASELINE_live_delay_list, EPRO_live_delay_list,MFQAS_live_delay_list, ACKKT_live_delay_list, TPMOA_live_delay_list]

    # 自定义颜色（使用tab20色彩映射生成6个颜色）
    custom_colors = plt.cm.tab20(np.linspace(0, 1, 6))

    # 自定义标记
    custom_markers = ['o', 'v', 's', '^', 'D', 'p']

    # 自定义标签
    custom_labels = labels = ['ProCES-360', 'BASELINE', 'EPRO', 'MFQAS', 'AC-KKT', 'TPMOA']

    # 绘制图形
    fig, ax = plot_bandwidth_cdf(
        data_list=all_data,
        colors=custom_colors,
        markers=custom_markers,
        labels=custom_labels,
        title='',
        show_grid=True,
        save_path='live_delay_fair.png',
        figure_size=(12, 8),
        line_width=2.5,
        alpha=0.85,
        marker_size=9,
        dpi=300
    )

    plt.show()