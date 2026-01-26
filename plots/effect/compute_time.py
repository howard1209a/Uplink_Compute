import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


def set_simple_chinese_font():
    """
    简单设置中文字体为黑体
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    print("已设置中文字体为SimHei（黑体）")


def format_draw_histogram(
        labels,
        data,
        x_label_name,
        y_label_name,
        y_bottom,
        bar_labels=None,
        algorithms=None,
        save_path='./info.png',
        font_family='SimHei',
        font_size=16,
        figsize=(10, 6),
        bar_width=0.8,  # 增大柱子宽度
        spacing=0.2,  # 新增：柱子间距参数，默认0.2
        colors=None,
        edge_colors=None,
        hatch_patterns=None,
        legend_fontsize=14,
        legend_loc='upper right',
        tick_rotation=0,
        show_chinese=True
):
    """
    绘制一维柱状图，支持控制柱子间距

    参数:
    spacing: 柱子之间的间距（0-1之间），值越小间距越短
    """
    set_simple_chinese_font()
    rcParams.update({'font.size': font_size})

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlabel(x_label_name, fontsize=font_size + 2)
    ax.set_ylabel(y_label_name, fontsize=font_size + 2)

    # 计算横坐标位置：通过调整间距缩短柱子间距离
    n = len(labels)
    x = np.arange(n) * (1 - spacing)  # 调整间距

    # 设置默认样式
    default_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948']

    colors = colors or [default_colors[i % len(default_colors)] for i in range(n)]
    edge_colors = edge_colors or ['white'] * n
    hatch_patterns = hatch_patterns or [''] * n

    # 确保数据是一维的
    if isinstance(data[0], list):
        data = [row[0] for row in data] if data else []

    # 设置图形样式
    ax.tick_params(which='major', direction='in', length=5, width=1.5, labelsize=font_size)
    ax.tick_params(axis='x', labelrotation=tick_rotation)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(bottom=y_bottom, top=max(data) * 1.15 if data else 1.15)

    # 设置边框线宽
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(1.5)

    # 绘制每个柱子
    for i in range(n):
        ax.bar(x[i], data[i], bar_width,
               edgecolor=edge_colors[i],
               color=colors[i],
               linewidth=0.9,
               hatch=hatch_patterns[i],
               label=bar_labels[0] if bar_labels and i == 0 else '')

    # 添加图例和算法标签
    if bar_labels:
        ax.legend(fontsize=legend_fontsize, loc=legend_loc)

    if algorithms:
        for i in range(n):
            ax.text(x[i], y_bottom - (max(data) * 0.05),
                    algorithms[i % len(algorithms)],
                    ha='center', va='top',
                    fontsize=font_size - 2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# 运行测试
if __name__ == "__main__":
    labels = ['ProCES-360', 'BASELINE', 'EPRO', 'MFQAS', 'AC-KKT', 'TPMOA']
    data = [1155, 4735, 5165, 5622, 3576, 6576]
    colors = ['#B24475', '#864CBC', '#386688', '#845D1C', '#8A543C', '#3D7747']
    hatch_patterns = ['x', 'o', '/', '+', '\\', '//']


    format_draw_histogram(
        labels=labels,
        data=data,
        y_label_name='计算耗时(s)',
        x_label_name='',
        y_bottom=0,
        colors=colors,
        hatch_patterns=hatch_patterns,
        save_path='./compute_time.png',
        bar_width=0.7,
        spacing=0.05,  # 进一步减小间距
        legend_loc='best'
    )