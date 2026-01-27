import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
import warnings

warnings.filterwarnings('ignore')


def set_simple_chinese_font():
    """
    简单设置中文字体为黑体
    直接使用SimHei，避免复杂的字体检测
    """
    # 方法1: 直接设置SimHei为默认字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    print("已设置中文字体为SimHei（黑体）")


def format_draw_boxplot(data, labels, x_label_name, y_label_name,
                        colors=None, hatch_patterns=None, save_path='./boxplot.png',
                        figsize=(10, 6), show_means=False, mean_line=False,
                        y_lim=None):
    """
    绘制一层分组箱型图（学术论文风格）
    加入红色虚线均值线，柱子有图案，与柱状图风格一致

    参数:
    ----------
    data : list of lists
        数据，每个子列表代表一个箱型图的数据
    labels : list of str
        每个箱型图的标签
    x_label_name : str
        x轴标签
    y_label_name : str
        y轴标签
    colors : list of str, optional
        箱体颜色列表
    hatch_patterns : list of str, optional
        箱体图案列表
    save_path : str, optional
        保存路径
    figsize : tuple, optional
        图形大小
    show_means : bool, optional
        是否显示均值标记
    mean_line : bool, optional
        是否显示均值线
    y_lim : tuple, optional
        纵轴范围 (y_min, y_max)，如果为None则自动计算
    """
    # 设置中文字体
    set_simple_chinese_font()

    plt.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # 设置坐标轴标签 - 使用常规方法，不指定fontproperties
    ax.set_xlabel(x_label_name, fontsize=18)
    ax.set_ylabel(y_label_name, fontsize=18)

    # 设置刻度样式
    ax.tick_params(which='major', direction='in', length=5, width=1.5, labelsize=16)
    ax.tick_params(axis='x', labelrotation=0)

    # 设置边框线宽
    linewidth = 1.5
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(linewidth)

    # 颜色配置（与柱状图相同的配色方案）
    if colors is None:
        colors = [
            '#B24475',  # 深粉色 - 5g（与柱状图一致）
            '#864CBC',  # 紫色   - wifi
            '#386688',  # 深蓝色 - VAAC-E
            '#845D1C',  # 棕色   - PW
            '#8A543C',  # 土褐色 - BCD
            '#3D7747'  # 深绿色 - Vega
        ]

    # 图案配置（与柱状图一致的图案）
    if hatch_patterns is None:
        hatch_patterns = [
            'x',  # 5g
            'o',  # wifi
            '/',  # VAAC-E
            '+',  # PW
            '\\',  # BCD
            '//'  # Vega
        ]

    # 配置箱型图属性
    boxprops = {
        'linewidth': 1.5,
        'edgecolor': 'black'
    }

    # 中位数线配置
    medianprops = {
        'color': 'black',
        'linewidth': 2.0
    }

    # 均值标记配置
    meanprops = {
        'marker': 'D',  # 菱形标记
        'markerfacecolor': 'red',
        'markeredgecolor': 'red',
        'markersize': 8,
        'linestyle': 'none'  # 不显示连接线
    } if show_means else None

    # 绘制箱型图
    bp = ax.boxplot(data,
                    patch_artist=True,
                    showfliers=False,  # 不显示异常值点
                    showmeans=show_means,  # 是否显示均值标记
                    medianprops=medianprops,
                    meanprops=meanprops,
                    boxprops=boxprops,
                    whiskerprops={'linewidth': 1.5, 'color': 'black'},
                    capprops={'linewidth': 1.5, 'color': 'black'},
                    widths=0.6  # 箱子宽度
                    )

    # 设置箱子颜色和图案（与柱状图一致）
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i % len(colors)])
        box.set_alpha(0.8)  # 透明度
        # 添加图案
        box.set_hatch(hatch_patterns[i % len(hatch_patterns)])
        box.set_edgecolor('black')

    # 计算并绘制红色虚线均值线
    if mean_line:
        for i, dataset in enumerate(data):
            mean_val = np.mean(dataset)
            x_pos = i + 1  # 箱型图的x位置从1开始

            # 绘制红色虚线均值线
            ax.hlines(mean_val,
                      x_pos - 0.3,  # 线的起点（箱子宽度的一半）
                      x_pos + 0.3,  # 线的终点
                      colors='red',
                      linestyles='--',
                      linewidth=2.0,
                      label='均值' if i == 0 else "")  # 只在第一个添加图例

            # 在均值线两端添加小标记
            ax.scatter([x_pos - 0.3, x_pos + 0.3],
                       [mean_val, mean_val],
                       color='red',
                       marker='|',
                       s=50,  # 标记大小
                       zorder=5)  # 确保标记在最上层

    # 设置横轴标签
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=16)

    # 设置纵轴范围
    if y_lim is not None:
        # 使用用户指定的纵轴范围
        y_min, y_max = y_lim
        ax.set_ylim(y_min, y_max)
        print(f"已设置纵轴范围: {y_min} 到 {y_max}")
    else:
        # 自动设置纵轴范围（原有逻辑）
        all_data = []
        for d in data:
            all_data.extend(d)

        if all_data:
            # 计算数据的基本统计量
            q1 = np.percentile(all_data, 25)
            q3 = np.percentile(all_data, 75)
            iqr = q3 - q1

            # 设置合理的纵轴范围
            y_min = q1 - 1.5 * iqr
            y_max = q3 + 1.5 * iqr
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            print(f"自动计算纵轴范围: {y_min - 0.1 * y_range:.2f} 到 {y_max + 0.1 * y_range:.2f}")

    # 添加网格线（可选，提高可读性）
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # 创建自定义图例
    from matplotlib.patches import Patch
    legend_patches = []

    # 如果有图例，添加到图表中
    if legend_patches:
        ax.legend(handles=legend_patches, fontsize=12, loc='upper right')

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='white')
    plt.show()


# 简洁版本（保持图案，无均值标记）- 同样添加y_lim参数
def format_draw_boxplot_simple(data, labels, x_label_name, y_label_name,
                               save_path='./boxplot_simple.png',
                               y_lim=None):
    """
    简洁版箱型图，保持图案样式

    参数:
    ----------
    y_lim : tuple, optional
        纵轴范围 (y_min, y_max)，如果为None则自动计算
    """
    # 设置中文字体
    set_simple_chinese_font()

    plt.rcParams.update({'font.size': 14})

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # 设置坐标轴标签
    ax.set_xlabel(x_label_name, fontsize=16)
    ax.set_ylabel(y_label_name, fontsize=16)

    # 设置刻度样式
    ax.tick_params(which='major', direction='in', length=4, width=1.2, labelsize=14)

    # 设置边框线宽
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(1.2)

    # 颜色和图案配置（与柱状图一致）
    colors = ['#B24475', '#864CBC', '#386688', '#845D1C', '#8A543C', '#3D7747']
    hatch_patterns = ['x', 'o', '/', '+', '\\', '//']

    # 绘制箱型图
    bp = ax.boxplot(data,
                    patch_artist=True,
                    showfliers=False,  # 不显示异常值
                    showmeans=False,  # 不显示均值标记
                    medianprops={'color': 'black', 'linewidth': 1.5},
                    boxprops={'linewidth': 1.2, 'edgecolor': 'black'},
                    whiskerprops={'linewidth': 1.2, 'color': 'black'},
                    capprops={'linewidth': 1.2, 'color': 'black'},
                    widths=0.5)

    # 设置箱子颜色和图案
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i % len(colors)])
        box.set_alpha(0.7)
        box.set_hatch(hatch_patterns[i % len(hatch_patterns)])

    # 设置横轴标签
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=14)

    # 设置纵轴范围
    if y_lim is not None:
        y_min, y_max = y_lim
        ax.set_ylim(y_min, y_max)
        print(f"已设置纵轴范围: {y_min} 到 {y_max}")

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='white')
    plt.show()


# 高级版本：可以同时设置多个自定义参数
def format_draw_boxplot_advanced(data, labels, x_label_name, y_label_name,
                                 save_path='./boxplot_advanced.png',
                                 figsize=(10, 6),
                                 show_means=False,
                                 mean_line=False,
                                 y_lim=None,
                                 x_lim=None,
                                 colors=None,
                                 hatch_patterns=None,
                                 grid_alpha=0.3,
                                 legend_location='upper right',
                                 dpi=300,
                                 show_outliers=False):
    """
    高级版箱型图，提供更多自定义选项
    """
    # 设置中文字体
    set_simple_chinese_font()

    plt.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # 设置坐标轴标签
    ax.set_xlabel(x_label_name, fontsize=18)
    ax.set_ylabel(y_label_name, fontsize=18)

    # 设置刻度样式
    ax.tick_params(which='major', direction='in', length=5, width=1.5, labelsize=16)
    ax.tick_params(axis='x', labelrotation=0)

    # 设置边框线宽
    linewidth = 1.5
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(linewidth)

    # 颜色配置
    if colors is None:
        colors = [
            '#B24475',  # 深粉色
            '#864CBC',  # 紫色
            '#386688',  # 深蓝色
            '#845D1C',  # 棕色
            '#8A543C',  # 土褐色
            '#3D7747'  # 深绿色
        ]

    # 图案配置
    if hatch_patterns is None:
        hatch_patterns = ['x', 'o', '/', '+', '\\', '//']

    # 绘制箱型图
    bp = ax.boxplot(data,
                    patch_artist=True,
                    showfliers=show_outliers,  # 是否显示异常值
                    showmeans=show_means,
                    medianprops={'color': 'black', 'linewidth': 2.0},
                    meanprops={'marker': 'D', 'markerfacecolor': 'red',
                               'markeredgecolor': 'red', 'markersize': 8} if show_means else None,
                    boxprops={'linewidth': 1.5, 'edgecolor': 'black'},
                    whiskerprops={'linewidth': 1.5, 'color': 'black'},
                    capprops={'linewidth': 1.5, 'color': 'black'},
                    widths=0.6)

    # 设置箱子颜色和图案
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i % len(colors)])
        box.set_alpha(0.8)
        box.set_hatch(hatch_patterns[i % len(hatch_patterns)])

    # 设置横轴标签
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=16)

    # 设置纵轴范围
    if y_lim is not None:
        y_min, y_max = y_lim
        ax.set_ylim(y_min, y_max)

    # 设置横轴范围（如果提供）
    if x_lim is not None:
        x_min, x_max = x_lim
        ax.set_xlim(x_min, x_max)

    # 添加网格线
    ax.grid(True, axis='y', alpha=grid_alpha, linestyle='--', linewidth=0.5)

    # 保存图表
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='white')
    plt.show()


if __name__ == "__main__":
    ProCES360_objective_function_list = np.load('ProCES-360_objective_function_list.npy').tolist()
    BASELINE_objective_function_list = np.load('BASELINE_objective_function_list.npy').tolist()
    EPRO_objective_function_list = np.load('EPRO_objective_function_list.npy').tolist()
    MFQAS_objective_function_list = np.load('MFQAS_objective_function_list.npy').tolist()
    ACKKT_objective_function_list = np.load('AC-KKT_objective_function_list.npy').tolist()
    TPMOA_objective_function_list = np.load('TPMOA_objective_function_list.npy').tolist()

    # ProCES360_objective_function_list = [num / 18.0 for num in ProCES360_objective_function_list]
    # BASELINE_objective_function_list = [num / 18.0 for num in BASELINE_objective_function_list]
    # EPRO_objective_function_list = [num / 18.0 for num in EPRO_objective_function_list]
    # MFQAS_objective_function_list = [num / 18.0 for num in MFQAS_objective_function_list]
    # ACKKT_objective_function_list = [num / 18.0 for num in ACKKT_objective_function_list]
    TPMOA_objective_function_list = [num / 3.0 for num in TPMOA_objective_function_list]

    data = [
        ProCES360_objective_function_list,
        BASELINE_objective_function_list,
        EPRO_objective_function_list,
        MFQAS_objective_function_list,
        ACKKT_objective_function_list,
        TPMOA_objective_function_list
    ]

    labels = ['ProCES-360', 'BASELINE', 'EPRO', 'MFQAS', 'AC-KKT', 'TPMOA']

    format_draw_boxplot(
        data=data,
        labels=labels,
        x_label_name='',
        y_label_name='目标函数',
        show_means=False,
        mean_line=True,
        save_path='./objective_function_strategy_fair.png',
        y_lim=(0, 0.7)
    )
