import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]


def format_draw_histogram(
        labels,
        data,
        x_label_name,
        y_label_name,
        ylim,
        save_path,
        colors,
        hatch_patterns,
        font_size=10.5,
        figsize=(4, 2.5),
        bar_width=0.75,
        dpi=300,
        hatch_density=1,
        hatch_linewidth=1
):
    plt.rcParams['hatch.linewidth'] = hatch_linewidth
    rcParams.update({'font.size': font_size})

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlabel(x_label_name, fontsize=font_size)
    ax.set_ylabel(y_label_name, fontsize=font_size)

    x = np.arange(len(labels))
    ax.tick_params(labelsize=font_size)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(ylim)

    for i in range(len(labels)):
        ax.bar(x[i], data[i], bar_width, label=labels[i], edgecolor='white', color=colors[i],
               hatch=hatch_patterns[i] * hatch_density)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    labels = ['PCS-360', 'BASE', 'EPRO', 'MFQAS', 'KKT', 'TPMOA']
    data = [49126.317286, 42420.038426, 41240.153140, 54099.092742, 43618.752676, 39742.256752]
    colors = ['#B24475', '#864CBC', '#386688', '#845D1C', '#8A543C', '#3D7747']
    hatch_patterns = ['x', 'o', '/', '+', '\\', '//']

    format_draw_histogram(
        labels=labels,
        data=data,
        y_label_name='观看质量(Mbps)',
        x_label_name='',
        ylim=(0, 62000),
        save_path='./video_quality.png',
        colors=colors,
        hatch_patterns=hatch_patterns,
    )