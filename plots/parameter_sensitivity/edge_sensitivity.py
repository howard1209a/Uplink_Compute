import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]


def format_draw_histogram(
        labels,
        data,
        x_label_name,
        y_label_name,
        y_lim,
        bar_labels,
        save_path,
        colors=None,
        hatch_patterns=None,
        legend_loc=None,
        font_size=10.5,
        figsize=(4, 2.5),
        bar_width=0.22,
        group_spacing=1,
        legend_fontsize=8.5,
        dpi=300,
        hatch_density=3,
        hatch_linewidth=0.5
):
    plt.rcParams['hatch.linewidth'] = hatch_linewidth

    plt.rcParams.update({'font.size': font_size})
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_xlabel(x_label_name, fontsize=font_size)
    ax.set_ylabel(y_label_name, fontsize=font_size)

    x = np.arange(len(labels))
    n_bars = len(data[0]) if data else 0
    offsets = [((i - n_bars / 2 + 0.5) * group_spacing * bar_width) for i in range(n_bars)]

    ax.tick_params(labelsize=font_size)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(y_lim)

    for idx in range(n_bars):
        bar_data = [row[idx] for row in data]
        hatch_pattern = hatch_patterns[idx] if hatch_patterns else None
        if hatch_pattern:
            hatch_pattern = hatch_pattern * hatch_density

        ax.bar(x + offsets[idx], bar_data, bar_width, label=bar_labels[idx],
               edgecolor='white', color=colors[idx], hatch=hatch_pattern)

    ax.legend(fontsize=legend_fontsize, loc=legend_loc, ncol=2, frameon=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    labels = ['0', '0.3', '0.6', '1.0']
    data = [
        [0.405, 0.467, 0.583, 0.537],
        [0.381, 0.361, 0.511, 0.444],
        [0.433, 0.446, 0.491, 0.529],
        [0.432, 0.476, 0.469, 0.455]
    ]
    bar_labels = ['卸载边比例=0.2', '卸载边比例=0.5', '卸载边比例=0.8', '卸载边比例=1']
    colors = ['#B24475', '#864CBC', '#386688', '#845D1C']

    format_draw_histogram(
        labels=labels,
        data=data,
        x_label_name='转发边比例',
        y_label_name='PCS-360/BASE',
        y_lim=(0, 0.8),
        bar_labels=bar_labels,
        save_path='./edge_sensitivity.png',
        colors=colors,
        hatch_patterns=['x', 'o', '/', '+'],
        legend_loc='upper center'
    )
