import matplotlib.pyplot as plt
import numpy as np
import matplotlib

plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]


def format_draw_boxplot(
        data,
        labels,
        x_label_name,
        y_label_name,
        y_lim,
        save_path,
        colors,
        hatch_patterns,
        showfliers,
        mean_line,
        font_size=10.5,
        figsize=(4, 2.5),
        dpi=300
):
    plt.rcParams.update({'font.size': font_size})
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlabel(x_label_name, fontsize=font_size)
    ax.set_ylabel(y_label_name, fontsize=font_size)

    medianprops = {'color': 'black'}
    flierprops = {'markersize': 2}

    bp = ax.boxplot(data,
                    patch_artist=True,
                    showfliers=showfliers,
                    showmeans=False,
                    medianprops=medianprops,
                    flierprops=flierprops,
                    widths=0.6)

    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i])
        box.set_hatch(hatch_patterns[i])
        box.set_edgecolor('black')

    if mean_line:
        for i, dataset in enumerate(data):
            mean_val = np.mean(dataset)
            x_pos = i + 1
            ax.hlines(mean_val, x_pos - 0.3, x_pos + 0.3, colors='red', linestyles='--', label='均值' if i == 0 else "")
            ax.scatter([x_pos - 0.3, x_pos + 0.3], [mean_val, mean_val], color='red', marker='|', s=50, zorder=5)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=font_size)
    y_min, y_max = y_lim
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='white')
    plt.show()


if __name__ == "__main__":
    ProCES360_live_delay_list = np.load('ProCES-360_live_delay_list.npy').tolist()
    BASELINE_live_delay_list = np.load('BASELINE_live_delay_list.npy').tolist()
    EPRO_live_delay_list = np.load('EPRO_live_delay_list.npy').tolist()
    MFQAS_live_delay_list = np.load('MFQAS_live_delay_list.npy').tolist()
    ACKKT_live_delay_list = np.load('AC-KKT_live_delay_list.npy').tolist()
    TPMOA_live_delay_list = np.load('TPMOA_live_delay_list.npy').tolist()

    ProCES360_live_delay_list = [num / 18.0 for num in ProCES360_live_delay_list]
    BASELINE_live_delay_list = [num / 18.0 for num in BASELINE_live_delay_list]
    EPRO_live_delay_list = [num / 18.0 for num in EPRO_live_delay_list]
    MFQAS_live_delay_list = [num / 18.0 for num in MFQAS_live_delay_list]
    ACKKT_live_delay_list = [num / 18.0 for num in ACKKT_live_delay_list]
    TPMOA_live_delay_list = [num / 2.5 / 18.0 for num in TPMOA_live_delay_list]

    data = [
        ProCES360_live_delay_list,
        BASELINE_live_delay_list,
        EPRO_live_delay_list,
        MFQAS_live_delay_list,
        ACKKT_live_delay_list,
        TPMOA_live_delay_list
    ]
    labels = ['PCS-360', 'BASE', 'EPRO', 'MFQAS', 'KKT', 'TPMOA']
    colors = ['#B24475', '#864CBC', '#386688', '#845D1C', '#8A543C', '#3D7747']
    hatch_patterns = ['x', 'o', '/', '+', '\\', '//']

    format_draw_boxplot(
        data=data,
        labels=labels,
        x_label_name='',
        y_label_name='直播延迟(s)',
        y_lim=(-0.05, 0.45),
        save_path='live_delay_strategy_fair.png',
        colors=colors,
        hatch_patterns=hatch_patterns,
        showfliers=False,
        mean_line=True,
    )
