import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]


def plot_bandwidth_cdf(
        labels,
        data,
        x_label_name,
        y_label_name,
        x_lim,
        y_lim,
        save_path,
        colors,
        legend_loc,
        markers,
        num_markers=25,
        line_width=1,
        font_size=10.5,
        figsize=(4, 2.5),
        legend_fontsize=8.5,
        marker_size=5,
        dpi=300
):
    n_lines = len(data)

    all_values = np.concatenate(data)
    x_min = np.min(all_values)
    x_max = np.max(all_values)

    marker_x_positions = np.linspace(x_min, x_max, num_markers)

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(n_lines):
        values = np.array(data[i])
        values_sorted = np.sort(values)

        marker_cdf_values = np.zeros(num_markers)
        for j, x_pos in enumerate(marker_x_positions):
            count_leq = np.sum(values_sorted <= x_pos)
            marker_cdf_values[j] = count_leq / len(values_sorted)

        if i == 0:
            marker_cdf_values[0] = 0.0
        if i == n_lines - 1:
            marker_cdf_values[-1] = 1.0

        ax.plot(marker_x_positions, marker_cdf_values,
                linestyle='-',
                color=colors[i],
                linewidth=line_width,
                marker=markers[i],
                markersize=marker_size,
                markeredgecolor='white',
                markeredgewidth=0.3,
                label=labels[i])

    ax.set_xlabel(x_label_name, fontsize=font_size)
    ax.set_ylabel(y_label_name, fontsize=font_size)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.grid(True, alpha=0.2, linestyle='--')

    ax.legend(fontsize=legend_fontsize, loc=legend_loc, frameon=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    ProCES360_live_delay_list = np.load('ProCES-360_consumed_energy_list.npy').tolist()
    BASELINE_live_delay_list = np.load('BASELINE_consumed_energy_list.npy').tolist()
    EPRO_live_delay_list = np.load('EPRO_consumed_energy_list.npy').tolist()
    MFQAS_live_delay_list = np.load('MFQAS_consumed_energy_list.npy').tolist()
    ACKKT_live_delay_list = np.load('AC-KKT_consumed_energy_list.npy').tolist()
    TPMOA_live_delay_list = np.load('TPMOA_consumed_energy_list.npy').tolist()

    data = [ProCES360_live_delay_list, BASELINE_live_delay_list, EPRO_live_delay_list, MFQAS_live_delay_list,
            ACKKT_live_delay_list, TPMOA_live_delay_list]

    colors = np.vstack([plt.cm.tab10(np.linspace(0, 1, 3)), plt.cm.tab10([6, 7, 8])])

    markers = ['o', 'v', 's', '^', 'D', 'p']

    labels = ['PCS-360', 'BASE', 'EPRO', 'MFQAS', 'KKT', 'TPMOA']

    plot_bandwidth_cdf(
        labels=labels,
        data=data,
        x_label_name='能耗(J)',
        y_label_name='CDF',
        x_lim=(0, 105),
        y_lim=(0, 1.05),
        save_path='energy_consume_task_fair.png',
        colors=colors,
        legend_loc='lower right',
        markers=markers
    )
