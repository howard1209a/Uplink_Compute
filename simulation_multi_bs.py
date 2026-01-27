import ast
import random
import numpy as np
import pandas as pd
import yaml
import os
import re
from matplotlib import pyplot as plt

from constant import TIME_SLOTS_LENGTH, GAMMA1, GAMMA2, GAMMA3, GAMMA4
from entity import BaseStation, EdgeServer, Video, ResultPerSlot
from strategy.ACKKT_strategy import ACKKT

from strategy.BASELINE_strategy import BASELINE
from strategy.EPRO_strategy import EPRO
from strategy.MFQAS_strategy import MFQAS
from strategy.ProCES360_no_all_strategy import ProCES360_No_All
from strategy.ProCES360_no_drop_strategy import ProCES360_No_Drop
from strategy.ProCES360_no_transmit_strategy import ProCES360_No_Transmit
from strategy.ProCES360_strategy import ProCES360
from strategy.TPMOA_strategy import TPMOA

# 设置全局字体为黑体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_comparison_bar_charts(results, strategy_names, simulation_count):
    """绘制五种指标的柱状图对比"""

    # 提取数据
    metrics = ['transmit_time', 'compute_time', 'consumed_energy', 'video_quality', 'target']
    metric_names = ['传输时间', '计算时间', '能耗', '视频质量', '综合目标值']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 每个指标一种颜色

    # 创建画布和子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # 绘制前5个子图（每个指标一个）
    for i, (metric, metric_name, color) in enumerate(zip(metrics, metric_names, colors)):
        ax = axes[i]

        # 获取该指标的所有策略值
        values = [results[strategy][metric] for strategy in strategy_names]

        # 创建柱状图
        bars = ax.bar(strategy_names, values, color=color, alpha=0.8, edgecolor='black')

        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01 * max(values),
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)

        # 设置标题和标签
        ax.set_title(f'{metric_name}对比', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=9)

        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--')

    # 第6个子图：综合对比（归一化）
    ax = axes[5]

    # 归一化数据（越小越好，所以对于video_quality需要特殊处理）
    normalized_data = {}
    for metric in metrics:
        values = [results[strategy][metric] for strategy in strategy_names]
        if metric == 'video_quality':  # 视频质量越大越好
            max_val = max(values)
            min_val = min(values)
            if max_val != min_val:
                normalized_values = [(v - min_val) / (max_val - min_val) for v in values]
            else:
                normalized_values = [1.0] * len(values)
        else:  # 其他指标越小越好
            max_val = max(values)
            min_val = min(values)
            if max_val != min_val:
                normalized_values = [(max_val - v) / (max_val - min_val) for v in values]
            else:
                normalized_values = [1.0] * len(values)
        normalized_data[metric] = normalized_values

    # 绘制堆叠柱状图或并列柱状图
    x = np.arange(len(strategy_names))
    width = 0.15

    for idx, (metric, metric_name, color) in enumerate(zip(metrics, metric_names, colors)):
        values = normalized_data[metric]
        ax.bar(x + (idx - 2) * width, values, width, label=metric_name, color=color, alpha=0.7)

    ax.set_title('归一化综合对比（所有指标）', fontsize=12, fontweight='bold')
    ax.set_ylabel('归一化值', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(strategy_names, rotation=45, fontsize=9)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')

    # 调整布局
    plt.suptitle(f'第 {simulation_count} 次模拟 - 算法性能对比', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    # 保存图片
    save_path = f'simulation_comparison_{simulation_count}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"对比图已保存到: {save_path}")


def plot_base_station_comparison(base_station_data, simulation_count):
    """绘制六个基站的五个指标平均值对比图"""
    metrics = ['transmit_time', 'compute_time', 'consumed_energy', 'video_quality', 'target_value']
    metric_names = ['传输时间', '计算时间', '能耗', '视频质量', '综合目标值']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # 创建画布和子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # 提取每个基站的平均值
    base_station_labels = [f'基站{i + 1}' for i in range(len(base_station_data))]

    # 绘制前5个子图（每个指标一个）
    for i, (metric, metric_name, color) in enumerate(zip(metrics, metric_names, colors)):
        ax = axes[i]

        # 获取该指标的所有基站平均值
        values = [base_station_data[bs_idx][metric] for bs_idx in range(len(base_station_data))]

        # 创建柱状图
        bars = ax.bar(base_station_labels, values, color=color, alpha=0.8, edgecolor='black')

        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01 * max(values),
                    f'{value:.4f}', ha='center', va='bottom', fontsize=9)

        # 设置标题和标签
        ax.set_title(f'{metric_name}对比', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')

    # 第6个子图：归一化综合对比
    ax = axes[5]

    # 归一化数据
    normalized_data = {}
    for metric in metrics:
        values = [base_station_data[bs_idx][metric] for bs_idx in range(len(base_station_data))]
        if metric == 'video_quality':  # 视频质量越大越好
            max_val = max(values)
            min_val = min(values)
            if max_val != min_val:
                normalized_values = [(v - min_val) / (max_val - min_val) for v in values]
            else:
                normalized_values = [1.0] * len(values)
        else:  # 其他指标越小越好
            max_val = max(values)
            min_val = min(values)
            if max_val != min_val:
                normalized_values = [(max_val - v) / (max_val - min_val) for v in values]
            else:
                normalized_values = [1.0] * len(values)
        normalized_data[metric] = normalized_values

    # 绘制堆叠柱状图
    x = np.arange(len(base_station_labels))
    width = 0.15

    for idx, (metric, metric_name, color) in enumerate(zip(metrics, metric_names, colors)):
        values = normalized_data[metric]
        ax.bar(x + (idx - 2) * width, values, width, label=metric_name, color=color, alpha=0.7)

    ax.set_title('归一化综合对比（所有指标）', fontsize=12, fontweight='bold')
    ax.set_ylabel('归一化值', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(base_station_labels, rotation=45, fontsize=9)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')

    # 调整布局
    plt.suptitle(f'第 {simulation_count} 次模拟 - 基站性能对比', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    # 保存图片
    save_path = f'base_station_comparison_{simulation_count}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"基站对比图已保存到: {save_path}")


def clear_environment_per_slot(base_station_list, edge_server_list):
    # 清空资源
    for base_station in base_station_list:
        base_station.clear()
    for edge_server in edge_server_list:
        edge_server.clear()


def clear_environment_per_simulation(base_station_list, edge_server_list, video_list):
    clear_environment_per_slot(base_station_list, edge_server_list)
    for video in video_list:
        video.clear()


def get_video_user_views(user_stats_dir):
    # 获取所有用户统计文件
    user_files = [f for f in os.listdir(user_stats_dir) if f.endswith('_stats.csv')]
    user_files.sort()

    if not user_files:
        raise ValueError(f"在目录 {user_stats_dir} 中未找到 *_stats.csv 文件")

    # 初始化三维列表
    three_dim_list = []

    # 读取每个用户的数据
    for time_idx in range(TIME_SLOTS_LENGTH):
        time_window_data = []

        for user_file in user_files:
            file_path = os.path.join(user_stats_dir, user_file)
            user_df = pd.read_csv(file_path)

            # 获取当前时间窗口的high_frequency_faces字符串
            faces_str = user_df.loc[time_idx, 'high_frequency_faces']

            # 解析字符串为列表
            try:
                faces_list = [int(x.strip()) for x in faces_str.strip('[]').split(',') if x.strip() != '']
            except:
                faces_list = []

            time_window_data.append(faces_list)

        three_dim_list.append(time_window_data)

    return three_dim_list


def get_video_sizes(folder_path):
    time_slots = TIME_SLOTS_LENGTH
    tiles = 6
    sizes = [[0] * tiles for _ in range(time_slots)]

    pattern = re.compile(r'video_\d+_tile(\d+)_(\d+)\.mp4')

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            tile_num = int(match.group(1)) - 1
            time_num = int(match.group(2))
            if time_num >= time_slots:
                continue

            file_path = os.path.join(folder_path, filename)
            size_bytes = os.path.getsize(file_path)
            sizes[time_num][tile_num] = size_bytes * 8

    return sizes


def get_video_views(file_path):
    df = pd.read_csv(file_path)
    view_list = []
    for i in range(len(df)):
        view_list.append(ast.literal_eval(df.loc[i, 'high_frequency_faces']))
    return view_list


def register_strategy(base_station_list, strategy_name):
    for base_station in base_station_list:
        if strategy_name == "ProCES-360":
            base_station.register_strategy(ProCES360(base_station))
        elif strategy_name == "BASELINE":
            base_station.register_strategy(BASELINE())
        elif strategy_name == "EPRO":
            base_station.register_strategy(EPRO())
        elif strategy_name == "MFQAS":
            base_station.register_strategy(MFQAS())
        elif strategy_name == "AC-KKT":
            base_station.register_strategy(ACKKT())
        elif strategy_name == "TPMOA":
            base_station.register_strategy(TPMOA(base_station))
        elif strategy_name == "ProCES-360-no-transmit":
            base_station.register_strategy(ProCES360_No_Transmit(base_station))
        elif strategy_name == "ProCES-360-no-drop":
            base_station.register_strategy(ProCES360_No_Drop(base_station))
        elif strategy_name == "ProCES-360-no-all":
            base_station.register_strategy(ProCES360_No_All(base_station))
        else:
            raise ValueError("yaml策略名无效")


def get_compute_time_from_edge_servers(edge_server_list, base_station_list):
    edge_server_task_map = {}
    for edge_server in edge_server_list:
        edge_server_task_map[edge_server] = []
    for base_station in base_station_list:
        for task in base_station.origin_task_list:
            if task.offloaded_edge_server is not None:
                edge_server_task_map[task.offloaded_edge_server].append(task)

    compute_time = 0

    for edge_server, task_list in edge_server_task_map.items():
        if len(task_list) == 0:
            continue
        cpu_cycles = []
        gpu_cycles = []
        for task in task_list:
            cpu_cycles.append(task.c)
            gpu_cycles.append(task.g)
        compute_time += ACKKT.optimal_resource_allocation_with_contention(cpu_cycles, gpu_cycles, edge_server.f,
                                                                          edge_server.u, edge_server.IO_conflict_factor)

    return compute_time


# 主程序开始
if __name__ == "__main__":
    # 初始化网络拓扑
    base_station_count = 6
    node_count_ratio = 3
    edge_server_count = base_station_count * node_count_ratio

    base_station_list = []
    edge_server_list = []

    for i in range(base_station_count):
        base_station_list.append(BaseStation(i))
    for i in range(edge_server_count):
        edge_server_list.append(EdgeServer(i))

    base_station_2_base_station_line_ratio = 0.5
    base_station_2_base_station_line_count = int(
        base_station_count * (base_station_count - 1) * base_station_2_base_station_line_ratio / 2)

    while base_station_2_base_station_line_count > 0:
        from_node = random.choice(base_station_list)
        to_node = random.choice(base_station_list)
        if from_node == to_node:
            continue
        if from_node.already_connected(to_node):
            continue
        from_node.connnet_base_station(to_node)
        base_station_2_base_station_line_count -= 1

    base_station_2_edge_server_line_ratio = 0.5
    base_station_2_edge_server_line_count = int(
        base_station_count * edge_server_count * base_station_2_edge_server_line_ratio)

    if base_station_2_edge_server_line_count < len(base_station_list):
        raise ValueError("每个通信基站至少分配一个边缘服务器")
    for base_station in base_station_list:
        base_station.connect_edge_server(random.choice(edge_server_list))
    base_station_2_edge_server_line_count -= len(base_station_list)

    while base_station_2_edge_server_line_count > 0:
        from_node = random.choice(base_station_list)
        to_node = random.choice(edge_server_list)
        if from_node.already_connected(to_node):
            continue
        from_node.connect_edge_server(to_node)
        base_station_2_edge_server_line_count -= 1

    with open("config.yaml", 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    video_name_list = config.get('videos', [])
    video_list = []
    for video_name in video_name_list:
        file_path = "D:\\codes\\Uplink_Compute\\source\\" + video_name + "\\tile\\output"
        view_path = "D:\\codes\\Uplink_Compute\\source\\" + video_name + "\\view.csv"
        user_view_path = "D:\\codes\\Uplink_Compute\\source\\" + video_name + "\\view\\stats"
        video_list.append(
            Video(video_name, get_video_sizes(file_path), get_video_views(view_path),
                  get_video_user_views(user_view_path)))

    # 随机为视频分配基站，保证每个基站至少分配到一个
    for base_station in base_station_list:
        base_station.register_video(video_list[0])
        base_station.register_video(video_list[1])
        base_station.register_video(video_list[2])

    for base_station in base_station_list:
        base_station.init_before_simulation()

    print("---------------------------start simulation---------------------------")

    # 为每个基站分配策略
    base_station_list[0].register_strategy(ProCES360(base_station_list[0]))
    base_station_list[1].register_strategy(BASELINE())
    base_station_list[2].register_strategy(EPRO())
    base_station_list[3].register_strategy(MFQAS())
    base_station_list[4].register_strategy(EPRO())
    base_station_list[5].register_strategy(TPMOA(base_station_list[5]))

    # 初始化数据存储结构
    # 存储每个基站在所有时隙的五个指标
    base_station_metrics = []
    for i in range(len(base_station_list)):
        base_station_metrics.append({
            'transmit_time': [],
            'compute_time': [],
            'consumed_energy': [],
            'video_quality': [],
            'target_value': []
        })

    # 设置仿真次数
    simulation_count = 1

    # 清空资源
    clear_environment_per_simulation(base_station_list, edge_server_list, video_list)

    transmit_time_list = []
    compute_time_list = []
    live_delay_list = []
    consumed_energy_list = []

    # 仿真循环
    for cycle_index in range(40):  # 100个周期
        print(f"正在运行周期 {cycle_index + 1}/40")

        # 清空资源
        clear_environment_per_simulation(base_station_list, edge_server_list, video_list)

        # 依次模拟每个时隙，一个时隙2s
        for time_slot in range(TIME_SLOTS_LENGTH):
            # 清空资源
            clear_environment_per_slot(base_station_list, edge_server_list)

            # 对于每个通信基站，初始化本时隙的任务队列
            for base_station in base_station_list:
                base_station.init_task_queue(time_slot)

            # 执行任务直到所有基站任务完成
            while True:
                for base_station in base_station_list:
                    base_station.interact(time_slot)

                all_clean = True
                for base_station in base_station_list:
                    all_clean = all_clean and base_station.task_clean()

                if all_clean:
                    break

            # 收集每个基站的统计结果
            for bs_idx, base_station in enumerate(base_station_list):
                base_station_result = ResultPerSlot(0, 0, 0, 0)
                for task in base_station.origin_task_list:
                    task.collect_statistics("")

                    transmit_time_list.append(task.transmit_time)
                    compute_time_list.append(task.compute_time)
                    live_delay_list.append(task.transmit_time + task.compute_time)
                    consumed_energy_list.append(task.consumed_energy)

                    base_station_result.transmit_time += task.transmit_time
                    base_station_result.compute_time += task.compute_time
                    base_station_result.consumed_energy += task.consumed_energy
                base_station_result.video_quality = base_station.collect_video_quality(time_slot)

                # 每时隙的后处理
                base_station.post_handle_per_slot(base_station, time_slot, base_station_result)

                # 计算目标值
                target_value = (base_station_result.transmit_time * GAMMA1 +
                                base_station_result.compute_time * GAMMA2 +
                                base_station_result.consumed_energy * 0 +
                                base_station_result.video_quality * GAMMA4 / len(video_list))

                # 存储当前时隙的数据
                base_station_metrics[bs_idx]['transmit_time'].append(base_station_result.transmit_time)
                base_station_metrics[bs_idx]['compute_time'].append(base_station_result.compute_time)
                base_station_metrics[bs_idx]['consumed_energy'].append(base_station_result.consumed_energy)
                base_station_metrics[bs_idx]['video_quality'].append(base_station_result.video_quality)
                base_station_metrics[bs_idx]['target_value'].append(target_value)

    print("\n仿真完成，开始保存数据...")

    # 转换为numpy数组并保存
    for bs_idx in range(len(base_station_list)):
        # 将列表转换为numpy数组
        for metric in ['transmit_time', 'compute_time', 'consumed_energy', 'video_quality', 'target_value']:
            base_station_metrics[bs_idx][metric] = np.array(base_station_metrics[bs_idx][metric])

        # 保存为npy文件
        np.save(f'base_station_{bs_idx + 1}_metrics.npy', base_station_metrics[bs_idx])
        print(f"基站{bs_idx + 1}数据已保存到: base_station_{bs_idx + 1}_metrics.npy")

        # 打印统计信息
        print(f"\n基站{bs_idx + 1}统计:")
        for metric in ['transmit_time', 'compute_time', 'consumed_energy', 'video_quality', 'target_value']:
            data = base_station_metrics[bs_idx][metric]
            print(
                f"  {metric}: 平均值={data.mean():.4f}, 标准差={data.std():.4f}, 最小值={data.min():.4f}, 最大值={data.max():.4f}")

    # 计算每个基站的平均值并绘制对比图
    print("\n计算平均值并绘制对比图...")
    base_station_averages = []
    for bs_idx in range(len(base_station_list)):
        avg_data = {}
        for metric in ['transmit_time', 'compute_time', 'consumed_energy', 'video_quality', 'target_value']:
            avg_data[metric] = base_station_metrics[bs_idx][metric].mean()
        base_station_averages.append(avg_data)

        print(f"\n基站{bs_idx + 1}平均值:")
        print(f"  传输时间: {avg_data['transmit_time']:.4f}")
        print(f"  计算时间: {avg_data['compute_time']:.4f}")
        print(f"  能耗: {avg_data['consumed_energy']:.4f}")
        print(f"  视频质量: {avg_data['video_quality']:.4f}")
        print(f"  综合目标值: {avg_data['target_value']:.4f}")

    # 绘制对比图
    plot_base_station_comparison(base_station_averages, simulation_count)

    # 计算所有基站的总体统计
    print("\n" + "=" * 60)
    print("所有基站总体统计:")
    print("=" * 60)

    for metric, metric_name in zip(
            ['transmit_time', 'compute_time', 'consumed_energy', 'video_quality', 'target_value'],
            ['传输时间', '计算时间', '能耗', '视频质量', '综合目标值']):
        all_data = np.concatenate([base_station_metrics[bs_idx][metric] for bs_idx in range(len(base_station_list))])
        print(f"\n{metric_name}:")
        print(f"  总体平均值: {all_data.mean():.4f}")
        print(f"  总体标准差: {all_data.std():.4f}")
        print(f"  总体最小值: {all_data.min():.4f}")
        print(f"  总体最大值: {all_data.max():.4f}")

    print("\n" + "=" * 60)
    print("仿真完成！")
    print(f"已保存的文件:")
    for bs_idx in range(len(base_station_list)):
        print(f"  base_station_{bs_idx + 1}_metrics.npy")
    print(f"  对比图: base_station_comparison_{simulation_count}.png")
    print("=" * 60)