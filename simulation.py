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


base_station_count = 3
node_count_ratio = 2
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
        Video(video_name, get_video_sizes(file_path), get_video_views(view_path), get_video_user_views(user_view_path)))

if len(video_list) < len(base_station_list):
    raise ValueError("yaml配置视频数不能少于设置的通信基站数")

# 随机为视频分配基站，保证每个基站至少分配到一个
random.shuffle(video_list)
for index in range(len(video_list)):
    if index < len(base_station_list):
        base_station_list[index].register_video(video_list[index])
    else:
        random.choice(base_station_list).register_video(video_list[index])

for base_station in base_station_list:
    base_station.init_before_simulation()

print("---------------------------start simulation---------------------------")

strategy_name_list = config.get("strategies", [])

simulation_count = 0
conditions_met = False

while not conditions_met:
    simulation_count += 1
    print(f"\n======= 第 {simulation_count} 次模拟 =======")

    results = {}  # 存储每个策略的结果

    # 每个策略模拟一次
    for strategy_name in strategy_name_list:
        # 清空资源
        clear_environment_per_simulation(base_station_list, edge_server_list, video_list)

        register_strategy(base_station_list, strategy_name)

        transmit_time = 0
        compute_time = 0
        consumed_energy = 0
        video_quality = 0

        for cycle_index in range(40):  # 100
            # 清空资源
            clear_environment_per_simulation(base_station_list, edge_server_list, video_list)

            # 依次模拟每个时隙，一个时隙2s
            for time_slot in range(TIME_SLOTS_LENGTH):
                # 清空资源
                clear_environment_per_slot(base_station_list, edge_server_list)

                # 对于每个通信基站，初始化本时隙的任务队列
                for base_station in base_station_list:
                    base_station.init_task_queue(time_slot)

                while True:
                    for base_station in base_station_list:
                        base_station.interact(time_slot)

                    all_clean = True
                    for base_station in base_station_list:
                        all_clean = all_clean and base_station.task_clean()

                    if all_clean:
                        break

                for base_station in base_station_list:
                    base_station_result = ResultPerSlot(0, 0, 0, 0)
                    for task in base_station.origin_task_list:
                        task.collect_statistics(strategy_name)
                        base_station_result.transmit_time += task.transmit_time
                        base_station_result.compute_time += task.compute_time
                        base_station_result.consumed_energy += task.consumed_energy
                    base_station_result.video_quality = base_station.collect_video_quality(time_slot)

                    # 每时隙的后处理
                    base_station.post_handle_per_slot(base_station, time_slot, base_station_result)

                    transmit_time += base_station_result.transmit_time
                    compute_time += base_station_result.compute_time
                    consumed_energy += base_station_result.consumed_energy
                    video_quality += base_station_result.video_quality

                # 特殊算法，从边缘服务器维度统计本时隙的计算耗时
                if strategy_name == "AC-KKT":
                    compute_time += get_compute_time_from_edge_servers(edge_server_list, base_station_list)

        # 计算目标值
        target_value = transmit_time * GAMMA1 + compute_time * GAMMA2 + consumed_energy * GAMMA3 + video_quality * GAMMA4

        # 存储结果
        results[strategy_name] = {
            'transmit_time': transmit_time,
            'compute_time': compute_time,
            'consumed_energy': consumed_energy,
            'video_quality': video_quality,
            'target': target_value
        }

        print("\n ------" + str(strategy_name) + "------\n")
        print("transmit_time: " + str(transmit_time))
        print("compute_time: " + str(compute_time))
        print("consumed_energy: " + str(consumed_energy))
        print("video_quality: " + str(video_quality))
        print("target: " + str(target_value))

        # 存储结果
        results[strategy_name] = {
            'transmit_time': transmit_time,
            'compute_time': compute_time,
            'consumed_energy': consumed_energy,
            'video_quality': video_quality,
            'target': target_value
        }

    # 绘制柱状图对比
    plot_comparison_bar_charts(results, strategy_name_list, simulation_count)
    conditions_met = True
