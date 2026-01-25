import ast
import random

import pandas as pd
import yaml
import os
import re

from constant import TIME_SLOTS_LENGTH, GAMMA1, GAMMA2, GAMMA3, GAMMA4
from entity import BaseStation, EdgeServer, Video, ResultPerSlot
from strategy import ProCES360, Random360, Drop

import matplotlib.pyplot as plt
import numpy as np


# TODO: 3. 训练模型，看reward曲线 4. 做好结果记录与显示 5. 实现对比算法 6. 按照实验图进行测试


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
        elif strategy_name == "Random360":
            base_station.register_strategy(Random360())
        elif strategy_name == "Drop":
            base_station.register_strategy(Drop())
        else:
            raise ValueError("yaml策略名无效")


base_station_count = 1
node_count_ratio = 2
edge_server_count = base_station_count * node_count_ratio

base_station_list = []
edge_server_list = []

for i in range(base_station_count):
    base_station_list.append(BaseStation(i))
for i in range(edge_server_count):
    edge_server_list.append(EdgeServer(i))

base_station_2_base_station_line_ratio = 0.3
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

# 每个策略模拟一次
for strategy_name in strategy_name_list:
    # 清空资源
    clear_environment_per_simulation(base_station_list, edge_server_list, video_list)

    register_strategy(base_station_list, strategy_name)

    transmit_time = 0
    compute_time = 0
    consumed_energy = 0
    video_quality = 0

    for cycle_index in range(100):
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
                    task.collect_statistics()
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

    print("\n ------" + str(strategy_name) + "------\n")
    print("transmit_time: " + str(transmit_time))
    print("compute_time: " + str(compute_time))
    print("consumed_energy: " + str(consumed_energy))
    print("video_quality: " + str(video_quality))
    print("target: " + str(
        transmit_time * GAMMA1 + compute_time * GAMMA2 + consumed_energy * GAMMA3 + video_quality * GAMMA4))

    if strategy_name == "ProCES-360":
        # 保存到文件
        np.save('reward_list.npy', base_station_list[0].strategy.reward_list)
        np.save('actor_loss_list.npy', base_station_list[0].strategy.actor_loss_list)
        np.save('critic_loss_list.npy', base_station_list[0].strategy.critic_loss_list)

        # 创建包含多个子图的图形
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 第一个基站
        # Reward曲线
        axes[0, 0].plot(base_station_list[0].strategy.reward_list, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Episode', fontsize=12)
        axes[0, 0].set_ylabel('Reward', fontsize=12)
        axes[0, 0].set_title('Reward Trend - Base Station 0', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)

        # Actor Loss曲线
        if base_station_list[0].strategy.actor_loss_list:
            axes[1, 0].plot(base_station_list[0].strategy.actor_loss_list, 'r-', linewidth=2)
            axes[1, 0].set_xlabel('Training Step', fontsize=12)
            axes[1, 0].set_ylabel('Actor Loss', fontsize=12)
            axes[1, 0].set_title('Actor Loss Trend - Base Station 0', fontsize=14)
            axes[1, 0].grid(True, alpha=0.3)

        # 第二个基站
        if len(base_station_list) > 1:
            axes[0, 1].plot(base_station_list[1].strategy.reward_list, 'g-', linewidth=2)
            axes[0, 1].set_xlabel('Episode', fontsize=12)
            axes[0, 1].set_ylabel('Reward', fontsize=12)
            axes[0, 1].set_title('Reward Trend - Base Station 1', fontsize=14)
            axes[0, 1].grid(True, alpha=0.3)

            if base_station_list[1].strategy.actor_loss_list:
                axes[1, 1].plot(base_station_list[1].strategy.actor_loss_list, 'm-', linewidth=2)
                axes[1, 1].set_xlabel('Training Step', fontsize=12)
                axes[1, 1].set_ylabel('Actor Loss', fontsize=12)
                axes[1, 1].set_title('Actor Loss Trend - Base Station 1', fontsize=14)
                axes[1, 1].grid(True, alpha=0.3)

        # 第三个基站
        if len(base_station_list) > 2:
            axes[0, 2].plot(base_station_list[2].strategy.reward_list, 'c-', linewidth=2)
            axes[0, 2].set_xlabel('Episode', fontsize=12)
            axes[0, 2].set_ylabel('Reward', fontsize=12)
            axes[0, 2].set_title('Reward Trend - Base Station 2', fontsize=14)
            axes[0, 2].grid(True, alpha=0.3)

            if base_station_list[2].strategy.actor_loss_list:
                axes[1, 2].plot(base_station_list[2].strategy.actor_loss_list, 'y-', linewidth=2)
                axes[1, 2].set_xlabel('Training Step', fontsize=12)
                axes[1, 2].set_ylabel('Actor Loss', fontsize=12)
                axes[1, 2].set_title('Actor Loss Trend - Base Station 2', fontsize=14)
                axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 单独绘制Critic Loss图（如果base_station_list[0]有数据）
        if base_station_list[0].strategy.critic_loss_list:
            plt.figure(figsize=(12, 6))
            plt.plot(base_station_list[0].strategy.critic_loss_list, 'g-', linewidth=2)
            plt.xlabel('Training Step', fontsize=12)
            plt.ylabel('Critic Loss (MSE)', fontsize=12)
            plt.title('Critic Loss Trend - Base Station 0', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
