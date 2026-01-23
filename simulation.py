import random

import pandas as pd
import yaml
import os
import re

from entity import BaseStation, EdgeServer, Video
from strategy import ProCES360, Random360


def get_video_sizes(folder_path):
    time_slots = 29
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
        view_list.append(df.loc[i, 'high_frequency_faces'])
    return view_list


def register_strategy(base_station_list, strategy_name):
    for base_station in base_station_list:
        if strategy_name == "ProCES-360":
            base_station.register_strategy(ProCES360())
        elif strategy_name == "Random360":
            base_station.register_strategy(Random360())
        else:
            raise ValueError("yaml策略名无效")


base_station_count = 5
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

base_station_2_edge_server_line_ratio = 0.3
base_station_2_edge_server_line_count = int(
    base_station_count * edge_server_count * base_station_2_edge_server_line_ratio)

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
    video_list.append(Video(video_name, get_video_sizes(file_path), get_video_views(view_path)))

if len(video_list) < len(base_station_list):
    raise ValueError("yaml配置视频数不能少于设置的通信基站数")

# 随机为视频分配基站，保证每个基站至少分配到一个
random.shuffle(video_list)
for index in range(len(video_list)):
    if index < len(base_station_list):
        base_station_list[index].register_video(video_list[index])
    else:
        random.choice(base_station_list).register_video(video_list[index])

print("---------------------------start simulation---------------------------")

strategy_name_list = config.get("strategies", [])
# 每个策略模拟一次
for strategy_name in strategy_name_list:
    register_strategy(base_station_list, strategy_name)

    # 依次模拟每个时隙，一个时隙2s
    for time_slot in range(1):
        # 清空资源
        for base_station in base_station_list:
            base_station.clear()
        for edge_server in edge_server_list:
            edge_server.clear()

        # 对于每个通信基站，初始化本时隙的任务队列
        for base_station in base_station_list:
            base_station.init_task_queue(time_slot)

        while True:
            for base_station in base_station_list:
                base_station.interact()

            all_clean = True
            for base_station in base_station_list:
                all_clean = all_clean and base_station.task_clean()

            if all_clean:
                break

        transmit_time = 0
        compute_time = 0
        consumed_energy = 0

        for base_station in base_station_list:
            for task in base_station.origin_task_list:
                task.collect_statistics()
                transmit_time += task.transmit_time
                compute_time += task.compute_time
                consumed_energy += task.consumed_energy

        print("transmit_time: " + str(transmit_time))
        print("compute_time: " + str(compute_time))
        print("consumed_energy: " + str(consumed_energy))
