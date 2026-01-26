import math
import random
import numpy as np


class BaseStation:
    def __init__(self, index):
        self.index = index

        self.antenna_count = random.randint(2, 5)

        self.origin_task_list = []
        self.task_queue = []
        self.task_queue_length = None

        self.task_cache_list = []

        self.edge_servers = []
        self.edge_server_channel_map = {}
        self.base_stations = []
        self.base_station_channel_map = {}

        self.video_list = []

        self.strategy = None

    def connnet_base_station(self, base_station):
        self.base_stations.append(base_station)
        self.base_station_channel_map[base_station] = Channel(self, base_station, False)

        base_station.base_stations.append(self)
        base_station.base_station_channel_map[self] = Channel(base_station, self, False)

    def connect_edge_server(self, edge_server):
        self.edge_servers.append(edge_server)
        self.edge_server_channel_map[edge_server] = Channel(self, edge_server, True)

    def already_connected(self, node):
        for base_station in self.base_stations:
            if node == base_station:
                return True
        for edge_server in self.edge_servers:
            if edge_server == node:
                return True
        return False

    def register_video(self, video):
        self.video_list.append(video)

    def register_strategy(self, strategy):
        self.strategy = strategy

    def init_task_queue(self, time_slot):
        index = 0
        for video in self.video_list:
            tile_list_slot = video.get_tile_list_slot(time_slot)
            for tile in tile_list_slot:
                task = Task(self, index, tile)
                self.task_queue.append(task)
                self.origin_task_list.append(task)
                index += 1

    # 清空资源以开启新时隙模拟
    def clear(self):
        self.origin_task_list = []
        self.task_queue = []
        self.task_cache_list = []

    def init_before_simulation(self):
        # 此时视频列表已分配完毕，任务队列长度在后续模拟过程固定
        self.task_queue_length = 6 * len(self.video_list)

    def task_clean(self):
        return len(self.task_queue) == 0 and len(self.task_cache_list) == 0

    def interact(self, time_slot):
        if len(self.task_cache_list) > 0:
            for cache_task in self.task_cache_list:
                # 如果来自其他通信基站的转发任务的转发次数耗尽，或者当前任务队列已经清空，或者当前任务队列已经满了，都会将任务随机卸载到相连边缘服务器
                if (not cache_task.can_transmit()) or (len(self.task_queue) == 0) or (
                        len(self.task_queue) == self.task_queue_length):
                    self.offload(cache_task, random.choice(self.edge_servers))
                else:
                    self.task_queue.append(cache_task)

        self.task_cache_list = []

        # 策略针对通信基站执行决策
        self.strategy.decide(self, time_slot)

    def receive(self, task):
        self.task_cache_list.append(task)

    def offload(self, task, edge_server):
        task.offloaded_to_edge_server(edge_server)
        edge_server.receive_task(task)

        task.transmit_time += task.tile.data_size / self.edge_server_channel_map[edge_server].R

    # 将某个已经卸载的任务从边缘服务器取回，执行与offload方法相反的操作，注意此方法必须与offload方法在同一次信息流交互执行
    def withdraw_offload(self, task):
        edge_server = task.offloaded_edge_server
        task.offloaded_edge_server = None
        edge_server.withdraw_task(task)

        task.transmit_time -= task.tile.data_size / self.edge_server_channel_map[edge_server].R



    def transmit(self, task, next_base_station):
        if task.l <= 0:
            raise ValueError("传输次数耗尽的任务执行了传输")
        task.l -= 1

        next_base_station.receive(task)

        task.transmit_time += task.tile.data_size / self.base_station_channel_map[next_base_station].R

    # # 获取通信基站内所有视频在当前时隙的视野内比特率之和
    # def collect_video_quality(self, time_slot):
    #     video_quality = 0
    #     for video in self.video_list:
    #         user_view_list_slot = video.user_view_list[time_slot]
    #         average_user_bitrate = 0
    #         for user_view in user_view_list_slot:
    #             user_bitrate = 0
    #             for tile_index in user_view:
    #                 user_bitrate += self.query_tile_bitrate(tile_index, video, time_slot)
    #             average_user_bitrate += user_bitrate / len(user_view)
    #         average_user_bitrate /= len(user_view_list_slot)
    #         video_quality += average_user_bitrate
    #
    #     return video_quality

    # # 获取通信基站内所有视频在当前时隙的视野内比特率之和
    # def collect_video_quality(self, time_slot):
    #     video_quality = 0
    #     for video in self.video_list:
    #         user_bitrate = 0
    #         view_list_slot = video.view_list[time_slot]
    #         for tile_index in view_list_slot:
    #             b=self.query_tile_bitrate(tile_index, video, time_slot)
    #             user_bitrate += self.query_tile_bitrate(tile_index, video, time_slot)
    #         video_quality += user_bitrate / len(view_list_slot)
    #
    #     return video_quality

    # 获取通信基站内所有视频在当前时隙的视野内比特率之和，随机抽取一个用户
    def collect_video_quality(self, time_slot):
        video_quality = 0
        for video in self.video_list:
            user_view = random.choice(video.user_view_list[time_slot])
            user_bitrate = 0
            for tile_index in user_view:
                user_bitrate += self.query_tile_bitrate(tile_index, video, time_slot)
            video_quality += user_bitrate / len(user_view)

        return video_quality

    def query_tile_bitrate(self, tile_index, video, time_slot):
        for tile in video.tile_list[time_slot]:
            if tile.index == tile_index:
                return tile.bitrate
        raise ValueError("查询瓦片比特率失败")

    def post_handle_per_slot(self, base_station, time_slot, result_per_slot):
        self.strategy.post_handle_per_slot(base_station, time_slot, result_per_slot)


class EdgeServer:
    def __init__(self, index):
        self.index = index

        self.f = np.random.uniform(20, 32) * 1e9
        self.u = np.random.uniform(4, 6) * 1e12
        self.k = np.random.uniform(1, 4.0) * 1e-27
        self.p = 0
        self.IO_conflict_factor = 0.1

    # 清空资源以开启新时隙模拟
    def clear(self):
        self.p = 0

    def receive_task(self, task):
        self.p += 1

    def withdraw_task(self, task):
        self.p -= 1

    def get_task_f(self):
        if self.p == 0:
            return self.f
        return self.f / self.p

    def get_task_u(self):
        if self.p == 0:
            return self.u
        return self.u / self.p


class Channel:
    def __init__(self, from_node, to_node, is_base_station_to_edge_server):
        self.B = random.uniform(14.0, 30.0) * 1e6
        self.E_s_divide_N_0 = random.uniform(16.0, 24.0)
        self.h_all = 0
        if is_base_station_to_edge_server:
            for _ in range(from_node.antenna_count):
                self.h_all += np.random.exponential(scale=1.0);
        else:
            for _ in range(from_node.antenna_count):
                for _ in range(to_node.antenna_count):
                    self.h_all += np.random.exponential(scale=1.0);
        self.SNR = self.E_s_divide_N_0 * self.h_all
        self.R = self.B * math.log2(1 + self.SNR)


class Video:
    def __init__(self, name, data_size_list, view_list, user_view_list):
        self.name = name
        self.tile_list = []
        for time_slot in range(len(data_size_list)):
            data_size_slot_list = data_size_list[time_slot]
            tile_slot_list = []
            for tile_index in range(len(data_size_slot_list)):
                tile_slot_list.append(Tile(tile_index + 1, data_size_slot_list[tile_index], self, time_slot))
            self.tile_list.append(tile_slot_list)
        self.view_list = view_list
        self.user_view_list = user_view_list

    def get_tile_list_slot(self, time_slot):
        return self.tile_list[time_slot]

    def check_tile_index_max_frequency(self, tile_index, time_slot):
        return tile_index in self.view_list[time_slot]

    def clear(self):
        for tile_list_slot in self.tile_list:
            for tile in tile_list_slot:
                tile.clear()


class Tile:
    def __init__(self, index, data_size, video, time_slot):
        self.index = index
        # 单位bit
        self.data_size = data_size
        # 瓦片长度固定2s，比特率单位bps
        self.bitrate = self.data_size / 2.0
        self.video = video
        # 瓦片所属时隙是固定的
        self.time_slot = time_slot

    # 瓦片丢弃，放弃转码，比特率为0，会看到黑块
    def drop(self):
        self.bitrate = 0

    def clear(self):
        self.bitrate = self.data_size / 2.0


class Task:
    def __init__(self, base_station, index, tile):
        self.base_station = base_station
        self.index = index
        self.c = np.random.uniform(12, 20) * 1e7
        self.g = np.random.uniform(27, 39) * 1e9
        self.l = random.randint(1, 6)

        self.offloaded_edge_server = None

        self.tile = tile

        self.transmit_time = 0
        self.compute_time = None
        self.consumed_energy = None

        self.dropped = False

    def can_transmit(self):
        return self.l > 0

    def offloaded_to_edge_server(self, edge_server):
        self.offloaded_edge_server = edge_server

    def collect_statistics(self):
        if self.dropped:
            self.compute_time = 0
            self.consumed_energy = 0
            return

        e_s = self.offloaded_edge_server
        a = self.c * (
                1 + e_s.IO_conflict_factor) ** e_s.p / e_s.get_task_f()
        b = self.g / e_s.get_task_u()
        self.compute_time = self.c * (
                1 + e_s.IO_conflict_factor) ** e_s.p / e_s.get_task_f() + self.g / e_s.get_task_u()

        self.consumed_energy = e_s.k * self.c * (e_s.get_task_f()) ** 2

    def drop(self):
        self.dropped = True
        self.tile.drop()


class ResultPerSlot:
    def __init__(self, transmit_time, compute_time, consumed_energy, video_quality):
        self.transmit_time = transmit_time
        self.compute_time = compute_time
        self.consumed_energy = consumed_energy
        self.video_quality = video_quality
