from functools import cmp_to_key
import random


class MFQAS:
    def __init__(self):
        pass

    def decide(self, base_station, time_slot):
        task_count = len(base_station.task_queue)
        if task_count == 0:
            return

        task_list = []
        for _ in range(task_count):
            task_list.append(base_station.task_queue.pop(0))

        # 1. 按视野信息排序
        task_list.sort(key=cmp_to_key(self.comparator))

        consumed_time_upper_bound = 2.2
        edge_server_list = base_station.edge_servers

        # 3. 遍历任务列表，均摊卸载
        for i, task in enumerate(task_list):
            # 选择当前应该分配到的边缘服务器（轮询方式）
            target_server = edge_server_list[i % len(edge_server_list)]

            # 先尝试卸载
            base_station.offload(task, target_server)

            # 计算当前已卸载任务的总耗时
            current_total_time = self.compute_consumed_time_all(i + 1, task_list, base_station)

            if current_total_time <= consumed_time_upper_bound:
                # 总耗时在阈值内，继续下一个任务
                continue
            else:
                # 总耗时超过阈值，撤回本次卸载
                base_station.withdraw_offload(task)

                # 撤回随机选择的任务
                pre_task = random.choice(task_list[0:i])
                base_station.withdraw_offload(pre_task)

                # 为撤回的任务随机选择一个新边缘服务器
                base_station.offload(pre_task, random.choice(edge_server_list))

                # 重新卸载当前任务
                base_station.offload(task, target_server)

                # 再次计算总耗时
                current_total_time = self.compute_consumed_time_all(i + 1, task_list, base_station)

                if current_total_time <= consumed_time_upper_bound:
                    # 调整后满足条件，继续下一个任务
                    continue
                else:
                    # 调整后仍不满足，结束循环，剩余任务丢弃
                    # 先撤回当前任务
                    base_station.withdraw_offload(task)

                    # 丢弃剩余任务（包括当前任务）
                    for remaining_task in task_list[i:]:
                        remaining_task.drop()
                    break

    @staticmethod
    def comparator(task1, task2):
        video1 = task1.tile.video
        in_view1 = video1.check_tile_index_max_frequency(task1.tile.index, task1.tile.time_slot)
        video2 = task2.tile.video
        in_view2 = video2.check_tile_index_max_frequency(task2.tile.index, task2.tile.time_slot)

        if in_view1 == True and in_view2 == False:
            return -1
        elif in_view1 == False and in_view2 == True:
            return 1
        else:
            return 0

    def compute_consumed_time_all(self, offloaded_task_count, task_list, base_station):
        time_all = 0
        for index in range(offloaded_task_count):
            task = task_list[index]
            time_all += self.compute_consumed_time(task, task.offloaded_edge_server, base_station)
        return time_all

    def compute_consumed_time(self, task, edge_server, base_station):
        e_s = edge_server
        transmit_time = task.tile.data_size / base_station.edge_server_channel_map[edge_server].R
        compute_time = task.c * (1 + e_s.IO_conflict_factor) ** e_s.p / e_s.get_task_f() + task.g / e_s.get_task_u()
        return transmit_time + compute_time

    def post_handle_per_slot(self, base_station, time_slot, result_per_slot):
        pass
