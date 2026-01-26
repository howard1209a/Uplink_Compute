from functools import cmp_to_key


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

        # 直接传递 comparator 函数，不需要加括号
        task_list.sort(key=cmp_to_key(self.comparator))

        consumed_time_upper_bound = 20
        edge_server_list = base_station.edge_servers

        print(1)

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
