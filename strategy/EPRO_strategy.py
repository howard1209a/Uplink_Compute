import numpy as np

from constant import GAMMA1, GAMMA2
from strategy.strategy import Strategy


class EPRO(Strategy):
    def __init__(self):
        super().__init__()

    # 不支持通信基站间转发，只支持边缘服务器卸载和概率丢弃
    def decide(self, base_station, time_slot):
        length = len(base_station.task_queue)
        if length == 0:
            return

        task_list = []
        drop_rate = 0.25

        # 第一步：概率丢弃
        for _ in range(length):
            task = base_station.task_queue.pop(0)
            if np.random.uniform(0, 1) < drop_rate:
                task.drop()
            else:
                task_list.append(task)

        if len(task_list) == 0:
            return

        edge_server_list = base_station.edge_servers

        if len(edge_server_list) == 1:
            for task in task_list:
                base_station.offload(task, edge_server_list[0])
            return

        allocated_tasks = set()  # 记录已分配的任务

        while len(allocated_tasks) < len(task_list):
            task_scores = {}
            for task in task_list:
                scores = []
                for edge_server in edge_server_list:
                    score = self.compute_score(task, edge_server, base_station)
                    scores.append((edge_server, score))

                # 按分值从低到高排序（分值越低越好）
                scores.sort(key=lambda x: x[1])
                task_scores[task] = scores

            # 第三步：遍历边缘服务器，为每个边缘服务器分配任务
            for edge_server in edge_server_list:
                # 找出所有未分配任务中，该边缘服务器是最佳选择的任务
                candidate_tasks = []

                for task in task_list:
                    if task in allocated_tasks:
                        continue

                    # 获取该任务的得分排序
                    scores = task_scores[task]

                    # 检查该边缘服务器是否是此任务的最佳选择
                    if scores[0][0] == edge_server:
                        # 计算最佳分值和次佳分值之间的差距
                        best_score = scores[0][1]
                        second_best_score = scores[1][1]
                        score_gap = second_best_score - best_score  # 差距越大越好

                        candidate_tasks.append((task, score_gap))

                # 如果有满足条件的任务
                if len(candidate_tasks) > 0:
                    # 找出最佳分值和次佳分值差距最大的任务
                    best_candidate = max(candidate_tasks, key=lambda x: x[1])
                    selected_task = best_candidate[0]

                    # 执行卸载
                    base_station.offload(selected_task, edge_server)
                    allocated_tasks.add(selected_task)

    def compute_score(self, task, edge_server, base_station):
        e_s = edge_server
        transmit_time = task.tile.data_size / base_station.edge_server_channel_map[edge_server].R
        compute_time = task.c * (1 + e_s.IO_conflict_factor) ** e_s.p / e_s.get_task_f() + task.g / e_s.get_task_u()
        return transmit_time * GAMMA1 + compute_time * GAMMA2

    def post_handle_per_slot(self, base_station, time_slot, result_per_slot):
        pass
