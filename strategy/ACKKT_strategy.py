import random

import numpy as np
from numpy.ma.core import choose

from strategy.strategy import Strategy


class ACKKT(Strategy):
    def __init__(self):
        super().__init__()

    def decide(self, base_station, time_slot):
        if len(base_station.task_queue) == 0:
            return

        first_task = base_station.task_queue.pop(0)

        if np.random.uniform(0, 1) < 0.25:
            first_task.drop()
            return

        all_edge_server_busy = True
        for edge_server in base_station.edge_servers:
            all_edge_server_busy = all_edge_server_busy and edge_server.p > 0
        if all_edge_server_busy and np.random.uniform(0, 1) < 0.1 and len(base_station.base_stations) > 0:
            base_station.transmit(first_task, random.choice(base_station.base_stations))

        self.best_offload(base_station, first_task)

    def best_offload(self, base_station, first_task):
        best_edge_server = None
        for edge_server in base_station.edge_servers:
            if best_edge_server is None:
                best_edge_server = edge_server
                continue
            if edge_server.get_task_f() > best_edge_server.get_task_f() and edge_server.get_task_u() > best_edge_server.get_task_u():
                best_edge_server = edge_server
        base_station.offload(first_task, best_edge_server)

    # def best_offload(self, base_station, first_task):
    #     best_edge_server = None
    #     for edge_server in base_station.edge_servers:
    #         if best_edge_server is None:
    #             best_edge_server = edge_server
    #             continue
    #         if edge_server.p < best_edge_server.p:
    #             best_edge_server = edge_server
    #     base_station.offload(first_task, best_edge_server)

    def post_handle_per_slot(self, base_station, time_slot, result_per_slot):
        pass

    # KKT朗格朗日乘子法实现计算资源分配
    @staticmethod
    def optimal_resource_allocation_with_contention(cpu_cycles, gpu_cycles, total_cpu, total_gpu, epsilon):
        """
        考虑任务竞争的资源分配优化
        Args:
            cpu_cycles: list, 每个任务所需的CPU周期数
            gpu_cycles: list, 每个任务所需的GPU周期数
            total_cpu: float, 总的CPU频率资源
            total_gpu: float, 总的GPU频率资源
            epsilon: float, 竞争系数
        Returns:
            f_optimal: list, 分配给每个任务的CPU频率
            u_optimal: list, 分配给每个任务的GPU频率
            total_time: float, 总计算时间
            contention_factor: float, 竞争影响因子
        """
        cpu_cycles = np.array(cpu_cycles)
        gpu_cycles = np.array(gpu_cycles)
        K = len(cpu_cycles)

        # 竞争因子
        contention_factor = (1 + epsilon) ** K

        # 计算平方根和
        sqrt_cpu_sum = np.sum(np.sqrt(cpu_cycles))
        sqrt_gpu_sum = np.sum(np.sqrt(gpu_cycles))

        # 最优分配（解析解）
        f_optimal = (np.sqrt(cpu_cycles) / sqrt_cpu_sum) * total_cpu
        u_optimal = (np.sqrt(gpu_cycles) / sqrt_gpu_sum) * total_gpu

        # 计算总时间（考虑竞争）
        total_time = np.sum(cpu_cycles * contention_factor / f_optimal) + np.sum(gpu_cycles / u_optimal)

        # 魔法值
        return total_time * 0.8
