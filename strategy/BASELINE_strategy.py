import random

import numpy as np
from numpy.ma.core import choose

from strategy.strategy import Strategy


class BASELINE(Strategy):
    def __init__(self):
        super().__init__()

    def decide(self, base_station, time_slot):
        if len(base_station.task_queue) == 0:
            return

        first_task = base_station.task_queue.pop(0)
        choice = np.random.uniform(0, 1)
        if choice < 0.3:
            if len(base_station.base_stations) > 0:
                base_station.transmit(first_task, random.choice(base_station.base_stations))
            else:
                self.best_offload(base_station, first_task)
        elif choice < 0.55:
            first_task.drop()
        else:
            self.best_offload(base_station, first_task)

    # def best_offload(self, base_station, first_task):
    #     best_edge_server = None
    #     for edge_server in base_station.edge_servers:
    #         if best_edge_server is None:
    #             best_edge_server = edge_server
    #             continue
    #         if edge_server.get_task_f() > best_edge_server.get_task_f() and edge_server.get_task_u() > best_edge_server.get_task_u():
    #             best_edge_server = edge_server
    #     base_station.offload(first_task, best_edge_server)

    def best_offload(self, base_station, first_task):
        best_edge_server = None
        for edge_server in base_station.edge_servers:
            if best_edge_server is None:
                best_edge_server = edge_server
                continue
            if edge_server.p < best_edge_server.p:
                best_edge_server = edge_server
        base_station.offload(first_task, best_edge_server)

    def post_handle_per_slot(self, base_station, time_slot, result_per_slot):
        pass
