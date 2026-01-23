import random

import numpy as np


class Strategy:
    def __init__(self):
        pass

    def decide(self, base_station):
        pass


class ProCES360(Strategy):
    def __init__(self):
        super().__init__()

    def decide(self, base_station):
        pass


class Random360(Strategy):
    def __init__(self):
        super().__init__()

    def decide(self, base_station):
        if base_station.task_queue_clean():
            return

        first_task = base_station.task_queue.pop(0)
        if len(base_station.base_stations) > 0 and np.random.uniform(0, 1) > 0.5:
            base_station.transmit(first_task, random.choice(base_station.base_stations))
        else:
            base_station.offload(first_task, random.choice(base_station.edge_servers))
