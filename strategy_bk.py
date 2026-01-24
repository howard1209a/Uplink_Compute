from constant import GAMMA1, GAMMA2, GAMMA3, GAMMA4


class Strategy:
    def __init__(self):
        pass

    def decide(self, base_station, time_slot):
        pass

    def post_handle_per_slot(self, base_station, time_slot, result_per_slot):
        pass


class Random360(Strategy):
    def __init__(self):
        super().__init__()

    def decide(self, base_station, time_slot):
        if len(base_station.task_queue) == 0:
            return

        first_task = base_station.task_queue.pop(0)
        if len(base_station.base_stations) > 0 and np.random.uniform(0, 1) > 0.5:
            base_station.transmit(first_task, random.choice(base_station.base_stations))
        else:
            base_station.offload(first_task, random.choice(base_station.edge_servers))

    def post_handle_per_slot(self, base_station, time_slot, result_per_slot):
        pass


import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class ActorNetwork(nn.Module):
    """Actor网络 - 策略网络（修改后的结构）"""

    def __init__(self, edge_server_count, max_tasks, hidden_dim=256, action_dim=None):
        super(ActorNetwork, self).__init__()
        self.edge_server_count = edge_server_count
        self.max_tasks = max_tasks

        # LayerNorm层 - 对边缘服务器信息进行归一化
        self.edge_server_layernorm = nn.LayerNorm(3)  # 3个特征: CPU, GPU, 传输速率

        # Embedding层
        self.edge_server_embedding = nn.Linear(3, 32)  # 3维 -> 32维

        # LayerNorm层 - 对计算任务信息进行归一化
        self.task_layernorm = nn.LayerNorm(3)  # 3个特征: CPU周期, GPU周期, 数据量

        # Embedding层
        self.task_embedding = nn.Linear(3, 32)  # 3维 -> 32维

        # 计算总输入维度
        total_embedding_dim = (edge_server_count * 32) + (max_tasks * 32) + 1  # +1 用于用户视野信息

        # 后续全连接层
        self.fc1 = nn.Linear(total_embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # 激活函数
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        # 初始化权重
        nn.init.xavier_uniform_(self.edge_server_embedding.weight)
        nn.init.xavier_uniform_(self.task_embedding.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        """
        前向传播
        x: 输入状态向量 [batch_size, state_dim]
        """
        batch_size = x.size(0)

        # 解析输入向量
        # 假设状态向量结构: [边缘服务器特征 * edge_server_count * 3] + [任务特征 * max_tasks * 3] + [用户视野信息]
        edge_server_features_dim = self.edge_server_count * 3
        task_features_dim = self.max_tasks * 3

        # 提取边缘服务器特征并重塑为 [batch_size, edge_server_count, 3]
        edge_server_features = x[:, :edge_server_features_dim].view(batch_size, self.edge_server_count, 3)

        # 提取任务特征并重塑为 [batch_size, max_tasks, 3]
        task_features = x[:, edge_server_features_dim:edge_server_features_dim + task_features_dim].view(batch_size,
                                                                                                         self.max_tasks,
                                                                                                         3)

        # 提取用户视野信息 [batch_size, 1]
        view_info = x[:, -1].unsqueeze(1)

        # 1. 对边缘服务器信息进行LayerNorm
        edge_server_features = self.edge_server_layernorm(edge_server_features)

        # 2. 边缘服务器信息embedding
        edge_server_embedded = self.edge_server_embedding(edge_server_features)  # [batch_size, edge_server_count, 32]
        edge_server_embedded = edge_server_embedded.view(batch_size, -1)  # 展平 [batch_size, edge_server_count*32]

        # 3. 对计算任务信息进行LayerNorm
        task_features = self.task_layernorm(task_features)

        # 4. 计算任务信息embedding
        task_embedded = self.task_embedding(task_features)  # [batch_size, max_tasks, 32]
        task_embedded = task_embedded.view(batch_size, -1)  # 展平 [batch_size, max_tasks*32]

        # 5. 拼接所有特征
        combined = torch.cat([edge_server_embedded, task_embedded, view_info], dim=1)

        # 6. 两层全连接层
        x = self.relu(self.fc1(combined))
        x = self.relu(self.fc2(x))

        # 7. 输出层 + softmax激活函数
        x = self.fc3(x)
        x = self.softmax(x)

        return x


class CriticNetwork(nn.Module):
    """Critic网络 - 价值网络（修改后的结构）"""

    def __init__(self, edge_server_count, max_tasks, hidden_dim=256, output_dim=1):
        super(CriticNetwork, self).__init__()
        self.edge_server_count = edge_server_count
        self.max_tasks = max_tasks

        # LayerNorm层 - 对边缘服务器信息进行归一化
        self.edge_server_layernorm = nn.LayerNorm(3)  # 3个特征: CPU, GPU, 传输速率

        # Embedding层
        self.edge_server_embedding = nn.Linear(3, 32)  # 3维 -> 32维

        # LayerNorm层 - 对计算任务信息进行归一化
        self.task_layernorm = nn.LayerNorm(3)  # 3个特征: CPU周期, GPU周期, 数据量

        # Embedding层
        self.task_embedding = nn.Linear(3, 32)  # 3维 -> 32维

        # 计算总输入维度
        total_embedding_dim = (edge_server_count * 32) + (max_tasks * 32) + 1  # +1 用于用户视野信息

        # 后续全连接层
        self.fc1 = nn.Linear(total_embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # 激活函数
        self.relu = nn.ReLU()

        # 初始化权重
        nn.init.xavier_uniform_(self.edge_server_embedding.weight)
        nn.init.xavier_uniform_(self.task_embedding.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        """
        前向传播
        x: 输入状态向量 [batch_size, state_dim]
        """
        batch_size = x.size(0)

        # 解析输入向量
        # 假设状态向量结构: [边缘服务器特征 * edge_server_count * 3] + [任务特征 * max_tasks * 3] + [用户视野信息]
        edge_server_features_dim = self.edge_server_count * 3
        task_features_dim = self.max_tasks * 3

        # 提取边缘服务器特征并重塑为 [batch_size, edge_server_count, 3]
        edge_server_features = x[:, :edge_server_features_dim].view(batch_size, self.edge_server_count, 3)

        # 提取任务特征并重塑为 [batch_size, max_tasks, 3]
        task_features = x[:, edge_server_features_dim:edge_server_features_dim + task_features_dim].view(batch_size,
                                                                                                         self.max_tasks,
                                                                                                         3)

        # 提取用户视野信息 [batch_size, 1]
        view_info = x[:, -1].unsqueeze(1)

        # 1. 对边缘服务器信息进行LayerNorm
        edge_server_features = self.edge_server_layernorm(edge_server_features)

        # 2. 边缘服务器信息embedding
        edge_server_embedded = self.edge_server_embedding(edge_server_features)  # [batch_size, edge_server_count, 32]
        edge_server_embedded = edge_server_embedded.view(batch_size, -1)  # 展平 [batch_size, edge_server_count*32]

        # 3. 对计算任务信息进行LayerNorm
        task_features = self.task_layernorm(task_features)

        # 4. 计算任务信息embedding
        task_embedded = self.task_embedding(task_features)  # [batch_size, max_tasks, 32]
        task_embedded = task_embedded.view(batch_size, -1)  # 展平 [batch_size, max_tasks*32]

        # 5. 拼接所有特征
        combined = torch.cat([edge_server_embedded, task_embedded, view_info], dim=1)

        # 6. 两层全连接层
        x = self.relu(self.fc1(combined))
        x = self.relu(self.fc2(x))

        # 7. 输出层（使用Linear激活函数）
        x = self.fc3(x)

        return x


class ProCES360(Strategy):
    def __init__(self, base_station):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 神经网络参数
        self.edge_server_count = None  # 将在运行时确定
        self.max_tasks = None  # 将在运行时确定
        self.action_dim = None  # 将在运行时确定
        self.hidden_dim = 256
        self.lr = 0.0003
        self.gamma = 0.99
        self.epsilon = 0.2
        self.epochs = 5
        self.batch_size = 10

        # 经验回放池
        self.memory = deque(maxlen=10000)

        # 在运行时初始化网络
        self.actor = None
        self.critic = None
        self.actor_optimizer = None
        self.critic_optimizer = None

        # 目标网络
        self.actor_target = None
        self.critic_target = None

        # 训练相关
        self.learn_step_counter = 0
        self.target_update_interval = 100

        # 获取参数
        self.edge_server_count = len(base_station.edge_servers)
        base_station_count = len(base_station.base_stations)
        self.max_tasks = base_station.task_queue_length

        # 动作空间：卸载到每个边缘服务器 + 转发到每个相邻基站 + 丢弃
        self.action_dim = self.edge_server_count + base_station_count + 1

        # 初始化Actor网络
        self.actor = ActorNetwork(
            edge_server_count=self.edge_server_count,
            max_tasks=self.max_tasks,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim
        ).to(self.device)

        # 初始化Critic网络
        self.critic = CriticNetwork(
            edge_server_count=self.edge_server_count,
            max_tasks=self.max_tasks,
            hidden_dim=self.hidden_dim,
            output_dim=1
        ).to(self.device)

        # 初始化目标网络
        self.actor_target = ActorNetwork(
            edge_server_count=self.edge_server_count,
            max_tasks=self.max_tasks,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim
        ).to(self.device)

        self.critic_target = CriticNetwork(
            edge_server_count=self.edge_server_count,
            max_tasks=self.max_tasks,
            hidden_dim=self.hidden_dim,
            output_dim=1
        ).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.pre_state = None
        self.pre_action_idx = None

    def extract_state(self, base_station, time_slot):
        """从基站提取状态特征"""
        state_features = []

        # 1. 边缘服务器信息
        for edge_server in base_station.edge_servers:
            # CPU可分配资源
            f_per_task = edge_server.get_task_f()
            # GPU可分配资源
            u_per_task = edge_server.get_task_u()
            # 传输速率
            channel = base_station.edge_server_channel_map.get(edge_server)

            # 归一化将在网络的LayerNorm层中处理,这里只需提供原始值
            state_features.extend([f_per_task, u_per_task, channel.R])

        # 2. 任务队列信息
        for i in range(self.max_tasks):
            if i < len(base_station.task_queue):
                task = base_station.task_queue[i]
                # 归一化将在网络的LayerNorm层中处理,这里只需提供原始值
                state_features.extend([task.c, task.g, task.tile.data_size])
            else:
                # 填充0
                state_features.extend([0, 0, 0])

        # 3. 队头任务视野信息，0代表不在最高频瓦片集合，1代表在最高频瓦片集合，如果当前任务队列为空，默认0
        view_info = 0
        if len(base_station.task_queue) > 0:
            tile = base_station.task_queue[0].tile
            view_info = 1 if tile.video.check_tile_index_max_frequency(tile.index, time_slot) else 0

        state_features.append(view_info)

        return np.array(state_features, dtype=np.float32)

    def choose_action(self, state, explore=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_probs = self.actor(state_tensor)

        if explore and random.random() < 0.1:  # 10%探索率
            action = random.randint(0, self.action_dim - 1)
        else:
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()

        return action

    def decode_action(self, action_idx, base_station):
        edge_server_count = len(base_station.edge_servers)
        base_station_count = len(base_station.base_stations)

        if action_idx < edge_server_count:
            # 卸载到边缘服务器
            edge_server = base_station.edge_servers[action_idx]
            return 'offload', edge_server
        elif action_idx < edge_server_count + base_station_count:
            # 转发到相邻基站
            base_station_idx = action_idx - edge_server_count
            next_base_station = base_station.base_stations[base_station_idx]
            return 'transmit', next_base_station
        else:
            # 丢弃任务
            return 'drop', None

    def store_transition(self, state, action, reward, next_state, done):
        """存储经验到回放池"""
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        # 从回放池采样
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为张量
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # PPO更新
        for _ in range(self.epochs):
            # 计算优势函数
            with torch.no_grad():
                next_values = self.critic_target(next_states)
                target_values = rewards + self.gamma * next_values * (1 - dones)
                advantages = target_values - self.critic(states)

            # 更新Critic
            values = self.critic(states)
            critic_loss = nn.MSELoss()(values, target_values)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

            # 更新Actor
            old_action_probs = self.actor(states).gather(1, actions).detach()
            new_action_probs = self.actor(states).gather(1, actions)

            ratio = new_action_probs / old_action_probs
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

        # 更新目标网络
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_interval == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())

    def decide(self, base_station, time_slot):
        if len(base_station.task_queue) == 0:
            return

        # 提取当前状态
        state = self.extract_state(base_station, time_slot)

        # 选择动作
        action_idx = self.choose_action(state)

        # 非本时隙的第一次决策，开始存储四元组
        if self.pre_state is not None:
            self.store_transition(self.pre_state, self.pre_action_idx, 0, state, False)
        self.pre_state = state
        self.pre_action_idx = action_idx

        # 解码动作
        action_type, target_node = self.decode_action(action_idx, base_station)

        first_task = base_station.task_queue.pop(0)

        if action_type == 'offload':
            # 直接卸载
            base_station.offload(first_task, target_node)
        elif action_type == 'transmit':
            # 如果队头任务还存在可转发次数，则可以转发
            if first_task.can_transmit():
                base_station.transmit(first_task, target_node)
            else:  # 否则随机卸载
                base_station.offload(first_task, random.choice(base_station.edge_servers))
        elif action_type == 'drop':
            # 丢弃任务
            first_task.drop()
        else:
            raise ValueError("ProCES360策略无效决策")

    def post_handle_per_slot(self, base_station, time_slot, result_per_slot):
        state = self.extract_state(base_station, time_slot)
        if self.pre_state is None or self.pre_action_idx is None:
            raise ValueError("前时隙状态或动作未存储！")

        reward = 1.0 / (result_per_slot.transmit_time * GAMMA1) + 1.0 / (
                result_per_slot.compute_time * GAMMA2) + 1.0 / (
                             result_per_slot.consumed_energy * GAMMA3) - result_per_slot.video_quality * GAMMA4
        self.store_transition(self.pre_state, self.pre_action_idx, reward, state, True)

        # 清空本轮轨迹的前置状态和动作记录
        self.pre_state = None
        self.pre_action_idx = None
