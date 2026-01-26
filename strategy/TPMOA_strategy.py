import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import torch.nn.functional as F

from constant import GAMMA1, GAMMA2, GAMMA3, GAMMA4
from strategy.strategy import Strategy


class TPMOA(Strategy):
    def __init__(self, base_station):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # DQN参数
        self.edge_server_count = len(base_station.edge_servers)
        self.base_station_count = len(base_station.base_stations)
        self.action_dim = self.edge_server_count + self.base_station_count + 1  # +1 for drop action
        self.state_dim = None  # 将在extract_state中计算

        # 神经网络参数
        self.hidden_dim = 256
        self.lr = 0.001
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start
        self.batch_size = 64
        self.target_update_freq = 100

        # 经验回放池
        self.memory = deque(maxlen=10000)

        # 网络
        self.policy_net = None
        self.target_net = None
        self.optimizer = None

        # 训练计数器
        self.steps_done = 0
        self.learn_step_counter = 0

        # 状态记录
        self.pre_state = None
        self.pre_action = None
        self.pre_reward = None

        # 损失记录
        self.loss_history = []

        # 初始化网络
        self._init_networks()

        # 训练状态
        self.training_mode = True

    def _init_networks(self):
        """初始化神经网络"""
        # 状态维度计算：边缘服务器特征 + 队头任务特征 + 任务队列长度
        state_dim = self.edge_server_count * 3 + 3 + 1  # 3: f_per_task, u_per_task, R for each server; 3: c, g, data_size for task; 1: queue_length

        self.policy_net = DQN(state_dim, self.hidden_dim, self.action_dim).to(self.device)
        self.target_net = DQN(state_dim, self.hidden_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def extract_state(self, base_station, time_slot):
        """提取状态特征"""
        state_features = []

        # 1. 边缘服务器信息
        for edge_server in base_station.edge_servers:
            # CPU可分配资源
            f_per_task = edge_server.get_task_f()
            # GPU可分配资源
            u_per_task = edge_server.get_task_u()
            # 传输速率
            channel = base_station.edge_server_channel_map.get(edge_server)
            R = channel.R if channel else 0

            state_features.extend([f_per_task, u_per_task, R])

        # 2. 队头任务信息
        if len(base_station.task_queue) > 0:
            first_task = base_station.task_queue[0]
            state_features.extend([first_task.c, first_task.g, first_task.tile.data_size])
        else:
            # 如果没有任务，用0填充
            state_features.extend([0, 0, 0])

        # 3. 任务队列长度
        queue_length = len(base_station.task_queue)
        state_features.append(queue_length)

        return np.array(state_features, dtype=np.float32)

    def choose_action(self, state, training=True):
        """根据状态选择动作（epsilon-greedy）"""
        if training and np.random.random() < self.epsilon:
            # 随机探索
            return np.random.randint(0, self.action_dim)

        # 利用策略网络选择最优动作
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # 切换到评估模式以避免BatchNorm问题
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        # 恢复训练模式
        if self.training_mode:
            self.policy_net.train()

        return torch.argmax(q_values).item()

    def decode_action(self, action_idx, base_station):
        """解码动作"""
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
        """从经验回放池中学习"""
        if len(self.memory) < self.batch_size:
            return

        # 从回放池中采样
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为张量
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 计算当前Q值
        current_q_values = self.policy_net(states).gather(1, actions)

        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # 计算损失
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)

        self.optimizer.step()

        # 记录损失
        self.loss_history.append(loss.item())

        # 更新epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        # 更新目标网络
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decide(self, base_station, time_slot):
        """决策函数：处理队头任务"""
        if len(base_station.task_queue) == 0:
            return

        # 提取当前状态
        state = self.extract_state(base_station, time_slot)

        # 选择动作
        action = self.choose_action(state, self.training_mode)

        # 如果之前有状态和动作，存储经验（奖励在post_handle_per_slot中计算）
        if self.pre_state is not None and self.pre_action is not None:
            # 临时存储，奖励稍后计算
            self.pre_reward = 0  # 临时值，将在post_handle_per_slot中更新
            self.store_transition(self.pre_state, self.pre_action, self.pre_reward, state, False)

        # 保存当前状态和动作
        self.pre_state = state
        self.pre_action = action

        # 解码并执行动作
        first_task = base_station.task_queue.pop(0)
        action_type, target_node = self.decode_action(action, base_station)

        if action_type == 'offload':
            base_station.offload(first_task, target_node)
        elif action_type == 'transmit':
            if first_task.can_transmit():
                base_station.transmit(first_task, target_node)
            else:
                # 转发次数耗尽，随机卸载
                base_station.offload(first_task, random.choice(base_station.edge_servers))
        elif action_type == 'drop':
            first_task.drop()
        else:
            raise ValueError("TPMOA策略无效决策")

    def post_handle_per_slot(self, base_station, time_slot, result_per_slot):
        """时隙后处理"""
        # 提取最终状态
        final_state = self.extract_state(base_station, time_slot)

        # 计算最终奖励
        if self.pre_state is not None and self.pre_action is not None:
            # 计算奖励
            reward = self.calculate_reward(result_per_slot)

            # 更新最后一条经验（临时存储的那条）
            if len(self.memory) > 0:
                # 更新最后一条经验的奖励和下一个状态
                last_experience = self.memory.pop()
                _, action, _, _, _ = last_experience
                self.memory.append((self.pre_state, action, reward, final_state, True))

            # 学习
            self.learn()

        # 重置状态记录
        self.pre_state = None
        self.pre_action = None
        self.pre_reward = None

        # 增加步数计数
        self.steps_done += 1

    def calculate_reward(self, result_per_slot):
        """计算奖励"""
        if result_per_slot is None:
            return 0

        try:
            # 使用与ProCES360类似的奖励函数
            transmit_reward = -np.log(result_per_slot.transmit_time * GAMMA1 + 1e-5)
            compute_reward = -np.log(result_per_slot.compute_time * GAMMA2 + 1e-5)
            energy_reward = -np.log(result_per_slot.consumed_energy * GAMMA3 + 1e-5)
            video_quality_reward = result_per_slot.video_quality * GAMMA4 * -3

            total_reward = transmit_reward + compute_reward + energy_reward + video_quality_reward
            return total_reward
        except:
            return 0

    def set_training_mode(self, mode):
        """设置训练模式"""
        self.training_mode = mode
        if mode:
            self.policy_net.train()
        else:
            self.policy_net.eval()


class DQN(nn.Module):
    """DQN网络"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)  # 使用LayerNorm替代BatchNorm

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)  # 使用LayerNorm替代BatchNorm

        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)  # 使用LayerNorm替代BatchNorm

        self.fc4 = nn.Linear(hidden_dim // 2, output_dim)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # LayerNorm不依赖于batch size，因此可以处理单个样本
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.fc4(x)
        return x