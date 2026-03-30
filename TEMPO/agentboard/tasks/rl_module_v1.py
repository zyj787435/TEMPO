"""
rl_module_v1.py — RL v1 原始版 DQN 模块（存档）
问题1: epsilon=0.2 固定，不衰减
问题2: 无 epsilon_min / epsilon_decay
其余逻辑与 v2/v3 相同（Q网络结构一致）
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os


class SimpleQNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SimpleQNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)


class AutoToolRLAgentV1:
    """RL v1: epsilon=0.2 固定不衰减，无 epsilon_min/decay"""
    def __init__(self, state_dim=4, action_dim=3, lr=0.001, gamma=0.9, epsilon=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon   # 固定 0.2，永不衰减

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = SimpleQNet(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        self.memory = deque(maxlen=5000)
        self.batch_size = 64
        print("🤖 [RL-v1] Fresh start — epsilon=0.2 fixed, no decay.", flush=True)

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.q_net(state_t)).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        q_eval = self.q_net(state).gather(1, action)
        with torch.no_grad():
            q_next = self.q_net(next_state).max(1)[0].view(self.batch_size, 1)
            q_target = reward + self.gamma * q_next * (1 - done)

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # *** 无 epsilon 衰减 ***
