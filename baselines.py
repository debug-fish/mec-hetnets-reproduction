import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class QLearningAgent:
    """
    简单 Q-Learning：每个 WD 一个 Q 表，状态离散化
    """
    def __init__(self, n_sbs, m_wd, n_states_bins=5, lr=0.1,
                 gamma=0.9, epsilon=0.1):
        self.n_sbs = n_sbs
        self.m_wd = m_wd
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.bins = n_states_bins

        # Q 表：基站选择 (n_bins, n_sbs)，卸载决策 (n_bins, 3)
        self.q_bs = np.zeros((n_states_bins, n_sbs))
        self.q_off = np.zeros((n_states_bins, 3))

        # 用于离散化信道增益的范围（动态更新）
        self.h_min = 1e-10
        self.h_max = 1e-2

    def _discretize(self, h):
        # 对数区间离散化
        h = max(h, self.h_min)
        ratio = np.log10(h / self.h_min) / np.log10(self.h_max / self.h_min)
        return int(np.clip(ratio * (self.bins - 1), 0, self.bins - 1))

    def select_actions(self, h_sbs, h_mbs):
        sbs_idx = np.zeros(self.m_wd, dtype=int)
        offload_actions = np.zeros(self.m_wd, dtype=int)

        for i in range(self.m_wd):
            # 基站选择：用最大信道增益作为状态
            state = self._discretize(h_sbs[i].max())
            if np.random.rand() < self.epsilon:
                j = np.random.randint(self.n_sbs)
            else:
                j = np.argmax(self.q_bs[state])
            sbs_idx[i] = j

            # 卸载决策
            state_off = self._discretize(h_mbs[i])
            if np.random.rand() < self.epsilon:
                a = np.random.randint(3)
            else:
                a = np.argmax(self.q_off[state_off])
            offload_actions[i] = a

        self.last_state_bs = np.array([self._discretize(h_sbs[i].max()) for i in range(self.m_wd)])
        self.last_state_off = np.array([self._discretize(h_mbs[i]) for i in range(self.m_wd)])
        self.last_action_bs = sbs_idx.copy()
        self.last_action_off = offload_actions.copy()

        return sbs_idx, offload_actions

    def update(self, reward, h_sbs_next, h_mbs_next):
        # 简单 Q 更新：用同一 reward 更新所有设备
        r_norm = reward / 1e8
        for i in range(self.m_wd):
            s = self.last_state_bs[i]
            a = self.last_action_bs[i]
            s_next = self._discretize(h_sbs_next[i].max())
            self.q_bs[s, a] += self.lr * (r_norm + self.gamma * self.q_bs[s_next].max() - self.q_bs[s, a])

            s = self.last_state_off[i]
            a = self.last_action_off[i]
            s_next = self._discretize(h_mbs_next[i])
            self.q_off[s, a] += self.lr * (r_norm + self.gamma * self.q_off[s_next].max() - self.q_off[s, a])


class DQNAgent:
    """
    简单 DQN：每个 WD 一个 Q 网络
    """
    def __init__(self, n_sbs, m_wd, lr=1e-3, gamma=0.9,
                 epsilon=0.1, buffer_size=1000, batch_size=32):
        self.n_sbs = n_sbs
        self.m_wd = m_wd
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size

        self.q_bs = nn.Sequential(
            nn.Linear(n_sbs, 64), nn.ReLU(),
            nn.Linear(64, n_sbs)
        )
        self.q_off = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.opt_bs = optim.Adam(self.q_bs.parameters(), lr=lr)
        self.opt_off = optim.Adam(self.q_off.parameters(), lr=lr)

        self.buffer_bs = deque(maxlen=buffer_size)
        self.buffer_off = deque(maxlen=buffer_size)

    def select_actions(self, h_sbs, h_mbs):
        sbs_idx = np.zeros(self.m_wd, dtype=int)
        offload_actions = np.zeros(self.m_wd, dtype=int)

        for i in range(self.m_wd):
            if np.random.rand() < self.epsilon:
                j = np.random.randint(self.n_sbs)
            else:
                with torch.no_grad():
                    q = self.q_bs(torch.tensor(h_sbs[i], dtype=torch.float32))
                    j = q.argmax().item()
            sbs_idx[i] = j

            h_off = np.array([h_mbs[i], h_sbs[i, j]])
            if np.random.rand() < self.epsilon:
                a = np.random.randint(3)
            else:
                with torch.no_grad():
                    q = self.q_off(torch.tensor(h_off, dtype=torch.float32))
                    a = q.argmax().item()
            offload_actions[i] = a

        self.last_h_sbs = h_sbs.copy()
        self.last_h_mbs = h_mbs.copy()
        self.last_sbs_idx = sbs_idx.copy()
        self.last_offload = offload_actions.copy()

        return sbs_idx, offload_actions

    def update(self, reward, h_sbs_next, h_mbs_next):
        r_norm = reward / 1e8
        for i in range(self.m_wd):
            # 基站选择
            s = torch.tensor(self.last_h_sbs[i], dtype=torch.float32)
            a = self.last_sbs_idx[i]
            s_next = torch.tensor(h_sbs_next[i], dtype=torch.float32)
            with torch.no_grad():
                q_next = self.q_bs(s_next).max().item()
            q_pred = self.q_bs(s)[a]
            loss = (q_pred - (r_norm + self.gamma * q_next)) ** 2
            self.opt_bs.zero_grad()
            loss.backward()
            self.opt_bs.step()

            # 卸载决策
            h_off = np.array([self.last_h_mbs[i], self.last_h_sbs[i, self.last_sbs_idx[i]]])
            h_off_next = np.array([h_mbs_next[i], h_sbs_next[i, self.last_sbs_idx[i]]])
            s = torch.tensor(h_off, dtype=torch.float32)
            a = self.last_offload[i]
            s_next = torch.tensor(h_off_next, dtype=torch.float32)
            with torch.no_grad():
                q_next = self.q_off(s_next).max().item()
            q_pred = self.q_off(s)[a]
            loss = (q_pred - (r_norm + self.gamma * q_next)) ** 2
            self.opt_off.zero_grad()
            loss.backward()
            self.opt_off.step()


def train_baseline(baseline_name, seed, n_frames=10000, n_sbs=3, m_wd=5):
    """
    训练一个基线（QL 或 DQN），返回训练日志
    """
    from env import MECEnvironment
    from main import actions_to_onehot

    env = MECEnvironment(n_sbs=n_sbs, m_wd=m_wd, seed=seed)

    if baseline_name == 'QL':
        agent = QLearningAgent(n_sbs, m_wd)
    elif baseline_name == 'DQN':
        agent = DQNAgent(n_sbs, m_wd)
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")

    log = []
    for t in range(n_frames):
        h_sbs = env.channel_gain(env.d_sbs)
        h_mbs = env.channel_gain(env.d_mbs)

        sbs_idx, offload_actions = agent.select_actions(h_sbs, h_mbs)
        x = actions_to_onehot(offload_actions, m_wd)
        R_total, _ = env.total_rate(x, sbs_idx, h_sbs, h_mbs)

        # 生成下一帧信道用于 Q 更新
        h_sbs_next = env.channel_gain(env.d_sbs)
        h_mbs_next = env.channel_gain(env.d_mbs)
        
        agent.update(R_total, h_sbs_next, h_mbs_next)
        log.append(R_total)

    return np.array(log)


if __name__ == "__main__":
    # 单独测试基线
    log_ql = train_baseline('QL', seed=42, n_frames=5000)
    print(f"QL final avg: {np.mean(log_ql[-1000:]):.4e}")

    log_dqn = train_baseline('DQN', seed=42, n_frames=5000)
    print(f"DQN final avg: {np.mean(log_dqn[-1000:]):.4e}")