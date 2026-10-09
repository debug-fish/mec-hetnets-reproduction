import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PolicyNetwork(nn.Module):
    """
    单设备的策略网络（基站选择 或 卸载决策）
    输入维度 input_dim，输出维度 output_dim
    """
    def __init__(self, input_dim, output_dim, hidden_dims=(64, 32)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        layers.append(nn.Softmax(dim=-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class WD_Agent:
    """
    单个 WD 的 DRL agent，包含两个策略网络：
    - policy_bs: 基站选择网络
    - policy_off: 卸载决策网络
    """
    def __init__(self, n_sbs, lr_bs=1e-3, lr_off=1e-3):
        self.n_sbs = n_sbs
        self.policy_bs = PolicyNetwork(input_dim=n_sbs, output_dim=n_sbs)
        self.policy_off = PolicyNetwork(input_dim=2, output_dim=3)

        self.opt_bs = optim.Adam(self.policy_bs.parameters(), lr=lr_bs)
        self.opt_off = optim.Adam(self.policy_off.parameters(), lr=lr_off)

        # 缓存这一步的 log_prob 和 reward，用于训练
        self.log_prob_bs = None
        self.log_prob_off = None

    def select_bs(self, h_sel):
        """
        h_sel: (n_sbs,) 到每个 SBS 的信道增益
        返回：sbs_idx (int)，选中的 SBS
        """
        h_tensor = torch.tensor(h_sel, dtype=torch.float32)
        probs = self.policy_bs(h_tensor)  # (n_sbs,)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.log_prob_bs = dist.log_prob(action)
        return action.item()

    def select_offload(self, h_off):
        """
        h_off: (2,) = [h_mbs, h_sbs_j]
        返回：offload_action (int)，0=本地, 1=SBS, 2=MBS
        """
        h_tensor = torch.tensor(h_off, dtype=torch.float32)
        probs = self.policy_off(h_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.log_prob_off = dist.log_prob(action)
        return action.item()

    def update(self, reward, bs_coef=1.0, off_coef=1.0):
        """
        用 REINFORCE 更新两个网络
        reward: 标量，本步的奖励（已归一化）
        """
        # 基站选择网络
        if self.log_prob_bs is not None:
            loss_bs = -self.log_prob_bs * reward * bs_coef
            self.opt_bs.zero_grad()
            loss_bs.backward(retain_graph=True)
            self.opt_bs.step()

        # 卸载决策网络
        if self.log_prob_off is not None:
            loss_off = -self.log_prob_off * reward * off_coef
            self.opt_off.zero_grad()
            loss_off.backward()
            self.opt_off.step()


class MultiAgentDRL:
    """
    管理 m 个 WD 的 agent
    """
    def __init__(self, m_wd, n_sbs, lr_bs=1e-3, lr_off=1e-3):
        self.m_wd = m_wd
        self.n_sbs = n_sbs
        self.agents = [
            WD_Agent(n_sbs, lr_bs, lr_off) for _ in range(m_wd)
        ]
        # 奖励归一化：用滑动平均
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0

    def select_actions(self, h_sbs, h_mbs):
        """
        h_sbs: (m_wd, n_sbs)
        h_mbs: (m_wd,)
        返回：
            sbs_idx: (m_wd,)
            offload_actions: (m_wd,) 0/1/2
        """
        sbs_idx = np.zeros(self.m_wd, dtype=int)
        offload_actions = np.zeros(self.m_wd, dtype=int)

        for i, agent in enumerate(self.agents):
            # 基站选择
            j = agent.select_bs(h_sbs[i])
            sbs_idx[i] = j

            # 卸载决策：输入 [h_mbs[i], h_sbs[i, j]]
            h_off = np.array([h_mbs[i], h_sbs[i, j]])
            a = agent.select_offload(h_off)
            offload_actions[i] = a

        return sbs_idx, offload_actions

    def update_all(self, reward):
        """
        用同一个 reward 更新所有 agent
        先对 reward 做归一化，避免梯度爆炸
        """
        # 滑动平均归一化
        self.reward_count += 1
        alpha = 0.01
        self.reward_mean = (1 - alpha) * self.reward_mean + alpha * reward
        self.reward_std = max(self.reward_std, abs(reward - self.reward_mean) + 1e-6)

        # 归一化奖励（除以 reward_mean，让量级接近 1）
        r_norm = reward / (abs(self.reward_mean) + 1e-6)

        for agent in self.agents:
            agent.update(r_norm)