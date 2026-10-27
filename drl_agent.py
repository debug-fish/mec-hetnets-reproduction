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

        # [MOD] 引入温度系数：在 MultiAgentDRL 中传入，或者简单点这里写死
        # 训练初期可以大一点探索，后期变小（比如设为 0.5）
        temperature = getattr(self, 'temperature', 1.0) 
        
        # 重新计算分布
        dist = torch.distributions.Categorical(logits=torch.log(probs + 1e-8) / temperature)
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

        temperature = getattr(self, 'temperature', 1.0)
        dist = torch.distributions.Categorical(logits=torch.log(probs + 1e-8) / temperature)
        action = dist.sample()
        self.log_prob_off = dist.log_prob(action)
        return action.item()

    def update(self, reward):
        """
        用 REINFORCE 更新两个网络
        reward: 标量，本步的奖励（已归一化）
        """
        # 基站选择网络
        if self.log_prob_bs is not None:
            loss_bs = -self.log_prob_bs * reward
            self.opt_bs.zero_grad()
            loss_bs.backward(retain_graph=True)
            self.opt_bs.step()

        # 卸载决策网络
        if self.log_prob_off is not None:
            loss_off = -self.log_prob_off * reward
            self.opt_off.zero_grad()
            loss_off.backward()
            self.opt_off.step()

    def save(self, path):
        #模型保存
        torch.save({
            'policy_bs': self.policy_bs.state_dict(),
            'policy_off': self.policy_off.state_dict(),
        }, path)

    def load(self, path):
        #模型加载
        ckpt = torch.load(path)
        self.policy_bs.load_state_dict(ckpt['policy_bs'])
        self.policy_off.load_state_dict(ckpt['policy_off'])


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
        self.reward_count = 0

        # [MOD] 新增：每个设备的独立奖励基线
        self.reward_means_individual = np.zeros(m_wd)
        self.reward_counts_individual = np.zeros(m_wd)

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
        用 Advantage (优势函数) 代替绝对奖励，抑制单步更新带来的剧烈震荡
        """
        self.reward_count += 1
        alpha = 0.01
        # 第一次直接用当前奖励初始化均值
        if self.reward_count == 1:
            self.reward_mean = reward
        else:
            self.reward_mean = (1 - alpha) * self.reward_mean + alpha * reward

        # 【核心修改】使用 r - baseline。如果比平均好，给正梯度；比平均差，给负梯度
        advantage = reward - self.reward_mean
        
        # 固定缩放，让梯度的量级保持在合理范围内（可根据实际 R_total 微调 1e8 这个分母）
        r_norm = advantage / 1e8
        
        # 可选：引入一个微小的放大系数，加速收敛
        r_norm *= 10.0 

        for agent in self.agents:
            agent.update(r_norm)

    def update_all_individual(self, rewards_each):
        """
        [MOD] 独立更新：每个 agent 根据自己计算的 R_each 更新自己的网络
        rewards_each: (m_wd,) 的数组，对应 env.total_rate 返回的 R_each
        """
        alpha = 0.05
        
        # [MOD] 先统一算好这一步的温度系数
        # reward_counts_individual[0] 代表第一个设备已经更新的步数
        current_temp = max(0.1, 1.0 - (self.reward_counts_individual[0] / 5000.0))
        
        for i, agent in enumerate(self.agents):
            reward = float(rewards_each[i])  # 转成浮点数，防止 numpy 广播
            
            self.reward_counts_individual[i] += 1
            
            # 初始化或更新当前 agent 的滑动平均基线
            if self.reward_counts_individual[i] == 1:
                self.reward_means_individual[i] = reward
            else:
                self.reward_means_individual[i] = (1 - alpha) * self.reward_means_individual[i] + alpha * reward
            
            # 计算优势（当前奖励 - 历史平均水平）
            advantage = reward - self.reward_means_individual[i]
            
            # 动态归一
            scale = abs(self.reward_means_individual[i]) + 1e-6
            r_norm = advantage / scale
            
            # [MOD] 设置当前 agent 的温度，并更新它的网络
            agent.temperature = current_temp
            agent.update(r_norm) 

    def save_all(self, dir_path):
        import os
        os.makedirs(dir_path, exist_ok=True)
        for i, agent in enumerate(self.agents):
            agent.save(f"{dir_path}/agent_{i}.pt")

    def load_all(self, dir_path):
        for i, agent in enumerate(self.agents):
            agent.load(f"{dir_path}/agent_{i}.pt")