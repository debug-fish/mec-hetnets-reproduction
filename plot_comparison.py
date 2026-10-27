import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch
import os
from env import MECEnvironment
from main import actions_to_onehot
from drl_agent import MultiAgentDRL
from bandwidth_alloc import allocate_bandwidth

# ================= 1. 理论上限（暴力穷举） =================
def get_optimal_frame(h_sbs, h_mbs, env):
    """极速版穷举：只算有意义的动作组合"""
    m_wd = env.m_wd
    max_R = 0
    
    # 1. 只穷举卸载动作 (3^5 = 243种)
    for offload_combo in itertools.product(range(3), repeat=m_wd):
        offload_array = np.array(offload_combo)
        x = actions_to_onehot(offload_array, m_wd)
        
        # 2. 找出真正需要卸载的设备（动作 1 或 2）
        offload_mask = offload_array > 0
        
        # 如果全选本地，基站选啥都一样，直接算一次
        if not np.any(offload_mask):
            R_total, _ = env.total_rate(x, np.zeros(m_wd, dtype=int), h_sbs, h_mbs)
            if R_total > max_R: max_R = R_total
            continue
            
        # 3. 只对卸载的设备穷举基站选择
        # 如果 5 个都卸载，穷举 3^5=243 次；如果只有 2 个卸载，穷举 3^2=9 次，极速！
        num_offload = np.sum(offload_mask)
        for bs_combo in itertools.product(range(env.n_sbs), repeat=num_offload):
            sbs_idx = np.zeros(m_wd, dtype=int)
            sbs_idx[offload_mask] = bs_combo  # 本地设备的 sbs_idx 随意填 0
            
            R_total, _ = env.total_rate(x, sbs_idx, h_sbs, h_mbs)
            if R_total > max_R:
                max_R = R_total
                
    return max_R

# ================= 2. 不同策略的测试 =================
def evaluate_policies(seed, n_frames=200):
    """测试 DRL、穷举、以及三种基线策略"""
    print(f"\n正在测试 Seed={seed} 的表现，共 {n_frames} 帧...")
    
    # 测试环境
    env = MECEnvironment(n_sbs=3, m_wd=5, seed=seed)
    
    # 加载 DRL 模型
    drl = MultiAgentDRL(m_wd=5, n_sbs=3)
    model_path = f"models/seed_{seed}"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型 {model_path}，请先运行 main.py 训练！")
    drl.load_all(model_path)
    for agent in drl.agents:
        agent.policy_bs.eval()
        agent.policy_off.eval()

    results = {'DRL': [], 'Optimal': [], 'All-Local': [], 'All-Offload': [], 'Random': []}

    for t in range(n_frames):
        h_sbs = env.channel_gain(env.d_sbs)
        h_mbs = env.channel_gain(env.d_mbs)

        # 1. 穷举理论最优
        R_opt = get_optimal_frame(h_sbs, h_mbs, env)
        results['Optimal'].append(R_opt)

        # 2. DRL 策略
        sbs_idx_drl = np.zeros(5, dtype=int)
        offload_actions_drl = np.zeros(5, dtype=int)
        for i, agent in enumerate(drl.agents):
            with torch.no_grad():
                probs = agent.policy_bs(torch.tensor(h_sbs[i], dtype=torch.float32))
                j = torch.distributions.Categorical(probs).sample().item()
                sbs_idx_drl[i] = j
                h_off = np.array([h_mbs[i], h_sbs[i, j]])
                probs_off = agent.policy_off(torch.tensor(h_off, dtype=torch.float32))
                offload_actions_drl[i] = torch.distributions.Categorical(probs_off).sample().item()
        x_drl = actions_to_onehot(offload_actions_drl, 5)
        R_drl, _ = env.total_rate(x_drl, sbs_idx_drl, h_sbs, h_mbs)
        results['DRL'].append(R_drl)

        # 3. 全本地
        x_local = actions_to_onehot(np.zeros(5, dtype=int), 5)
        R_local, _ = env.total_rate(x_local, np.zeros(5, dtype=int), h_sbs, h_mbs)
        results['All-Local'].append(R_local)

        # 4. 全卸载到最近的 SBS
        sbs_idx_near = np.argmax(h_sbs, axis=1)
        x_offload = actions_to_onehot(np.ones(5, dtype=int), 5)
        R_offload, _ = env.total_rate(x_offload, sbs_idx_near, h_sbs, h_mbs)
        results['All-Offload'].append(R_offload)

        # 5. 随机策略
        random_actions = np.random.randint(0, 3, size=5)
        random_bs = np.random.randint(0, 3, size=5)
        x_random = actions_to_onehot(random_actions, 5)
        R_random, _ = env.total_rate(x_random, random_bs, h_sbs, h_mbs)
        results['Random'].append(R_random)

    # 计算平均值
    return {k: np.mean(v) for k, v in results.items()}

# ================= 3. 主程序与画图 =================
if __name__ == "__main__":
    seeds = [42, 43, 44]
    all_metrics = {'DRL': [], 'Optimal': [], 'All-Local': [], 'All-Offload': [], 'Random': []}

    for s in seeds:
        metrics = evaluate_policies(s, n_frames=5)
        for k in all_metrics:
            all_metrics[k].append(metrics[k])

    # 取三个种子的平均值
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}

    print("\n" + "="*40)
    print("最终评测结果 (3 个种子平均):")
    for k, v in avg_metrics.items():
        print(f"{k:12s}: {v:.4e} bit/s")
    print(f"DRL 达到理论上限的比例: {(avg_metrics['DRL'] / avg_metrics['Optimal']) * 100:.2f}%")
    print("="*40)

    # ====== 绘制分组柱状图 ======
    labels = ['Optimal (Bound)', 'DRL (Ours)', 'All-Offload', 'All-Local', 'Random']
    values = [avg_metrics['Optimal'], avg_metrics['DRL'], avg_metrics['All-Offload'], avg_metrics['All-Local'], avg_metrics['Random']]
    
    # 颜色设置：理论最优用灰色，我们的模型用亮红色，其他用浅色
    colors = ['#A9A9A9', '#E63946', '#457B9D', '#1D3557', '#F4A261']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors, edgecolor='black', width=0.6)

    # 在柱子上标注具体数值 (格式化为 1e8 的简写)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (max(values) * 0.02), 
                 f'{yval/1e8:.2f}e8', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.ylabel('Average Computation Rate (bit/s)', fontsize=13)
    plt.title('Performance Comparison in Constrained MEC Environment (n=3, m=5)', fontsize=14, fontweight='bold')
    plt.ylim(0, max(values) * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    
    # 保存并展示
    plt.tight_layout()
    plt.savefig("comparison_chart.png", dpi=300)
    plt.show()