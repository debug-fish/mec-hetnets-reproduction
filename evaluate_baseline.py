import numpy as np
from env import MECEnvironment
from main import actions_to_onehot
from baselines import QLearningAgent, DQNAgent


def train_and_evaluate_baseline(name, seed, n_train=10000, n_eval=2000,
                                 n_sbs=3, m_wd=5):
    """
    训练一个基线（QL 或 DQN），然后评估
    """
    env = MECEnvironment(n_sbs=n_sbs, m_wd=m_wd, seed=seed)

    if name == 'QL':
        agent = QLearningAgent(n_sbs, m_wd)
    elif name == 'DQN':
        agent = DQNAgent(n_sbs, m_wd)
    else:
        raise ValueError(f"Unknown baseline: {name}")

    # 训练
    for t in range(n_train):
        h_sbs = env.channel_gain(env.d_sbs)
        h_mbs = env.channel_gain(env.d_mbs)
        sbs_idx, offload_actions = agent.select_actions(h_sbs, h_mbs)
        x = actions_to_onehot(offload_actions, m_wd)
        R_total, _ = env.total_rate(x, sbs_idx, h_sbs, h_mbs)

        h_sbs_next = env.channel_gain(env.d_sbs)
        h_mbs_next = env.channel_gain(env.d_mbs)
        agent.update(R_total, h_sbs_next, h_mbs_next)

    # 评估：关闭探索
    if hasattr(agent, 'epsilon'):
        agent.epsilon = 0.0

    log = []
    for t in range(n_eval):
        h_sbs = env.channel_gain(env.d_sbs)
        h_mbs = env.channel_gain(env.d_mbs)
        sbs_idx, offload_actions = agent.select_actions(h_sbs, h_mbs)
        x = actions_to_onehot(offload_actions, m_wd)
        R_total, _ = env.total_rate(x, sbs_idx, h_sbs, h_mbs)
        log.append(R_total)

    return np.array(log)


if __name__ == "__main__":
    seeds = [42, 43, 44]
    for name in ['QL', 'DQN']:
        results = []
        for s in seeds:
            log = train_and_evaluate_baseline(name, s)
            avg = np.mean(log)
            results.append(avg)
            print(f"{name} seed={s}: {avg:.4e}")
        print(f"{name}: {np.mean(results):.4e} ± {np.std(results):.4e}\n")