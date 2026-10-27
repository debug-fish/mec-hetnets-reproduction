import numpy as np
import torch
from env import MECEnvironment
from drl_agent import MultiAgentDRL
from main import actions_to_onehot


def evaluate_drl(seed, n_frames=2000, n_sbs=3, m_wd=5):
    """
    加载训练好的模型，用 argmax 评估
    """
    env = MECEnvironment(n_sbs=n_sbs, m_wd=m_wd, seed=seed + 1000)
    
    drl = MultiAgentDRL(m_wd=m_wd, n_sbs=n_sbs)
    drl.load_all(f"models/seed_{seed}")

    for agent in drl.agents:
        agent.policy_bs.eval()
        agent.policy_off.eval()

    env = MECEnvironment(n_sbs=n_sbs, m_wd=m_wd, seed=seed)

    log = []
    for t in range(n_frames):
        h_sbs = env.channel_gain(env.d_sbs)
        h_mbs = env.channel_gain(env.d_mbs)

        sbs_idx = np.zeros(m_wd, dtype=int)
        offload_actions = np.zeros(m_wd, dtype=int)

        for i, agent in enumerate(drl.agents):
            with torch.no_grad():
                
                probs = agent.policy_bs(torch.tensor(h_sbs[i], dtype=torch.float32))
                j = probs.argmax().item()
                sbs_idx[i] = j

                h_off = np.array([h_mbs[i], h_sbs[i, j]])
                probs_off = agent.policy_off(torch.tensor(h_off, dtype=torch.float32))
                offload_actions[i] = probs_off.argmax().item()

        x = actions_to_onehot(offload_actions, m_wd)
        R_total, _ = env.total_rate(x, sbs_idx, h_sbs, h_mbs)
        log.append(R_total)

    return np.array(log)


if __name__ == "__main__":
    seeds = [42, 43, 44]
    results = []
    for s in seeds:
        log = evaluate_drl(s)
        avg = np.mean(log)
        results.append(avg)
        print(f"DPP-DRL seed={s}: {avg:.4e}")
    print(f"DPP-DRL: {np.mean(results):.4e} ± {np.std(results):.4e}")