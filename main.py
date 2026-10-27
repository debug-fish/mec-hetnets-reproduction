import numpy as np
import os
from env import MECEnvironment
from drl_agent import MultiAgentDRL


def actions_to_onehot(offload_actions, m_wd):
    """
    把 0/1/2 的卸载动作转成 (m_wd, 3) 的 one-hot
    """
    x = np.zeros((m_wd, 3))
    x[np.arange(m_wd), offload_actions] = 1
    return x

def train_one_run(seed, n_frames=10000, n_sbs=3, m_wd=5,
                  lr_bs=1e-3, lr_off=1e-3, print_every=1000):
    """
    单次训练，返回训练日志
    """
    env = MECEnvironment(n_sbs=n_sbs, m_wd=m_wd, seed=seed)
    drl = MultiAgentDRL(m_wd=m_wd, n_sbs=n_sbs, lr_bs=lr_bs, lr_off=lr_off)

    log = []
    for t in range(n_frames):
        
        h_sbs = env.channel_gain(env.d_sbs)
        h_mbs = env.channel_gain(env.d_mbs)

        sbs_idx, offload_actions = drl.select_actions(h_sbs, h_mbs)
        x = actions_to_onehot(offload_actions, m_wd)

        R_total, R_each = env.total_rate(x, sbs_idx, h_sbs, h_mbs)

        rewards_to_drl = R_each.copy()


        drl.update_all_individual(rewards_to_drl)
        
        # 日志记录真实的系统总速率
        log.append(R_total)

        if (t + 1) % print_every == 0:
            recent = np.mean(log[-print_every:])
            print(f"  Frame {t+1}/{n_frames} | R_total={R_total:.4e} | Recent avg={recent:.4e}")

    # 训练完保存模型
    save_dir = f"models/seed_{seed}"
    drl.save_all(save_dir)

    return np.array(log)


def main():
    # ===== 参数 =====
    n_runs = 3
    n_frames = 10000
    all_logs = []

    for run in range(n_runs):
        seed = 42 + run
        print(f"\n=== Run {run+1}/{n_runs} (seed={seed}) ===")
        log = train_one_run(seed, n_frames=n_frames)
        all_logs.append(log)
        np.save(f"training_log_seed_{seed}.npy", log)

    # 保存所有日志
    all_logs = np.array(all_logs)
    np.save("training_logs_all.npy", all_logs)

    # 打印最终结果
    print("\n=== Final Results ===")
    for run in range(n_runs):
        seed = 42 + run
        final_avg = np.mean(all_logs[run][-1000:])
        print(f"Seed {seed}: final avg (last 1000 frames) = {final_avg:.4e}")

    mean_final = np.mean([np.mean(all_logs[r][-1000:]) for r in range(n_runs)])
    std_final = np.std([np.mean(all_logs[r][-1000:]) for r in range(n_runs)])
    print(f"\nOverall: {mean_final:.4e} ± {std_final:.4e}")


if __name__ == "__main__":
    main()