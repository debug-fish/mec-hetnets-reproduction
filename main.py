import numpy as np
from env import MECEnvironment
from drl_agent import MultiAgentDRL


def actions_to_onehot(offload_actions, m_wd):
    """
    把 0/1/2 的卸载动作转成 (m_wd, 3) 的 one-hot
    """
    x = np.zeros((m_wd, 3))
    x[np.arange(m_wd), offload_actions] = 1
    return x


def main():
    # ===== 参数 =====
    n_sbs = 3
    m_wd = 5
    n_frames = 5000       # 先跑 5000 帧，验证训练能收敛
    print_every = 500

    env = MECEnvironment(n_sbs=n_sbs, m_wd=m_wd, seed=42)
    drl = MultiAgentDRL(m_wd=m_wd, n_sbs=n_sbs, lr_bs=1e-3, lr_off=1e-3)

    log = []

    for t in range(n_frames):
        # 1. 生成信道
        h_sbs = env.channel_gain(env.d_sbs)
        h_mbs = env.channel_gain(env.d_mbs)

        # 2. DRL 选动作
        sbs_idx, offload_actions = drl.select_actions(h_sbs, h_mbs)

        # 3. 转成 one-hot
        x = actions_to_onehot(offload_actions, m_wd)

        # 4. 计算 R_total
        R_total, R_each = env.total_rate(x, sbs_idx, h_sbs, h_mbs)

        # 5. 更新 DRL（奖励 = R_total）
        drl.update_all(R_total)

        log.append(R_total)

        # 6. 打印进度
        if (t + 1) % print_every == 0:
            recent = np.mean(log[-print_every:])
            print(f"Frame {t+1}/{n_frames} | "
                  f"R_total={R_total:.4e} | "
                  f"Recent avg={recent:.4e} | "
                  f"Normalized={R_total / (abs(drl.reward_mean) + 1e-6):.3f}")

    # 保存日志
    np.save("training_log.npy", np.array(log))
    print("Training done. Log saved to training_log.npy")


if __name__ == "__main__":
    main()