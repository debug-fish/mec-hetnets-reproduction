import numpy as np

class MECEnvironment:
    """
    WP-MEC HetNets 仿真环境
    对应论文 Section III 的系统模型
    """
    def __init__(self, n_sbs=3, m_wd=5, seed=42):
        self.n_sbs = n_sbs          # SBS 数量
        self.m_wd = m_wd            # WD 数量
        self.rng = np.random.default_rng(seed)

        # ===== 论文 Table II 的参数 =====
        self.mu = 0.8                          # 能量转换效率
        self.P_sbs = 3.0                       # SBS 发射功率 (W)
        self.phi_wd = 5900                     # WD 处理 1 bit 周期数 (cycles/byte 近似)
        self.phi_sbs = 1000                    # SBS 处理 1 bit 周期数 (cycles/bit)
        self.phi_mbs = 1000                    # MBS 处理 1 bit 周期数 (cycles/bit)
        self.k_wd = 1e-28                      # WD 计算能效系数
        self.b_sbs_total = 5e6                 # SBS 总带宽 (Hz)
        self.b_mbs_total = 10e6                # MBS 总带宽 (Hz)
        self.N0 = 2e-13                        # 噪声功率谱密度 (W/Hz)
        self.F_sbs_max = 200e9                 # SBS 最大算力 (cycles/s)
        self.F_mbs_max = 400e9                 # MBS 最大算力 (cycles/s)
        self.a = 0.5                           # WPT 时长占比
        self.T = 1.0                           # 时隙长度

        # 设备到基站的距离（可自行调整）
        self.d_sbs = self.rng.uniform(5, 20, size=(self.m_wd, self.n_sbs))
        self.d_mbs = self.rng.uniform(20, 100, size=self.m_wd)

    # ========== 信道模型 ==========
    def channel_gain(self, d, alpha=2.0, Gamma=0.2):
        small_scale = self.rng.exponential(1.0, size=d.shape)
        return Gamma * (d ** (-alpha)) * small_scale

    # ========== 能量收获（公式 1） ==========
    def harvested_energy(self, h_sbs, sbs_idx):
        """
        h_sbs: (m_wd, n_sbs) 信道增益矩阵
        sbs_idx: (m_wd,) 每个 WD 选中的 SBS 索引
        """
        h_assoc = h_sbs[np.arange(self.m_wd), sbs_idx]
        E = self.mu * self.P_sbs * h_assoc * self.a * self.T
        return E  # (m_wd,)

    # ========== 本地计算速率（公式 3） ==========
    def local_rate(self, E):
        """
        R_local = (1/phi_wd) * (mu * P_sbs * h * a / k_wd)^(1/3)
        注意：公式 3 里 E 已经包含了 mu * P_sbs * h * a，
        这里直接用 E / (k_wd * T) 再开 1/3 次方。
        """
        c_star = (E / (self.k_wd * self.T)) ** (1.0 / 3.0)
        R_local = c_star / self.phi_wd
        return R_local  # (m_wd,)

    # ========== 卸载发射功率（公式 2） ==========
    def offload_power(self, E, x_offload):
        """
        x_offload: (m_wd,) 0/1，表示是否卸载（到 SBS 或 MBS）
        """
        return x_offload * E / ((1 - self.a) * self.T)

    # ========== SBS 卸载速率（公式 4-6） ==========
    def sbs_offload_rate(self, P_wd, h_sbs, sbs_idx, b_sbs):
        """
        P_wd: (m_wd,) 发射功率
        h_sbs: (m_wd, n_sbs) 信道增益
        sbs_idx: (m_wd,) 选中的 SBS
        b_sbs: (m_wd,) 每个 WD 分到的带宽
        返回：每个 WD 卸载到 SBS 的实际速率（取传输速率和计算速率的 min）
        """
        h_assoc = h_sbs[np.arange(self.m_wd), sbs_idx]
        # 传输速率（公式 4）
        sinr = P_wd * h_assoc / (self.N0 * b_sbs + 1e-30)
        R_trans = b_sbs * np.log2(1 + sinr)
        # 计算速率（公式 5），这里假设算力平均分配，后续会换成凸优化结果
        f_sbs = 1.0 / self.m_wd
        R_comp = self.F_sbs_max * f_sbs / self.phi_sbs
        # 取 min（公式 6）
        return np.minimum(R_trans, R_comp)

    def mbs_offload_rate(self, P_wd, h_mbs, b_mbs):
        sinr = P_wd * h_mbs / (self.N0 * b_mbs + 1e-30)
        R_trans = b_mbs * np.log2(1 + sinr)
        f_mbs = 1.0 / self.m_wd
        R_comp = self.F_mbs_max * f_mbs / self.phi_mbs
        return np.minimum(R_trans, R_comp)

    # ========== 总计算速率（公式 10） ==========
    def total_rate(self, x, sbs_idx, b_sbs, b_mbs, h_sbs, h_mbs):
        """
        x: (m_wd, 3) one-hot，每行表示本地/SBS/MBS
        """
        E = self.harvested_energy(h_sbs, sbs_idx)
        R_local = self.local_rate(E)

        x_offload = x[:, 1] + x[:, 2]
        P_wd = self.offload_power(E, x_offload)

        R_sbs = self.sbs_offload_rate(P_wd, h_sbs, sbs_idx, b_sbs)
        R_mbs = self.mbs_offload_rate(P_wd, h_mbs, b_mbs)

        R_total = (
            x[:, 0] * R_local +
            x[:, 1] * R_sbs +
            x[:, 2] * R_mbs
        )
        return R_total.sum(), R_total


# ================= 测试 =================
if __name__ == "__main__":
    env = MECEnvironment(n_sbs=3, m_wd=5, seed=0)

    # 生成信道
    h_sbs = env.channel_gain(env.d_sbs)
    h_mbs = env.channel_gain(env.d_mbs)

    # 随机基站选择
    sbs_idx = env.rng.integers(0, env.n_sbs, size=env.m_wd)

    # 随机卸载决策（one-hot）
    x = np.zeros((env.m_wd, 3))
    choices = env.rng.integers(0, 3, size=env.m_wd)
    x[np.arange(env.m_wd), choices] = 1

    # 平均带宽（先用平均分配验证）
    b_sbs = np.full(env.m_wd, env.b_sbs_total / env.m_wd)
    b_mbs = np.full(env.m_wd, env.b_mbs_total / env.m_wd)

    R_total, R_each = env.total_rate(x, sbs_idx, b_sbs, b_mbs, h_sbs, h_mbs)
    print("每个 WD 的计算速率:", R_each)
    print("系统总计算速率 R_total:", R_total)
    print("SBS 信道增益均值:", h_sbs.mean())
    print("MBS 信道增益均值:", h_mbs.mean())