import numpy as np
from bandwidth_alloc import allocate_bandwidth

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
        self.phi_wd = 5900                     # [KEEP] 论文 Table II 原值 cycles/byte；式(3) 理论用 cycles/bit
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
        self.d_sbs = self.rng.uniform(2, 10, size=(self.m_wd, self.n_sbs))
        self.d_mbs = self.rng.uniform(10, 30, size=self.m_wd)

        self.h_sbs_base = self.channel_gain(self.d_sbs)
        self.h_mbs_base = self.channel_gain(self.d_mbs)

    # ========== 信道模型 ==========
    def channel_gain(self, d, alpha=2.0, Gamma=0.3):
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

    def sbs_offload_rate(self, P_wd, h_sbs, sbs_idx, b_sbs_total,x_mask):
        """
        每个 SBS 独立分配自己的带宽，用凸优化
        """
        m = self.m_wd
        R_trans = np.zeros(m)
        R_comp = np.zeros(m)                               # [MOD-2] 单独存计算速率

        for j in range(self.n_sbs):
            # [MOD-2] 同时排除「未卸载」和「卸载到 MBS」的用户
            mask = (sbs_idx == j) & (x_mask > 0)
            if mask.sum() == 0:
                continue

            A_vec = P_wd[mask] * h_sbs[mask, j] / self.N0
            A_vec = np.maximum(A_vec, 1e-12)
            b_vec = allocate_bandwidth(A_vec, b_sbs_total)
            R_trans[mask] = b_vec * np.log2(1 + A_vec / b_vec)

            # [MOD-2] 算力只在「该 SBS 的卸载子集」上分配并归一（C4: sum f = 1）
            R_sub = R_trans[mask]
            f_sub = self.allocate_compute(R_sub, self.F_sbs_max, self.phi_sbs)
            R_comp[mask] = self.F_sbs_max * f_sub / self.phi_sbs

        return np.minimum(R_trans, R_comp)

    def mbs_offload_rate(self, P_wd, h_mbs, b_mbs_total, x_mask):
        """
        MBS 独立分配带宽，只服务卸载到 MBS 的设备
        """
        m = self.m_wd
        R_trans = np.zeros(m)
        R_comp = np.zeros(m)                               # [MOD-3]

        # [MOD-3] 显式用卸载决策 mask，不再依赖 P_wd > 0
        mask = x_mask > 0
        if mask.sum() > 0:
            A_vec = P_wd[mask] * h_mbs[mask] / self.N0
            A_vec = np.maximum(A_vec, 1e-12)
            b_vec = allocate_bandwidth(A_vec, b_mbs_total)
            R_trans[mask] = b_vec * np.log2(1 + A_vec / b_vec)

            # [MOD-3] 算力只在 MBS 卸载子集上分配并归一（C4: sum f = 1）
            R_sub = R_trans[mask]
            f_sub = self.allocate_compute(R_sub, self.F_mbs_max, self.phi_mbs)
            R_comp[mask] = self.F_mbs_max * f_sub / self.phi_mbs

        return np.minimum(R_trans, R_comp)
    
    # ========== 总计算速率（公式 10） ==========
    def total_rate(self, x, sbs_idx, h_sbs, h_mbs):
        """
        x: (m_wd, 3) one-hot，每行表示本地/SBS/MBS
        """
        E = self.harvested_energy(h_sbs, sbs_idx)
        R_local = self.local_rate(E)

        x_sbs = x[:, 1]          # 卸载到 SBS 的用户
        x_mbs = x[:, 2]          # 卸载到 MBS 的用户
        P_sbs = self.offload_power(E, x_sbs)
        P_mbs = self.offload_power(E, x_mbs)

        R_sbs = self.sbs_offload_rate(P_sbs, h_sbs, sbs_idx, self.b_sbs_total, x_sbs)
        R_mbs = self.mbs_offload_rate(P_mbs, h_mbs, self.b_mbs_total, x_mbs)

        R_total = (
            x[:, 0] * R_local +
            x[:, 1] * R_sbs +
            x[:, 2] * R_mbs
        )
        return R_total.sum(), R_total

    def allocate_compute(self, R_offload, F_max, phi):
        """
        论文 Section IV-B2 算力分配：
        情况 1（算力不够）：按卸载速率从高到低优先分配
        情况 2（算力够用）：按卸载速率比例分配
        """
        m = len(R_offload)
        f = np.zeros(m)

        total_offload_rate = R_offload.sum()
        max_supported_rate = F_max / phi

        if total_offload_rate > max_supported_rate:
            # 情况 1：算力不够
            sorted_idx = np.argsort(R_offload)[::-1]
            remaining_rate = max_supported_rate
            for idx in sorted_idx:
                if remaining_rate <= 0:
                    break
                allocated = min(R_offload[idx], remaining_rate)
                f[idx] = allocated * phi / F_max
                remaining_rate -= allocated
        else:
            # 情况 2：算力够用，按比例分配
            if total_offload_rate > 0:
                f = R_offload / total_offload_rate

        return f


# ================= 测试 =================
if __name__ == "__main__":
    env = MECEnvironment(n_sbs=3, m_wd=5, seed=0)
    h_sbs = env.channel_gain(env.d_sbs)
    h_mbs = env.channel_gain(env.d_mbs)
    sbs_idx = env.rng.integers(0, env.n_sbs, size=env.m_wd)
    x = np.zeros((env.m_wd, 3))
    choices = env.rng.integers(0, 3, size=env.m_wd)
    x[np.arange(env.m_wd), choices] = 1

    E = env.harvested_energy(h_sbs, sbs_idx)
    R_local = env.local_rate(E)

    x_sbs, x_mbs = x[:, 1], x[:, 2]
    P_sbs = env.offload_power(E, x_sbs)
    P_mbs = env.offload_power(E, x_mbs)
    R_sbs = env.sbs_offload_rate(P_sbs, h_sbs, sbs_idx, env.b_sbs_total, x_sbs)
    R_mbs = env.mbs_offload_rate(P_mbs, h_mbs, env.b_mbs_total, x_mbs)
    R_total, R_each = env.total_rate(x, sbs_idx, h_sbs, h_mbs)

    print("卸载决策 (0=本地,1=SBS,2=MBS):", choices)
    print("基站选择:", sbs_idx)
    print("E (J):", E)
    print("R_local (bit/s):", R_local)
    print("R_sbs   (bit/s):", R_sbs)
    print("R_mbs   (bit/s):", R_mbs)
    print("R_each  (bit/s):", R_each)
    print("R_total (bit/s):", R_total)

    # ===== C4 验证 =====
    print("\n===== C4 验证（每个 SBS / MBS 的 f 之和应=1）=====")
    for j in range(env.n_sbs):
        mask = (sbs_idx == j) & (x_sbs > 0)
        if mask.sum() == 0:
            print(f"SBS{j}: 无卸载用户"); continue
        R_sub = R_sbs[mask]                      # 注意：这里 R_sbs 已是 min 后结果
        # 用原始 R_trans 更准，但这里近似验证
        A_vec = P_sbs[mask] * h_sbs[mask, j] / env.N0
        from bandwidth_alloc import allocate_bandwidth
        b_vec = allocate_bandwidth(np.maximum(A_vec,1e-12), env.b_sbs_total)
        R_trans_sub = b_vec * np.log2(1 + A_vec / b_vec)
        f_sub = env.allocate_compute(R_trans_sub, env.F_sbs_max, env.phi_sbs)
        print(f"SBS{j}: 卸载用户数={mask.sum()}, sum(f)={f_sub.sum():.6f}")

    if x_mbs.sum() > 0:
        A_vec = P_mbs[x_mbs>0] * h_mbs[x_mbs>0] / env.N0
        from bandwidth_alloc import allocate_bandwidth
        b_vec = allocate_bandwidth(np.maximum(A_vec,1e-12), env.b_mbs_total)
        R_trans_sub = b_vec * np.log2(1 + A_vec / b_vec)
        f_mbs = env.allocate_compute(R_trans_sub, env.F_mbs_max, env.phi_mbs)
        print(f"MBS: 卸载用户数={x_mbs.sum()}, sum(f)={f_mbs.sum():.6f}")
    else:
        print("MBS: 无卸载用户")