import numpy as np


def solve_b_for_lambda(A, lam, b_init=None, tol=1e-10, max_iter=100):
    """
    给定 lambda，用 Newton 法解 b：
        ln(1 + A/b) - A/(b+A) = lambda * ln2

    参数：
        A: 标量，= P_wd * h / N0
        lam: 标量，拉格朗日乘子
        b_init: 初始猜测（默认用 A 作为初值）
    返回：
        b: 标量，最优带宽
    """
    if b_init is None:
        b_init = A

    b = b_init
    target = lam * np.log(2)

    for _ in range(max_iter):
        # f(b) = ln(1 + A/b) - A/(b+A) - target
        f = np.log(1 + A / b) - A / (b + A) - target
        # f'(b) = -A/(b(b+A)) + A/(b+A)^2
        fp = -A / (b * (b + A)) + A / (b + A) ** 2

        if abs(fp) < 1e-30:
            break

        b_new = b - f / fp
        if b_new <= 0:
            b_new = b / 2

        if abs(b_new - b) < tol:
            b = b_new
            break
        b = b_new

    return max(b, 1e-12)


def allocate_bandwidth(A_vec, B_total, tol=1e-6, max_bisect=100):
    """
    论文 Algorithm 1：二分法找 lambda，Newton 法解 b
    最大化 sum_i b_i * log2(1 + A_i / b_i)，约束 sum_i b_i = B_total

    参数：
        A_vec: (m,) 每个设备的 A_i = P_wd * h / N0
        B_total: 标量，总带宽 (Hz)
    返回：
        b_vec: (m,) 每个设备分到的带宽
    """
    m = len(A_vec)
    A_vec = np.asarray(A_vec, dtype=float)

    # 初始化二分区间
    lam_low = 0.0
    lam_high = np.max(np.log2(1 + A_vec / 1e-6))
    lam_high = max(lam_high, 1.0)

    # 确保 lam_high 能让 sum(b) < B_total
    while True:
        b_vec = np.array([solve_b_for_lambda(A_vec[i], lam_high) for i in range(m)])
        if b_vec.sum() < B_total:
            break
        lam_high *= 2
        if lam_high > 1e20:
            break

    # 二分法
    for _ in range(max_bisect):
        if lam_high - lam_low < tol:
            break
        lam_mid = (lam_low + lam_high) / 2
        b_vec = np.array([solve_b_for_lambda(A_vec[i], lam_mid) for i in range(m)])
        B_sum = b_vec.sum()

        if B_sum > B_total:
            lam_low = lam_mid
        else:
            lam_high = lam_mid

    # 最后算一次并归一化
    lam_final = (lam_low + lam_high) / 2
    b_vec = np.array([solve_b_for_lambda(A_vec[i], lam_final) for i in range(m)])

    if b_vec.sum() > 0:
        b_vec = b_vec * (B_total / b_vec.sum())

    return b_vec


# ================= 单独测试 =================
if __name__ == "__main__":
    np.random.seed(0)
    m = 5
    A_vec = np.random.uniform(1e5, 1e7, size=m)
    B_total = 5e6  # 5 MHz

    b_vec = allocate_bandwidth(A_vec, B_total)

    print("A_vec:", A_vec)
    print("分配带宽:", b_vec)
    print("带宽总和:", b_vec.sum(), "目标:", B_total)
    print("各设备速率:", b_vec * np.log2(1 + A_vec / b_vec))
    print("凸优化总速率:", (b_vec * np.log2(1 + A_vec / b_vec)).sum())

    b_avg = np.full(m, B_total / m)
    rate_avg = (b_avg * np.log2(1 + A_vec / b_avg)).sum()
    print("平均分配总速率:", rate_avg)
    print("提升比例:", (b_vec * np.log2(1 + A_vec / b_vec)).sum() / rate_avg)