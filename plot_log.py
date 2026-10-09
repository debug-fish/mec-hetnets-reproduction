import numpy as np
import matplotlib.pyplot as plt

all_logs = np.load("training_logs_all.npy")
n_runs = all_logs.shape[0]
window = 200

plt.figure(figsize=(10, 5))
for run in range(n_runs):
    log = all_logs[run]
    ma = np.convolve(log, np.ones(window)/window, mode='valid')
    plt.plot(ma, label=f'Run {run+1} (seed={42+run})')

plt.xlabel("Frame")
plt.ylabel("R_total (moving avg)")
plt.title("DRL Training Progress (3 runs)")
plt.legend()
plt.grid(True)
plt.savefig("training_curves_multi.png", dpi=150)
plt.show()

# 打印最终平均值
finals = [np.mean(all_logs[r][-1000:]) for r in range(n_runs)]
print(f"Final avg: {np.mean(finals):.4e} ± {np.std(finals):.4e}")