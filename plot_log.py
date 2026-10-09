import numpy as np
import matplotlib.pyplot as plt

log = np.load("training_log.npy")
# 移动平均
window = 100
ma = np.convolve(log, np.ones(window)/window, mode='valid')

plt.figure(figsize=(10, 5))
plt.plot(log, alpha=0.3, label='Raw')
plt.plot(ma, label=f'Moving avg ({window})')
plt.xlabel("Frame")
plt.ylabel("R_total")
plt.title("DRL Training Progress")
plt.legend()
plt.grid(True)
plt.savefig("training_curve.png", dpi=150)
plt.show()