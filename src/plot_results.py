"""成果圖:Person-in-WiFi-3D 學習曲線(MPJPE 隨 epoch 下降 → 逼近官方 91.7mm)。
  python src/plot_results.py   # → runs/piw3d_results.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 每 epoch 驗證 MPJPE(mm);stride 越小用越多資料
s5 = [212.4, 148.9, 145.0, 135.0, 134.8, 127.8, 129.3, 142.3, 126.5, 119.8, 125.2, 117.0, 118.3,
      118.1, 117.8, 132.7, 114.2, 111.7, 115.9, 125.6, 113.6, 110.9, 112.1, 109.5, 109.3, 109.5,
      106.7, 109.6, 110.3, 111.7, 110.8, 115.8, 106.2, 106.8, 105.1, 107.4, 104.2, 113.5, 102.4,
      115.0, 106.2, 113.3, 113.1, 106.0, 101.5, 108.9, 101.3, 104.4, 107.3, 102.1, 101.7, 107.9,
      100.4, 101.8, 102.9, 103.0, 107.2, 100.0, 100.4, 111.2]
s2 = [154.3, 141.4, 125.8, 126.0, 119.2, 119.0, 113.6, 112.6, 111.8, 118.1, 118.8, 116.2, 108.0,
      112.2, 109.1, 106.7, 108.5, 101.7, 102.6, 99.8, 102.9, 110.0, 99.7, 102.6, 100.5, 101.8,
      97.7, 98.0, 97.6, 98.4, 100.1, 96.9, 99.8, 103.1, 99.1, 98.4, 103.5, 96.3, 97.9, 97.0,
      101.1, 101.3, 97.7, 101.1, 96.1, 99.3, 99.6, 97.3, 96.4, 96.3, 94.4, 100.3, 93.4, 94.8,
      93.0, 95.9, 95.9, 92.6, 94.4, 95.5]
BASE, OFFICIAL = 143.7, 91.7

fig, ax = plt.subplots(figsize=(8.5, 5.2))
ep = range(1, len(s2))                                   # 跳過 ep0(未訓練)
ax.plot(ep, s2[1:], "-", color="tab:blue", label="stride 2 (14k samples)")
ax.plot(ep, s5[1:], "-", color="tab:orange", alpha=.85, label="stride 5 (5.6k samples)")
ax.axhline(BASE, ls="--", color="black", label=f"mean-pose baseline ({BASE:.0f} mm)")
ax.axhline(OFFICIAL, ls="--", color="tab:green", label=f"official PETR ({OFFICIAL:.1f} mm)")
ax.annotate("best 92.6", (57, 92.6), color="tab:blue", fontsize=9)
ax.set_xlabel("epoch")
ax.set_ylabel("MPJPE (mm, 3D)")
ax.set_ylim(85, 160)
ax.set_title("Person-in-WiFi-3D: WiFi → 3D pose learning curve\n"
             "(simple CNN, single-person; best 92.6mm ≈ official 91.7mm)")
ax.legend()
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig("runs/piw3d_results.png", dpi=110)
print("saved runs/piw3d_results.png")
