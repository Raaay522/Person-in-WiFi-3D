"""讀 runs/metrics.csv + runs/_eval.npz → 畫 5 張圖到 runs/。
  train.py 結束會自動呼叫;也可單獨跑:python src/plot_metrics.py
產出:loss.png、mpjpe.png、pck.png、per_joint.png、val_overlay.png
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(out):
    cols = {"epoch": [], "train_loss": [], "mpjpe": [], "baseline": [], "pck": []}
    with open(os.path.join(out, "metrics.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for k in cols:
                cols[k].append(float(row[k]))
    return {k: np.array(v) for k, v in cols.items()}


def mst_edges(pts):
    """用平均姿態 14 點建最小生成樹當骨架(免知道官方關節順序)。"""
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import cdist
    T = minimum_spanning_tree(cdist(pts, pts)).toarray()
    return [(i, j) for i in range(len(pts)) for j in range(len(pts)) if T[i, j] > 0]


def make_all(out):
    M = load_metrics(out)
    ep = M["epoch"]

    # 1) Loss
    plt.figure(figsize=(7, 4.2))
    plt.plot(ep, M["train_loss"], color="tab:purple")
    plt.xlabel("epoch"); plt.ylabel("train MSE loss"); plt.title("Training loss")
    plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(out, "loss.png"), dpi=110); plt.close()

    # 2) MPJPE vs 平均姿態基線
    mp, base = M["mpjpe"], M["baseline"]
    plt.figure(figsize=(7, 4.2))
    plt.plot(ep, mp, color="tab:blue", label="model MPJPE")
    plt.axhline(base[-1], ls="--", color="black", label=f"mean-pose baseline ({base[-1]:.0f} mm)")
    plt.annotate(f"best {mp.min():.1f}", (ep[mp.argmin()], mp.min()), fontsize=9, color="tab:blue")
    plt.xlabel("epoch"); plt.ylabel("MPJPE (mm)"); plt.title("Validation MPJPE")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(out, "mpjpe.png"), dpi=110); plt.close()

    # 3) PCK
    plt.figure(figsize=(7, 4.2))
    plt.plot(ep, M["pck"], color="tab:green")
    plt.xlabel("epoch"); plt.ylabel("PCK @150mm"); plt.ylim(0, 1)
    plt.title("Validation PCK @150mm (joints within 150mm)")
    plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(out, "pck.png"), dpi=110); plt.close()

    ev = os.path.join(out, "_eval.npz")
    if not os.path.exists(ev):
        print("無 _eval.npz,跳過 per_joint / val_overlay")
        return
    d = np.load(ev)
    pj, pjb, pred, gt = d["per_joint_mp"], d["per_joint_base"], d["pred"], d["gt"]

    # 4) 逐關節 MPJPE(模型 vs 基線)
    x = np.arange(len(pj)); w = 0.4
    plt.figure(figsize=(9, 4.2))
    plt.bar(x - w / 2, pj, w, label="model", color="tab:blue")
    plt.bar(x + w / 2, pjb, w, label="mean-pose baseline", color="gray")
    plt.xlabel("joint index (0..13)"); plt.ylabel("MPJPE (mm)")
    plt.title("Per-joint MPJPE: model vs baseline")
    plt.xticks(x); plt.legend(); plt.grid(axis="y", alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(out, "per_joint.png"), dpi=110); plt.close()

    # 5) val 重疊圖:GT(綠)vs Pred(紅),取兩個展幅最大的軸當平面 + MST 骨架
    mean_pose = gt.mean(0)                                     # [14,3] 平均姿態(供 MST 骨架)
    try:
        edges = mst_edges(mean_pose)
    except Exception:
        edges = []
    spread = gt.reshape(-1, 3).std(0)
    order = np.argsort(spread)[::-1]
    ax_v, ax_h = int(order[0]), int(order[1])                 # 最寬→垂直,次寬→水平
    K = len(pred)
    cols = 4
    rows = (K + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.6 * rows))
    axes = np.array(axes).ravel()
    for k, axp in enumerate(axes):
        if k >= K:
            axp.axis("off"); continue
        g, p = gt[k], pred[k]
        for i, j in edges:
            axp.plot([g[i, ax_h], g[j, ax_h]], [g[i, ax_v], g[j, ax_v]], "-", color="tab:green", lw=1, alpha=.5)
            axp.plot([p[i, ax_h], p[j, ax_h]], [p[i, ax_v], p[j, ax_v]], "-", color="tab:red", lw=1, alpha=.5)
        axp.scatter(g[:, ax_h], g[:, ax_v], c="tab:green", s=22, label="GT")
        axp.scatter(p[:, ax_h], p[:, ax_v], c="tab:red", s=22, marker="x", label="Pred")
        axp.set_title(f"val sample {k}", fontsize=9); axp.set_aspect("equal"); axp.grid(alpha=.2)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Validation overlay — GT (green) vs Pred (red)  |  axes {ax_h}(x)-{ax_v}(y)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out, "val_overlay.png"), dpi=100); plt.close()

    print("saved: loss.png mpjpe.png pck.png per_joint.png val_overlay.png")


if __name__ == "__main__":
    make_all(os.environ.get("RUNS", "F:/Person-in-WiFi-3D/runs"))
