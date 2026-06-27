"""PiW-3D 單人 3D 姿態訓練(path-b)。先建快取再訓練;誠實評估 MPJPE + 平均姿態基線 + 逐關節。

用法(piwifi):
  python src/train.py --root F:/Person-in-WiFi-3D/data/wifipose --stride 3 --epochs 60
  # --unit_scale 1000:GT 公尺→輸出 mm(已由 inspect 確認是公尺)
  # 首次會建快取 _piw3d_cache_s{stride}.npz(較久);之後秒載
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import build_arrays, N_JOINT   # noqa: E402
from model import WifiPose3DNet             # noqa: E402


@torch.no_grad()
def evaluate(model, X, Y, dev, scale):
    model.eval()
    P = []
    for i in range(0, len(X), 512):
        P.append(model(X[i:i + 512].to(dev)).cpu())
    P = torch.cat(P).reshape(-1, N_JOINT, 3)
    G = Y.reshape(-1, N_JOINT, 3)
    mp = (P - G).norm(dim=2)                          # [N,14]
    bp = (G - G.mean(0, keepdim=True)).norm(dim=2)    # 平均姿態基線
    return mp.mean().item() * scale, bp.mean().item() * scale, mp.mean(0) * scale, bp.mean(0) * scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="F:/Person-in-WiFi-3D/data/wifipose")
    ap.add_argument("--stride", type=int, default=3, help="抽樣(連續幀相似,降量省記憶體)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--root_joint", type=int, default=0)
    ap.add_argument("--unit_scale", type=float, default=1000.0)
    ap.add_argument("--out", default="F:/Person-in-WiFi-3D/runs")
    args = ap.parse_args()

    cache = f"F:/Person-in-WiFi-3D/data/_piw3d_cache_s{args.stride}.npz"
    if os.path.exists(cache):
        d = np.load(cache)
        Xtr, Ytr, Xva, Yva = d["Xtr"], d["Ytr"], d["Xva"], d["Yva"]
        print(f"用快取 {cache}", flush=True)
    else:
        print("建快取(讀 .mat,首次較久)...", flush=True)
        Xtr, Ytr, sm1 = build_arrays(args.root, "train_data", args.stride, args.root_joint)
        Xva, Yva, sm2 = build_arrays(args.root, "test_data", args.stride, args.root_joint)
        np.savez(cache, Xtr=Xtr, Ytr=Ytr, Xva=Xva, Yva=Yva)
        print(f"快取已存。跳過多人:train {sm1} / test {sm2}", flush=True)
    print(f"train {Xtr.shape}  test {Xva.shape}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = Xtr.mean((0, 2, 3), keepdims=True), Xtr.std((0, 2, 3), keepdims=True) + 1e-6  # 逐通道
    Xtr -= mu; Xtr /= sd; Xva -= mu; Xva /= sd        # 原地(省記憶體)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
                    batch_size=args.batch, shuffle=True, drop_last=True)
    Xva_t = torch.from_numpy(Xva)
    Yva_t = torch.from_numpy(Yva)

    base = (Yva_t.reshape(-1, N_JOINT, 3) - Yva_t.reshape(-1, N_JOINT, 3).mean(0, keepdim=True)
            ).norm(dim=2).mean().item() * args.unit_scale
    print(f"平均姿態基線 MPJPE = {base:.1f}mm  (模型要顯著低於才算學到)", flush=True)

    model = WifiPose3DNet(in_ch=9, n_joint=N_JOINT).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.MSELoss()
    os.makedirs(args.out, exist_ok=True)
    best = 1e9
    for ep in range(args.epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()
        m, b, mj, bj = evaluate(model, Xva_t, Yva_t, dev, args.unit_scale)
        best = min(best, m)
        flag = "OK 贏基線" if m < b else "✗ 未贏"
        msg = f"ep{ep} MPJPE={m:.1f}mm 基線={b:.1f} [{flag}]"
        if ep % 5 == 0:
            msg += f" | 逐關節贏 {int((mj < bj).sum())}/{N_JOINT}"
        print(msg, flush=True)
        torch.save({"model": model.state_dict(), "epoch": ep}, os.path.join(args.out, "last.pt"))
    print(f"\n最佳 MPJPE={best:.1f}mm vs 基線 {base:.1f}mm → "
          f"{'有信號(贏 %.0f%%)' % (100*(base-best)/base) if best < base*0.97 else '無顯著信號'}",
          flush=True)


if __name__ == "__main__":
    main()
