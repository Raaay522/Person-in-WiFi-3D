"""PiW-3D 單人 3D 姿態訓練(path-b)。先建快取再訓練;誠實評估 MPJPE + 平均姿態基線 + 逐關節。
訓練時記錄 metrics.csv(loss/MPJPE/PCK),結束畫曲線圖到 runs/(見 plot_metrics.py)。

用法(piwifi):
  python src/train.py --root F:/Person-in-WiFi-3D/data/wifipose_dataset --stride 2 --epochs 60
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
def evaluate(model, X, Y, dev, scale, pck_thr=150.0):
    model.eval()
    P = []
    for i in range(0, len(X), 512):
        P.append(model(X[i:i + 512].to(dev)).cpu())
    P = torch.cat(P).reshape(-1, N_JOINT, 3)
    G = Y.reshape(-1, N_JOINT, 3)
    err = (P - G).norm(dim=2) * scale                          # [N,14] mm
    base_err = (G - G.mean(0, keepdim=True)).norm(dim=2) * scale
    pck = (err < pck_thr).float().mean().item()                # 命中率 @ 門檻
    return (err.mean().item(), base_err.mean().item(),
            err.mean(0), base_err.mean(0), pck, P, G)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="F:/Person-in-WiFi-3D/data/wifipose_dataset")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--root_joint", type=int, default=0)
    ap.add_argument("--unit_scale", type=float, default=1000.0)
    ap.add_argument("--pck_thr", type=float, default=150.0, help="PCK 門檻(mm)")
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
    mu, sd = Xtr.mean((0, 2, 3), keepdims=True), Xtr.std((0, 2, 3), keepdims=True) + 1e-6
    Xtr -= mu; Xtr /= sd; Xva -= mu; Xva /= sd
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
                    batch_size=args.batch, shuffle=True, drop_last=True)
    Xva_t = torch.from_numpy(Xva)
    Yva_t = torch.from_numpy(Yva)

    model = WifiPose3DNet(in_ch=9, n_joint=N_JOINT).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.MSELoss()
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,mpjpe,baseline,pck\n")

    best = 1e9
    for ep in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb); n += len(xb)
        train_loss = tot / n
        m, b, mj, bj, pck, P, G = evaluate(model, Xva_t, Yva_t, dev, args.unit_scale, args.pck_thr)
        best = min(best, m)
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(f"{ep},{train_loss:.6f},{m:.2f},{b:.2f},{pck:.4f}\n")
        flag = "OK 贏基線" if m < b else "✗ 未贏"
        print(f"ep{ep} loss={train_loss:.4f} MPJPE={m:.1f}mm 基線={b:.1f} "
              f"PCK@{args.pck_thr:.0f}={pck:.3f} [{flag}]", flush=True)
        torch.save({"model": model.state_dict(), "epoch": ep}, os.path.join(args.out, "last.pt"))

    # 存最終評估資料(逐關節 + 幾筆 val 樣本)供畫圖
    K = min(8, len(P))
    idx = np.linspace(0, len(P) - 1, K).astype(int)
    np.savez(os.path.join(args.out, "_eval.npz"),
             per_joint_mp=mj.numpy(), per_joint_base=bj.numpy(),
             pred=P[idx].numpy(), gt=G[idx].numpy())
    print(f"\n最佳 MPJPE={best:.1f}mm vs 基線 {b:.1f}mm → "
          f"{'有信號(贏 %.0f%%)' % (100*(b-best)/b) if best < b*0.97 else '無顯著信號'}", flush=True)

    try:
        import plot_metrics
        plot_metrics.make_all(args.out)
        print("已產生曲線圖 → runs/(loss/mpjpe/pck/per_joint/val_overlay)", flush=True)
    except Exception as e:
        print(f"畫圖略過({e})", flush=True)


if __name__ == "__main__":
    main()
