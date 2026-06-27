"""探測 PiW-3D 處理版資料的真實格式 —— 下載解壓後、動手訓練前務必先跑這支。
(別假設格式,先驗證再動手。)

預期結構:
  <root>/{train_data,test_data}/
      csi/<name>.mat        # MATLAB v7.3,變數 csi_out(複數 real+imag),約 [3Tx,3Rx,30子載波,20時間]
      keypoint/<name>.npy   # [N人,14關節,3D]
      <mode>_data_list.txt  # 每行一個 name(可有可無;沒有就掃 csi/)

用法(piwifi):
  python src/inspect_data.py --root F:/Person-in-WiFi-3D/data/wifipose --split train_data
"""
import argparse
import collections
import glob
import os

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None


def load_csi(p):
    arr = h5py.File(p, "r")["csi_out"][()]
    if arr.dtype.names and "real" in arr.dtype.names:    # 複數以 compound(real,imag)存
        csi = arr["real"] + 1j * arr["imag"]
    else:
        csi = np.asarray(arr)
    return np.asarray(csi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="F:/Person-in-WiFi-3D/data/wifipose")
    ap.add_argument("--split", default="train_data")
    args = ap.parse_args()
    if h5py is None:
        print("需要 h5py:pip install h5py")
        return

    d = os.path.join(args.root, args.split)
    if not os.path.isdir(d):
        print(f"找不到 {d};請確認解壓位置/結構")
        return
    print(f"目錄:{d}\n內容:{sorted(os.listdir(d))}")

    lst = glob.glob(os.path.join(d, "*_data_list.txt"))
    if lst:
        names = [x.strip() for x in open(lst[0]) if x.strip()]
        print(f"list {os.path.basename(lst[0])}:{len(names)} 筆;前 3 = {names[:3]}")

    mats = sorted(glob.glob(os.path.join(d, "csi", "*.mat")))
    kpts = sorted(glob.glob(os.path.join(d, "keypoint", "*.npy")))
    print(f"csi .mat:{len(mats)} 個 | keypoint .npy:{len(kpts)} 個")

    if mats:
        csi = load_csi(mats[0])
        print(f"\n[CSI] 原始 shape={csi.shape} dtype={csi.dtype} 複數={np.iscomplexobj(csi)}")
        print(f"      → 轉成 (3,3,30,20)? 各軸大小 = {csi.shape}")
        amp = np.abs(csi)
        print(f"      amplitude 範圍 [{amp.min():.3g}, {amp.max():.3g}]")

    if kpts:
        kp = np.load(kpts[0])
        print(f"\n[keypoint] shape={kp.shape} dtype={kp.dtype}  (預期 [N人,14,3])")
        flat = kp.reshape(-1, kp.shape[-1])
        for c, ax in enumerate("xyz"[:kp.shape[-1]]):
            print(f"      {ax} 範圍 [{flat[:, c].min():.3f}, {flat[:, c].max():.3f}]")
        mx = float(np.abs(flat[:, :3]).max())
        print(f"      座標量級 ~{mx:.2f} → {'公尺(MPJPE 要 ×1000 成 mm)' if mx < 10 else 'mm 或像素,需確認'}")
        Ns = [np.load(k).shape[0] for k in kpts[:300]]
        print(f"      人數分布(前300):{dict(collections.Counter(Ns))}  ← 看單人(N=1)有多少")


if __name__ == "__main__":
    main()
