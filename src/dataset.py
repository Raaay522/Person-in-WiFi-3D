"""PiW-3D 資料讀取(path-b)。實測格式:
  csi/<name>.mat  : h5py 變數 csi_out 複數,原始 [20時間,30子載波,3Tx,3Rx] → transpose 成 [3,3,30,20]
  keypoint/<name>.npy : [1人,14,3](單人,公尺)

為效率,build_arrays 一次把(單人)樣本載成 numpy 陣列(供 train.py 快取)。
"""
import glob
import os

import h5py
import numpy as np

N_JOINT = 14


def load_csi_amp(path):
    """csi_out 複數 → 振幅 [3,3,30,20] float32。"""
    arr = h5py.File(path, "r")["csi_out"][()]
    if arr.dtype.names and "real" in arr.dtype.names:
        csi = arr["real"] + 1j * arr["imag"]
    else:
        csi = np.asarray(arr)
    csi = np.asarray(csi)
    if csi.shape != (3, 3, 30, 20) and csi.ndim == 4:
        csi = csi.transpose(3, 2, 1, 0)            # (20,30,3,3) → (3,3,30,20)
    return np.abs(csi).astype(np.float32)


def list_names(d):
    lst = glob.glob(os.path.join(d, "*_data_list.txt"))
    if lst:
        return [x.strip() for x in open(lst[0]) if x.strip()]
    return [os.path.splitext(os.path.basename(m))[0]
            for m in sorted(glob.glob(os.path.join(d, "csi", "*.mat")))]


def build_arrays(root, split, stride=1, root_joint=0):
    """載入(單人)樣本 → X[N,9,30,20], Y[N,42](root-relative)。回傳 (X, Y, n_skip_multi)。"""
    d = os.path.join(root, split)
    names = list_names(d)[::stride]
    X, Y, skip_multi = [], [], 0
    for i, n in enumerate(names):
        csf = os.path.join(d, "csi", n + ".mat")
        kpf = os.path.join(d, "keypoint", n + ".npy")
        if not (os.path.exists(csf) and os.path.exists(kpf)):
            continue
        kp = np.load(kpf).astype(np.float32)
        if kp.shape[0] != 1:                       # 只取單人
            skip_multi += 1
            continue
        kp = kp[0]
        kp = kp - kp[root_joint:root_joint + 1]    # root-relative
        X.append(load_csi_amp(csf).reshape(9, 30, 20))
        Y.append(kp.reshape(-1))
        if i % 5000 == 0:
            print(f"  {split}: {i}/{len(names)} (跳過多人 {skip_multi})", flush=True)
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), skip_multi
