# Person-in-WiFi-3D

用 WiFi CSI 估計單人 3D 人體姿態(純 PyTorch / Windows,不依賴官方 OPERA/mmcv)。

## 功能

- **輸入**:WiFi CSI(Intel 5300,複數 `[3Tx, 3Rx, 30子載波, 20時間]`)。
- **輸出**:單人 3D 姿態,14 關節 × 3D 座標。
- **模型**:簡單 CNN(`[9,30,20] → 14×3`,1.63M 參數),PyTorch 2.5。
- **評估**:MPJPE(mm)+「平均姿態基線」對照 + 逐關節,訓練時即時印出。
- **成果**:best MPJPE **92.6mm**(≈ 官方 PETR 91.7mm)。

## 專案結構

```
Person-in-WiFi-3D/
├─ src/
│  ├─ inspect_data.py   驗證資料格式(換資料先跑這支)
│  ├─ dataset.py        讀 CSI 複數→振幅 [9,30,20];單人 [14,3] root-relative;建快取
│  ├─ model.py          CNN [9,30,20] → 14×3
│  ├─ train.py          建快取 → 訓練 + MPJPE + 平均姿態基線 + 逐關節
│  └─ plot_results.py   成果圖
├─ data/                          (不入庫)
│  ├─ wifipose_dataset/
│  │  ├─ train_data/{csi/*.mat, keypoint/*.npy, train_data_list.txt}   (89946)
│  │  └─ test_data /{csi/*.mat, keypoint/*.npy, test_data_list.txt}    (7824)
│  └─ _piw3d_cache_s{2,5}.npz     訓練快取
├─ runs/
│  ├─ last.pt                     訓練權重(不入庫)
│  └─ piw3d_results.png           成果圖
├─ repo/                          官方 OPERA repo(參考用,不入庫)
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## 安裝指令

```bash
conda create -n piwifi python=3.10 -y
conda activate piwifi

# PyTorch(CUDA 12.1;CPU 版把 --index-url 整段拿掉)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 其餘依賴
pip install -r requirements.txt
```

## 執行指令

```bash
conda activate piwifi
cd F:\Person-in-WiFi-3D

# 1) 驗證資料格式(換資料必跑:印 CSI/keypoint shape、座標單位、人數分布)
python src/inspect_data.py --root data/wifipose_dataset

# 2) 訓練(首次建快取較久;--stride 越小用越多資料,越準)
python src/train.py --root data/wifipose_dataset --stride 2 --epochs 60

# 3) 畫成果圖
python src/plot_results.py
```
