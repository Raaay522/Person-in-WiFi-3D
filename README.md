# Person-in-WiFi-3D

用 WiFi CSI 估計單人 3D 人體姿態(純 PyTorch / Windows,不依賴官方 OPERA/mmcv)。

## 功能

- **輸入**:WiFi CSI(Intel 5300,複數 `[3Tx, 3Rx, 30子載波, 20時間]`)。
- **輸出**:單人 3D 姿態,14 關節 × 3D 座標。
- **模型**:簡單 CNN(`[9,30,20] → 14×3`,1.63M 參數),PyTorch 2.5。
- **評估**:MPJPE(mm)+「平均姿態基線」對照 + PCK + 逐關節,訓練時即時印出並畫圖。
- **成果**:best MPJPE **92.6mm**(≈ 官方 PETR 91.7mm)。

## 專案結構

```
Person-in-WiFi-3D/
├─ src/
│  ├─ inspect_data.py   驗證資料格式(換資料先跑這支)
│  ├─ dataset.py        讀 CSI 複數→振幅 [9,30,20];單人 [14,3] root-relative;建快取
│  ├─ model.py          CNN [9,30,20] → 14×3
│  ├─ train.py          建快取 → 訓練 + MPJPE/PCK/逐關節 + 記 metrics.csv + 自動畫圖
│  ├─ plot_metrics.py   讀 metrics → 畫 5 張訓練指標圖(train.py 結束自動呼叫)
│  └─ plot_results.py   stride 對比學習曲線圖
├─ data/                          (不入庫)
│  ├─ wifipose_dataset/
│  │  ├─ train_data/{csi/*.mat, keypoint/*.npy, train_data_list.txt}   (89946)
│  │  └─ test_data /{csi/*.mat, keypoint/*.npy, test_data_list.txt}    (7824)
│  └─ _piw3d_cache_s{2,5}.npz     訓練快取
├─ runs/
│  ├─ last.pt                     訓練權重(不入庫)
│  ├─ metrics.csv                 每 epoch 的 loss/MPJPE/PCK 數值
│  ├─ loss.png                    訓練 loss 曲線
│  ├─ mpjpe.png                   val MPJPE vs 平均姿態基線
│  ├─ pck.png                     val PCK@150mm 曲線
│  ├─ per_joint.png               逐關節 MPJPE(模型 vs 基線)
│  ├─ val_overlay.png             val GT vs 預測 骨架重疊
│  └─ piw3d_results.png           stride 對比學習曲線
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
cd /d F:\Person-in-WiFi-3D   # Anaconda Prompt(cmd)需 /d 才會切換磁碟機

# 1) 驗證資料格式(換資料必跑:印 CSI/keypoint shape、座標單位、人數分布)
python src/inspect_data.py --root data/wifipose_dataset

# 2) 訓練(首次建快取較久;--stride 越小用越多資料,越準;結束自動畫 runs/ 指標圖)
python src/train.py --root data/wifipose_dataset --stride 2 --epochs 60

# 3) (可選)單獨重畫:指標圖 / stride 對比圖
python src/plot_metrics.py
python src/plot_results.py
```

## 成果圖(runs/)

訓練結束 `train.py` 會自動產生(也可 `python src/plot_metrics.py` 重畫):

| 圖 | 內容 |
|---|---|
| `loss.png` | 訓練 MSE loss 隨 epoch 下降 |
| `mpjpe.png` | val MPJPE vs 平均姿態基線(主指標,best 92.6mm) |
| `pck.png` | val PCK@150mm(關節落在 15cm 內的比例,~0.80) |
| `per_joint.png` | 14 關節各自 MPJPE,模型 vs 基線(高動態關節贏最多) |
| `val_overlay.png` | 幾筆 val 的 GT(綠)vs 預測(紅)3D 骨架重疊 |
| `metrics.csv` | 每 epoch 的 loss/MPJPE/PCK 數值紀錄 |
| `piw3d_results.png` | stride 2 vs 5 的學習曲線對比 |

> `val_overlay` 的骨架是以平均姿態自動建最小生成樹連線(官方未公布 14 關節順序)、取展幅最大兩軸投影,純為視覺化。
