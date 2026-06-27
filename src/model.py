"""PiW-3D 單人 3D 姿態網路(path-b,自己的乾淨架構)。
輸入 CSI 振幅 [B,9,30,20](9=3Tx×3Rx 天線對,30 子載波 × 20 時間)→ 回歸 14×3 = 42。
"""
import torch.nn as nn


def blk(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class WifiPose3DNet(nn.Module):
    def __init__(self, in_ch=9, n_joint=14):
        super().__init__()
        self.feat = nn.Sequential(
            blk(in_ch, 64), blk(64, 64), nn.MaxPool2d(2),       # 30x20 -> 15x10
            blk(64, 128), blk(128, 128), nn.MaxPool2d(2),       # -> 7x5
            blk(128, 256), nn.AdaptiveAvgPool2d((4, 2)),        # -> 256 x 4 x 2
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(256 * 4 * 2, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(512, n_joint * 3))

    def forward(self, x):
        return self.head(self.feat(x))      # [B, 42]
