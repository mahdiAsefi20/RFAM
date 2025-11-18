import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1)
        self.bn1   = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_c)

        self.skip = None
        if stride != 1 or in_c != out_c:
            self.skip = nn.Conv2d(in_c, out_c, 1, stride=stride)

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.skip is not None:
            identity = self.skip(identity)

        return F.relu(out + identity)


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Block 1 → output: (728,19,19)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 728, 3, stride=16, padding=1),   # 299→?→19 approx
            nn.BatchNorm2d(728),
            nn.ReLU(),
        )

        # Block 2 → output: (728,19,19)
        self.block2 = ConvBlock(728, 728)

        # Block 3 → output: (2048,10,10)
        self.block3 = nn.Sequential(
            nn.Conv2d(728, 2048, 3, stride=2, padding=1),  # 19→10
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )

        # Block 4 → FC → scalar
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(625, 2)

    # ---- block-by-block forward calls ----
    def block_1(self, x):
        return self.block1(x)

    def block_2(self, x):
        return self.block2(x)

    def block_3(self, x):
        return self.block3(x)

    def block_4(self, x):
        # x = self.avg(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
