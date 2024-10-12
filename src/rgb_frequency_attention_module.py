from array import array

import torch
import torch.nn as nn

class RFAM(nn.Module):
    def __init__(self, in_channels):
        super(RFAM, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,1, 1, 0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels, 2, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, U_1 , U_2):

        U = torch.cat((U_1, U_2), dim=1)
        out = self.layer1(U)
        out = self.layer2(out)
        A_1, A_2 = torch.split(out, 1, dim=1)
        return A_1, A_2