import torch
import torch.nn as nn

class RFAM(nn.Module):
    def __init__(self):
        super(RFAM, self).__init__()

        self.in_channels = 4
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels,1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.in_channels, 2, 3),
            nn.Sigmoid()
        )


    def forward(self, U_1 , U_2):

        U = torch.cat((U_1, U_2), dim=1)
        self.in_channels = U.shape[0] # number of feature map's channels after concatenation
        out = self.layer1(U)
        out = self.layer2(out)
        A_1, A_2 = torch.split(out, 1, dim=0)
        return A_1, A_2