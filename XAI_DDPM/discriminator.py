import torch
from torch import nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Input Dimension: (nc) x 64 x 64
        self.conv1 = nn.Conv2d(params['nc'], params['ndf'],
            4, 2, 1, bias=False)

        # Input Dimension: (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(params['ndf'], params['ndf']*2,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params['ndf']*2)

        # Input Dimension: (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(params['ndf']*2, params['ndf']*4,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params['ndf']*4)

        # Input Dimension: (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(params['ndf']*4, params['ndf']*8,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(params['ndf']*8)

        # Input Dimension: (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(params['ndf']*8, 1, 1, 1, 0, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)#.clone()
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)#.clone()
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)#.clone()
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)#.clone()

        # Global Average Pooling (GAP)
        x = F.adaptive_avg_pool2d(x, (1, 1))#.clone()

        # 마지막 레이어에는 활성화 함수를 사용하지 않음
        x = self.conv5(x).clone()

        # 시그모이드 함수를 사용하여 0과 1로 변환
        x = F.sigmoid(x)#.clone()

        return x.view(x.size(0), -1)