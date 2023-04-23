import torch
import torch.nn as nn


class myGlobalContextBlock(nn.Module):
    def __init__(self, in_channels, scale = 16):
        super(myGlobalContextBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels//scale

        self.Conv_key = nn.Conv2d(self.in_channels, 1, 1)
        self.SoftMax = nn.Softmax(dim=1)

        self.Conv_value = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1),
            nn.LayerNorm([self.out_channels, 1, 1]),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.in_channels, 1),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        key = self.SoftMax(self.Conv_key(x).view(b, 1, -1).permute(0, 2, 1).view(b, -1, 1).contiguous())
        mean_x = torch.mean(key, dim=1, keepdim=True)
        variance_x = torch.pow(key - mean_x, 2)
        variance_key = self.SoftMax(variance_x)

        return variance_key

class Fusion05(nn.Module):
    def __init__(self, in_channels):
        super(Fusion05, self).__init__()
        self.flow = myGlobalContextBlock(in_channels)
        self.gamma = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.gamma.data.fill_(1.)  # ped2: 0 else 1
        self.Conv_value = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//16, 1),
            nn.LayerNorm([in_channels//16, 1, 1]),
            nn.ReLU(),
            nn.Conv2d(in_channels//16, in_channels, 1),
        )

    def forward(self, flow, img):
        b, c, h, w = img.size()
        flow_map = self.flow(flow)
        img = img.view(b, c, h * w)
        E = torch.matmul(img,flow_map)
        E = E.view(b, c, 1, 1).contiguous()
        img = img.view(b, c, h , w).contiguous()
        E = self.Conv_value(E)
        E =  self.gamma * E +  img
        return E
