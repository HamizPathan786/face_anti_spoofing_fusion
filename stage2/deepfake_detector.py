import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x

class XceptionNet(nn.Module):
    """
    Custom Implementation of Xception Net from scratch.
    """
    def __init__(self):
        super(XceptionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x

class FrequencyBranch(nn.Module):
    def __init__(self, top_p=0.1):
        super().__init__()
        self.top_p = top_p
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128)
        )

    def forward(self, x):
        # x: (B, 3, H, W) -> Convert to grayscale for DCT (B, 1, H, W)
        x_gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        
        # 2D DCT via torch.fft
        # DCT can be approximated with FFT:
        fft2 = torch.fft.fft2(x_gray)
        magnitude = torch.abs(torch.fft.fftshift(fft2))
        
        # Keep top freq components (high frequency) -> mask center low freqs
        B, C, H, W = magnitude.shape
        center_h, center_w = H // 2, W // 2
        
        mask = torch.ones_like(magnitude)
        # remove center low frequencies (approx top_p of area)
        h_drop, w_drop = int(H * self.top_p), int(W * self.top_p)
        mask[:, :, center_h-h_drop:center_h+h_drop, center_w-w_drop:center_w+w_drop] = 0.0
        
        high_freq = magnitude * mask
        high_freq = torch.log(high_freq + 1e-7)  # Log scale
        
        out = self.cnn(high_freq)
        return out

class DeepfakeDetector(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.spatial_stream = XceptionNet()
        self.frequency_stream = FrequencyBranch(top_p=0.1)
        
        self.fc = nn.Sequential(
            nn.Linear(2048 + 128, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        spatial_feat = self.spatial_stream(x)
        freq_feat = self.frequency_stream(x)
        
        concat_feat = torch.cat([spatial_feat, freq_feat], dim=1)
        embedding = self.fc(concat_feat)
        score = self.classifier(embedding)
        
        return score.squeeze(1), embedding

    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'embedding_dim': self.embedding_dim,
        }, path)

    @classmethod
    def load_checkpoint(cls, path, device='cpu'):
        checkpoint = torch.load(path, map_location=device)
        model = cls(embedding_dim=checkpoint.get('embedding_dim', 512))
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        return model
