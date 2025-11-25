import torch, torch.nn as nn


class TinyAudioCNN(nn.Module):
    """Backbone that consumes log-mel spectrograms (B,1,M,T).
    Returns: final_feat (B,64), taps [t1(B,16), t2(B,32)] for early exits.
    """
    def __init__(self, n_mels=64):
        super().__init__()
        self.block1 = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((2,2))
        )
        self.block2 = nn.Sequential(
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2,2))
        )
        self.block3 = nn.Sequential(
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
    )
    def forward(self, x):
        # x: (B,1,M,T)
        f1 = self.block1(x) # (B,16,M/2,T/2)
        f2 = self.block2(f1) # (B,32,M/4,T/4)
        f3 = self.block3(f2) # (B,64,1,1)
        # reduce time/freq with adaptive/max and mean to channel vectors
        t1 = torch.amax(f1, dim=-1).mean(-1) # (B,16)
        t2 = torch.amax(f2, dim=-1).mean(-1) # (B,32)
        t3 = f3.view(f3.size(0), -1) # (B,64)
        return t3, [t1, t2]