import torch
import torch.nn as nn


class ExitNet(nn.Module):
    """Two early exits (on backbone taps) + one final head.
    forward(x) -> [logits_exit1, logits_exit2, logits_final]
    """
    def __init__(self, backbone: nn.Module, tap_dims=(16,32), final_dim=64, num_classes: int = 2):
        super().__init__()
        self.backbone = backbone
        self.exit1 = nn.Linear(tap_dims[0], num_classes)
        self.exit2 = nn.Linear(tap_dims[1], num_classes)
        self.final = nn.Linear(final_dim, num_classes)

    def forward(self, x):
        final_feat, taps = self.backbone(x) # final_feat (B,64), taps [t1(B,16), t2(B,32)]
        logits1 = self.exit1(taps[0])
        logits2 = self.exit2(taps[1])
        logits3 = self.final(final_feat)
        return [logits1, logits2, logits3]