import torch
import torch.nn as nn


class TinyAudioCNN(nn.Module):
    """
    TinyAudioCNN
    ------------
    A small convolutional neural network backbone for audio data.

    Expected input:
        x: log-mel spectrograms of shape (B, 1, M, T)
           B = batch size
           1 = mono channel
           M = number of mel-frequency bins (e.g., 64)
           T = number of time frames

    Outputs:
        final_feat: Tensor of shape (B, 64)
            - Deep, final feature representation after 3 conv blocks.
            - This is typically used by the final (deepest) classifier head.

        taps: list [t1, t2]
            t1: Tensor of shape (B, 16)
                - Shallow feature vector derived from the output of block1.
                - Used as an early-exit feature (early classifier head).
            t2: Tensor of shape (B, 32)
                - Intermediate feature vector derived from the output of block2.
                - Used as another early-exit feature.

    High-level purpose in ASHADIP:
        - Acts as an adapter/backbone for audio modality.
        - Extracts multi-level features that can feed early-exit heads.
        - Keeps the rest of the system agnostic to raw spectrogram details.
    """

    def __init__(self, n_mels: int = 64):
        """
        Args:
            n_mels (int):
                Number of mel-frequency bins in the input spectrogram.
                This parameter is not explicitly used in the layers
                (since conv/pool are spatially generic), but documents
                the expected input format and can be useful for sanity checks.
        """
        super().__init__()

        # -------------------------
        # Block 1: (1 -> 16 channels)
        # -------------------------
        # - Conv2d: learns local time-frequency patterns from log-mel input.
        # - BatchNorm2d: stabilizes training and normalizes feature maps.
        # - ReLU: introduces non-linearity.
        # - MaxPool2d: reduces resolution by a factor of 2 in both M and T.
        #
        # Input:  (B,  1, M,   T)
        # Output: (B, 16, M/2, T/2)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        # -------------------------
        # Block 2: (16 -> 32 channels)
        # -------------------------
        # Same pattern: conv + batch-norm + ReLU + max-pooling.
        #
        # Input:  (B, 16, M/2,   T/2)
        # Output: (B, 32, M/4,   T/4)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        # -------------------------
        # Block 3: (32 -> 64 channels)
        # -------------------------
        # - Conv2d: deeper feature extraction.
        # - BatchNorm2d + ReLU as before.
        # - AdaptiveAvgPool2d((1, 1)): spatially pools each feature map down
        #   to a single value, independent of the original M,T size.
        #
        # Input:  (B, 32, M/4,   T/4)
        # Output: (B, 64, 1,     1)
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the TinyAudioCNN.

        Args:
            x: Tensor of shape (B, 1, M, T)
               Batch of log-mel spectrograms.

        Returns:
            final_feat: Tensor of shape (B, 64)
                Deep feature vector from block3.
            taps: list [t1, t2]
                t1: (B, 16) shallow feature vector (from block1)
                t2: (B, 32) intermediate feature vector (from block2)
        """
        # x: (B, 1, M, T)

        # Pass through first conv block.
        # f1: (B, 16, M/2, T/2)
        f1 = self.block1(x)

        # Pass through second conv block.
        # f2: (B, 32, M/4, T/4)
        f2 = self.block2(f1)

        # Pass through third conv block.
        # f3: (B, 64, 1, 1)
        f3 = self.block3(f2)

        # -------------------------
        # Construct early-exit feature taps
        # -------------------------
        # For t1 and t2, we want to reduce the 2D time-frequency map to a
        # compact 1D vector per channel.
        #
        # Strategy:
        #   1. torch.amax(..., dim=-1)  -> max over time dimension T
        #   2. .mean(-1)                -> mean over frequency dimension M
        #
        # This gives a single summary value per channel.

        # From f1: (B,16,M/2,T/2)
        #   1. amax over last dim (T/2) -> (B,16,M/2)
        #   2. mean  over last dim (M/2) -> (B,16)
        t1 = torch.amax(f1, dim=-1).mean(-1)  # (B, 16)

        # From f2: (B,32,M/4,T/4)
        # Same pooling strategy.
        t2 = torch.amax(f2, dim=-1).mean(-1)  # (B, 32)

        # For the final deep feature, f3 is already (B,64,1,1),
        # so we just flatten the last two spatial dimensions.
        t3 = f3.view(f3.size(0), -1)  # (B, 64)

        # final_feat: deep representation
        # taps: early-exit representations at shallower depths
        return t3, [t1, t2]
