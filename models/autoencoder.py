import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------
# Basic convolutional block
# --------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# --------------------------------------------------
# ECG U-Net 1D with residual learning
# --------------------------------------------------
class ECGUNetDenoiser(nn.Module):
    """
    1D U-Net for ECG denoising with residual learning.
    Input  : noisy ECG  [B, 12, T]
    Output : estimated noise [B, 12, T]
    Clean  : noisy - estimated_noise
    """
    def __init__(self, input_channels=12):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(input_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)

        self.pool = nn.MaxPool1d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(128, 256)

        # Decoder
        self.up3 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        # Output: predict noise
        self.out = nn.Conv1d(32, input_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)               # [B, 32, T]
        e2 = self.enc2(self.pool(e1))   # [B, 64, T/2]
        e3 = self.enc3(self.pool(e2))   # [B, 128, T/4]

        # Bottleneck
        b = self.bottleneck(self.pool(e3))  # [B, 256, T/8]

        # Decoder
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        noise_hat = self.out(d1)

        # Residual denoising
        clean_hat = x - noise_hat
        return clean_hat

class ECGAutoencoder(nn.Module):
    """
    Convolutional Denoising Autoencoder (CDAE) for ECG signal restoration.
    The architecture consists of a symmetric encoder-decoder structure with 1D convolutions.
    """
    def __init__(self, input_channels=12):
        super(ECGAutoencoder, self).__init__()
        
        # Encoder: Feature extraction and dimensionality reduction
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, stride=2, padding=1), # -> [32, 500]
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1), # -> [64, 250]
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1), # -> [128, 125]
            nn.ReLU()
        )
        
        # Decoder: Signal reconstruction from latent representation
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # -> [64, 250]
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # -> [32, 500]
            nn.ReLU(),
            nn.ConvTranspose1d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # -> [12, 1000]
            # No final activation to allow full range reconstruction
        )

    def forward(self, x):
        """
        Forward pass of the autoencoder.
        Args:
            x (torch.Tensor): Noisy ECG signal [Batch, Channels, Length]
        Returns:
            torch.Tensor: Reconstructed clean signal [Batch, Channels, Length]
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded