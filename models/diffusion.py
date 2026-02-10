import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionModel(nn.Module):
    """
    Convolutional Neural Network for Denoising Diffusion Probabilistic Models (DDPM).
    Architecture optimized for 1D ECG signals with time-step conditioning.
    """
    def __init__(self, input_channels=12, time_emb_dim=64):
        super(DiffusionModel, self).__init__()
        
        # Time embedding to inform the model about the current diffusion step
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Encoder: Downsampling path
        self.down1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.time_proj_d1 = nn.Linear(time_emb_dim, 32) # Projection for x1
        self.down2 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.time_proj_d2 = nn.Linear(time_emb_dim, 64) # Projection for x2
        self.down3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.time_proj_d3 = nn.Linear(time_emb_dim, 128) # Projection for x3
        
        # Decoder: Upsampling path
        self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.time_proj_u1 = nn.Linear(time_emb_dim, 64) # Projection for u1
        self.up2 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.time_proj_u2 = nn.Linear(time_emb_dim, 32) # Projection for u2
        self.final = nn.Conv1d(32, input_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        # x: [Batch, Channels, Length], t: [Batch]
        t_emb = self.time_mlp(t.view(-1, 1).float()) # [Batch, time_emb_dim]
        
        # Encoder
        x1_out = self.down1(x) # [Batch, 32, Length]
        # Add time embedding after convolution, before activation
        x1 = F.relu(x1_out + self.time_proj_d1(t_emb).unsqueeze(-1).expand(-1, -1, x1_out.shape[-1]))

        x2_out = self.down2(x1) # [Batch, 64, Length/2]
        # Add time embedding after convolution, before activation
        x2 = F.relu(x2_out + self.time_proj_d2(t_emb).unsqueeze(-1).expand(-1, -1, x2_out.shape[-1]))

        x3_out = self.down3(x2) # [Batch, 128, Length/4]
        # Add time embedding after convolution, before activation
        x3 = F.relu(x3_out + self.time_proj_d3(t_emb).unsqueeze(-1).expand(-1, -1, x3_out.shape[-1]))
        
        # Decoder (simplified U-Net style)
        u1_out = self.up1(x3) # [Batch, 64, Length/2]
        # Add time embedding after convolution, before activation
        u1 = F.relu(u1_out + self.time_proj_u1(t_emb).unsqueeze(-1).expand(-1, -1, u1_out.shape[-1]))

        # Skip connection: u1 + x2
        u2_out = self.up2(u1 + x2) # [Batch, 32, Length]
        # Add time embedding after convolution, before activation
        u2 = F.relu(u2_out + self.time_proj_u2(t_emb).unsqueeze(-1).expand(-1, -1, u2_out.shape[-1]))

        # Skip connection: u2 + x1
        return self.final(u2 + x1)

class ECGDiffusion:
    """
    Handler for the Diffusion process (Forward and Reverse).
    Implements the DDPM sampling logic.
    """
    def __init__(self, model, n_steps=100, beta_start=1e-4, beta_end=0.02, device='cpu'): # Added device argument
        self.model = model
        self.n_steps = n_steps
        self.device = device # Store device
        # Move beta to the specified device during initialization
        self.beta = torch.linspace(beta_start, beta_end, n_steps, device=device)
        self.alpha = 1. - self.beta
        # Ensure alpha_cumprod is also on the correct device
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x, t):
        "Forward process: adds noise to the signal at step t."
        noise = torch.randn_like(x, device=self.device) # Ensure noise is on the correct device
        # alpha_cumprod is already on the correct device, so indexing with t (also on device) is fine
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1. - self.alpha_cumprod[t]).view(-1, 1, 1)
        return sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise, noise

    @torch.no_grad()
    def denoise(self, x_t):
        """Reverse process: iteratively removes noise to restore the signal."""
        self.model.eval()
        # The device from model parameters is used to ensure consistency
        current_device = next(self.model.parameters()).device # Use model's device
        x_t = x_t.to(current_device)
        
        for t in reversed(range(self.n_steps)):
            t_batch = torch.full((x_t.shape[0],), t, dtype=torch.long, device=current_device)
            predicted_noise = self.model(x_t, t_batch)
            
            # These tensors are already on the correct device from __init__
            alpha_t = self.alpha[t]
            alpha_cumprod_t = self.alpha_cumprod[t]
            beta_t = self.beta[t]
            
            # DDPM Sampling formula
            x_t = (1 / torch.sqrt(alpha_t)) * (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t) * predicted_noise)
            
            if t > 1:#0: #here t>1 in Ho et al. but using temporally t>0 
                noise = torch.randn_like(x_t, device=current_device) # Ensure noise is on the correct device
                x_t += torch.sqrt(beta_t) * noise
        return x_t
"""
class ECGDiffusion:
    ""
    ""
    def __init__(self, model, n_steps=100, beta_start=1e-4, beta_end=0.02, device='cpu', fs=100, noise_snr_range=(10, 24)):
        self.model = model
        self.n_steps = n_steps
        self.device = device
        self.fs = fs
        self.noise_snr_range = noise_snr_range
        self.beta = torch.linspace(beta_start, beta_end, n_steps, device=device)
        self.alpha = 1. - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)

    def gaussian_noise(self, signal, noise_power):
        return torch.randn_like(signal) * torch.sqrt(noise_power)

    def baseline_wander(self, signal, noise_power, device):
        t = torch.linspace(0, signal.shape[-1] / self.fs, signal.shape[-1], device=device)
        rw = torch.cumsum(torch.randn_like(signal), dim=-1)
        rw = rw / rw.std() * torch.sqrt(noise_power * 0.3)
        return rw

    def powerline_noise(self, signal, noise_power, device):
        t = torch.linspace(0, signal.shape[-1] / self.fs, signal.shape[-1], device=device)
        f = np.random.choice([50, 60])  # 50 in EU & 60 in US
        pl = torch.sin(2 * np.pi * f * t)
        return pl * torch.sqrt(noise_power * 0.15)

    def muscle_artifact(self, signal, noise_power, device):
        emg = torch.randn_like(signal)
        #emg = F.avg_pool1d(emg.unsqueeze(0), kernel_size=7, stride=1, padding=3).squeeze(0)
        emg = F.avg_pool1d(emg, kernel_size=7, stride=1, padding=3)
        burst_mask = (torch.rand_like(signal) < 0.03).float()
        return emg * burst_mask * torch.sqrt(noise_power * 0.6)

    def add_noise(self, x, t):
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1. - self.alpha_cumprod[t]).view(-1, 1, 1)
        noise_gauss = torch.randn_like(x, device=self.device)
        x_noisy = sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise_gauss

        snr_db = np.random.uniform(*self.noise_snr_range)
        signal_power = torch.mean(x ** 2, dim=(1, 2), keepdim=True)
        noise_power = signal_power / (10 ** (snr_db / 10))

        if torch.rand(1) < 0.8:
            x_noisy += self.baseline_wander(x, noise_power, x.device)#sqrt_one_minus_alpha_cumprod * self.baseline_wander(x, noise_power, x.device)
        if torch.rand(1) < 0.5:
            x_noisy += self.powerline_noise(x, noise_power, x.device)#sqrt_one_minus_alpha_cumprod * self.powerline_noise(x, noise_power, x.device) 
        if torch.rand(1) < 0.4:
            x_noisy += self.muscle_artifact(x, noise_power, x.device)#sqrt_one_minus_alpha_cumprod * self.muscle_artifact(x, noise_power, x.device) 

        total_noise = x_noisy - sqrt_alpha_cumprod * x
        total_noise /= sqrt_one_minus_alpha_cumprod
        return x_noisy, total_noise

    @torch.no_grad()
    def denoise(self, x_t):
        ""Reverse process: iteratively removes noise to restore the signal.""
        self.model.eval()
        # The device from model parameters is used to ensure consistency
        current_device = next(self.model.parameters()).device # Use model's device
        x_t = x_t.to(current_device)
        
        for t in reversed(range(self.n_steps)):
            t_batch = torch.full((x_t.shape[0],), t, dtype=torch.long, device=current_device)
            predicted_noise = self.model(x_t, t_batch)
            
            # These tensors are already on the correct device from __init__
            alpha_t = self.alpha[t]
            alpha_cumprod_t = self.alpha_cumprod[t]
            beta_t = self.beta[t]
            
            # DDPM Sampling formula
            x_t = (1 / torch.sqrt(alpha_t)) * (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t) * predicted_noise)
            
            if t > 1:#0: #here t>1 in Ho et al. but using temporally t>0 
                noise = torch.randn_like(x_t, device=current_device) # Ensure noise is on the correct device
                x_t += torch.sqrt(beta_t) * noise
        return x_t
"""