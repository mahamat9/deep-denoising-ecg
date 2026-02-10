"""
Utilities for training ECG denoising models.
Contains loss functions, metrics, evaluation utilities, and visualization tools.
"""

import numpy as np
import torch

import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========================================================================================
# EARLY STOPPING
# ========================================================================================
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Reference: https://medium.com/biased-algorithms/a-practical-guide-to-implementing-early-stopping-in-pytorch-for-model-training-99a7cbd46e9d
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# ========================================================================================
# LOSS FUNCTIONS
# ========================================================================================

def ecg_loss(pred, target, alpha=0.5):
    """
    ECG-aware loss combining amplitude and slope fidelity.
    
    Interpretation:
        - diff_loss → slope fidelity
          * protects QRS complexes
          * prevents over-smoothing
    
    Args:
        pred: predicted ECG signal (B, C, T)
        target: target clean ECG signal (B, C, T)
    
    Returns:
        Combined loss value
    """
    
    diff_pred = pred[:, :, 1:] - pred[:, :, :-1]
    diff_target = target[:, :, 1:] - target[:, :, :-1]
    diff_loss = torch.mean((diff_pred - diff_target) ** 2)

    return diff_loss


def ncc_loss(pred, target, eps=1e-8):
    """
    Normalized Cross-Correlation loss (1 - NCC).
    
    Penalizes:
        - Phase shifts
        - Bad morphologies
        - Errors on QRS complexes
    
    Args:
        pred: predicted ECG signal (B, C, T)
        target: target clean ECG signal (B, C, T)
        eps: small constant for numerical stability
    
    Returns:
        NCC loss value (1 - NCC)
    """
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    num = torch.sum(pred * target, dim=-1)
    den = torch.sqrt(
        torch.sum(pred ** 2, dim=-1) *
        torch.sum(target ** 2, dim=-1) + eps
    )

    ncc = num / den
    return 1 - ncc.mean()


def spectral_loss(x, y, loss_type):
    """
    Spectral loss in frequency domain.
    
    Args:
        x: predicted signal (B, C, T)
        y: target signal (B, C, T)
        loss_type: type of loss ("L1" or "L2" or "Huber" or "SmoothL1" or "ncc")
    
    Returns:
        Spectral loss value
    """

    X = spectral_signal(x)
    Y = spectral_signal(y)

    return loss_type(X, Y)



def get_combined_loss(
    output,
    clean,
    noisy,
    signal_loss_fn,
    noise_loss_fn,
    alpha_noise=0.1,
    alpha_spectral=1.0,
):
    """
    Factory function to create a combined loss function.
    
    Combined loss = signal_loss_fn(output, clean)
                    + alpha_spectral * spectral_loss(output, clean, signal_loss_fn)
                    + alpha_noise * noise_loss_fn(noise_pred, noise_target)
    
    Args:
        output: model output (denoised signal)
        clean: clean target signal
        noisy: noisy input signal
        signal_loss_fn: temporal domain loss function (e.g., ncc_loss, ecg_loss or nn losses)
        noise_loss_fn: loss for noise estimation (e.g., nn.HuberLoss() #nn.SmoothL1Loss() #nn.MSELoss())
        alpha_noise: weight for noise loss (default: 0.1)
        alpha_spectral: weight for spectral loss (default: 1.0)
    Returns:
        Combined loss value
    """
    noise_target = noisy - clean
    noise_pred = noisy - output
        
    main_loss = signal_loss_fn(output, clean)
    spec_loss = spectral_loss(output, clean, signal_loss_fn)
    noise_loss = noise_loss_fn(noise_pred, noise_target)

    return main_loss + alpha_spectral * spec_loss + alpha_noise * noise_loss


# ========================================================================================
# METRICS
# ========================================================================================

def autocorr_metric(x, max_lag=None):
    """
    Compute average normalized autocorrelation.
    
    Args:
        x: signal tensor (B, C, T)
        max_lag: maximum lag to compute (default: signal length)
    
    Returns:
        Mean autocorrelation value
    
    Example usage:
        ac_clean = autocorr_metric(clean)
        ac_out = autocorr_metric(output)
        history_ac['clean'].append(ac_clean)
        history_ac['out'].append(ac_out)
    """
    x = x - x.mean(dim=-1, keepdim=True)
    ac = []

    if max_lag is None:
        max_lag = x.shape[-1]
    elif max_lag > x.shape[-1]:
        max_lag = min(max_lag, x.shape[-1])
        print(f"max_lag bigger than signal's size, using {max_lag}")

    for lag in range(1, max_lag):
        #[..., :-lag] or [ , , :-lag]
        ac_lag = (x[..., :-lag] * x[..., lag:]).mean(dim=-1)
        ac.append(ac_lag)

    ac = torch.stack(ac, dim=-1)  # (B, C, L)
    return ac.mean().item()


def snr_improvement(clean, noisy, denoised, eps=1e-8):
    """
    Compute SNR before and after denoising.
    
    SNR = 10 * log10(signal_power / noise_power)
    
    Args:
        clean: clean signal (B, C, T)
        noisy: noisy signal (B, C, T)
        denoised: denoised signal (B, C, T)
        eps: small constant for numerical stability
    
    Returns:
        Tuple (snr_before, snr_after) in dB
    """
    # SNR before denoising
    noise_before = noisy - clean
    clean_signal_power = torch.sum(clean ** 2, dim=-1) #dim=(1, 2))
    clean_noise_power = torch.sum(noise_before ** 2, dim=-1) + eps
    snr_clean = 10 * torch.log10(clean_signal_power / clean_noise_power)

    # SNR after denoising
    noise_after = denoised - clean
    denoised_signal_power = torch.sum(denoised ** 2, dim=-1)
    denoised_noise_power = torch.sum(noise_after ** 2, dim=-1) + eps
    snr_denoised = 10 * torch.log10(denoised_signal_power / denoised_noise_power)

    return snr_clean.mean().item(), snr_denoised.mean().item()


# ========================================================================================
# EVALUATION UTILITIES
# ========================================================================================

def evaluate_model(model, loader, device, criterion, is_diffusion=False, handler=None):
    """
    Evaluate model on a dataset.
    
    Args:
        model: the model to evaluate
        loader: DataLoader for evaluation
        device: torch device
        criterion: loss function
        is_diffusion: whether the model is a diffusion model
        handler: diffusion handler (required if is_diffusion=True)
    
    Returns:
        Average loss over the dataset
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for noisy, clean in tqdm(loader, desc="Evaluating"):
            noisy, clean = noisy.to(device), clean.to(device)
            if is_diffusion:
                denoised = handler.denoise(noisy)
            else:
                denoised = model(noisy)
            total_loss += criterion(denoised, clean).item() * noisy.size(0)
    return total_loss / len(loader.dataset)


# ========================================================================================
# VISUALIZATION UTILITIES
# ========================================================================================

def plot_denoising_example(model, diffusion_handler, test_dataset, device, model_name, 
                          num_channels=12, save_path=None):
    """
    Plot a random denoising example from the test dataset.
    
    Args:
        model: trained model
        diffusion_handler: diffusion handler (None for autoencoder)
        test_dataset: test dataset
        device: torch device
        model_name: name of the model ("Autoencoder" or "Diffusion")
        num_channels: number of ECG channels (default: 12)
        save_path: path to save the figure (optional)
    """
    example_idx = np.random.randint(0, len(test_dataset))
    model.eval()
    with torch.no_grad():
        noisy, clean = test_dataset[example_idx]
        noisy, clean = noisy.unsqueeze(0).to(device), clean.unsqueeze(0).to(device)
        
        if model_name == "Autoencoder":
            denoised = model(noisy)
        elif model_name == "Diffusion":
            denoised = diffusion_handler.denoise(noisy)
        else:
            raise ValueError("Invalid model name for denoising example.")

        noisy = noisy.squeeze(0).cpu().numpy()
        clean = clean.squeeze(0).cpu().numpy()
        denoised = denoised.squeeze(0).cpu().numpy()

        plt.figure(figsize=(20, 7))
        lead_to_plot = np.random.randint(0, num_channels)
        
        plt.plot(clean[lead_to_plot], label='Clean ECG', color='green', linewidth=1)
        plt.plot(noisy[lead_to_plot], label='Noisy ECG', color='red', alpha=0.7)
        plt.plot(denoised[lead_to_plot], label='Denoised ECG', color='black', 
                linewidth=1, linestyle='--')
        plt.title(f'{model_name} - Lead {lead_to_plot + 1}')
        plt.xlabel('Time Steps')
        plt.ylabel('Amplitude')
        plt.legend()
        #plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def plot_training_curves(history, title="Training Curves", save_path=None):
    """
    Plot training and validation loss curves.
    
    Args:
        history: dictionary with 'train' and 'val' keys containing loss lists
        title: plot title
        save_path: path to save the figure (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train'], label='Train Loss')
    plt.plot(history['val'], label='Val Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    #plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_snr_evolution(history_snr, title="SNR Evolution", save_path=None):
    """
    Plot SNR evolution during training.
    
    Args:
        history_snr: dictionary with 'before' and 'after' keys (or just 'after' for diffusion)
        title: plot title
        save_path: path to save the figure (optional)
    """
    plt.figure(figsize=(10, 6))
    
    if 'before' in history_snr:
        plt.plot(history_snr['before'], label='SNR before (val)', color='yellow')
    if 'after' in history_snr:
        plt.plot(history_snr['after'], label='SNR after (val)', color='green')
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('SNR (dB)')
    plt.legend()
    #plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_combined_results(history_loss, history_snr, model_name, save_path=None):
    """
    Plot combined training curves and SNR evolution.
    
    Args:
        history_loss: dictionary with 'train' and 'val' loss
        history_snr: dictionary with SNR metrics
        model_name: name of the model
        save_path: path to save the figure (optional)
    """
    plt.figure(figsize=(20, 8))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(history_loss['train'], label='Train Loss')
    plt.plot(history_loss['val'], label='Val Loss')
    plt.title(f"{model_name} Loss Curves")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # SNR evolution
    plt.subplot(1, 2, 2)
    if 'before' in history_snr:
        plt.plot(history_snr['before'], label='SNR before (val)', color='red')
    if 'after' in history_snr:
        plt.plot(history_snr['after'], label='SNR after (val)', color='green')
    plt.title(f"{model_name} SNR Evolution")
    plt.xlabel('Epoch')
    plt.ylabel('SNR (dB)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ========================================================================================
# HELPER FUNCTIONS
# ========================================================================================

def spectral_signal(x):
    """
    Compute the magnitude spectrum of the signal using FFT.
    
    Args:
        x: input signal tensor (B, C, T)
    
    Returns:
        Magnitude spectrum tensor (B, C, T)
    """
    X_f = torch.fft.fft(x, dim=-1)
    X_mag = torch.abs(X_f)
    return X_mag