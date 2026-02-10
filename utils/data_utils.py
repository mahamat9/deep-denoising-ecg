import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import wfdb
import ast
import os
from scipy.signal import butter, filtfilt
from tqdm import tqdm
from joblib import Parallel, delayed

def load_single_wfdb(f, path):
    """Utility function to load a single WFDB file (used for parallelization)."""
    signal, meta = wfdb.rdsamp(os.path.join(path, f))
    return signal

def load_raw_data(df, sampling_rate, path, n_jobs=-1):
    """Loads raw signals in parallel from WFDB files."""
    filenames = df.filename_lr if sampling_rate == 100 else df.filename_hr
    print(f"Parallel loading of {len(filenames)} ECG signals (n_jobs={n_jobs})...")
    
    data = Parallel(n_jobs=n_jobs)(
        delayed(load_single_wfdb)(f, path) for f in tqdm(filenames, desc="WFDB Reading")
    )
    return np.array(data)

def apply_filter_to_signal(x, b, a):
    """Applies filter to all channels of a single signal."""
    x_filtered = np.zeros_like(x)
    for c in range(x.shape[1]):
        x_filtered[:, c] = filtfilt(b, a, x[:, c])
    return x_filtered

def load_ptbxl_data(path, sampling_rate=100, test_fold=10, preprocess=False, n_jobs=-1):
    """Loads PTB-XL database with parallelization and optional preprocessing."""
    # Load annotations
    Y = pd.read_csv(os.path.join(path, 'ptbxl_database.csv'), index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load signals in parallel
    X = load_raw_data(Y, sampling_rate, path, n_jobs=n_jobs)

    # Optional preprocessing: Parallel low-pass filtering
    if preprocess:
        print(f"Parallel signal preprocessing (n_jobs={n_jobs})...")
        nyq = 0.5 * sampling_rate
        cutoff = 40
        normal_cutoff = cutoff / nyq
        b, a = butter(2, normal_cutoff, btype='low', analog=False)
        
        X = np.array(Parallel(n_jobs=n_jobs)(
            delayed(apply_filter_to_signal)(x, b, a) for x in tqdm(X, desc="Filtering")
        ))

    # Split Train / Test based on recommended folds
    X_train = X[np.where(Y.strat_fold != test_fold)]
    X_test = X[np.where(Y.strat_fold == test_fold)]
    
    return X_train, X_test

def get_optimized_dataset(data_path, sampling_rate=100, preprocess=False, n_jobs=-1):
    """
    Loads data via a .npy file for maximum speed.
    Creates the file if it doesn't exist (with parallelization).
    """
    suffix = "_preprocessed" if preprocess else ""
    npy_file = os.path.join(data_path, f"ptbxl_data_{sampling_rate}{suffix}.npy")
    
    if os.path.exists(npy_file):
        print(f"Loading data from cache: {npy_file}")
        return np.load(npy_file)
    else:
        print("Cache file not found. Initial parallel data loading...")
        X_train, X_test = load_ptbxl_data(data_path, sampling_rate=sampling_rate, preprocess=preprocess, n_jobs=n_jobs)
        X_all = np.concatenate([X_train, X_test], axis=0)
        print(f"Saving data to cache: {npy_file}")
        np.save(npy_file, X_all)
        return X_all

class ECGDenoisingDataset(Dataset):
    def __init__(self, data, noise_all = True, fs=100, noise_snr_range=(10, 24)):#(5, 25)):
        """
            data (np.array): ECG data (N, samples, channels)
            fs (int): sampling frequency (100 or 500)
            noise_snr_range (tuple): SNR range (min, max) in dB for noise.
        """
        # Convert to (N, Channels, Length) for Conv1d
        self.data = torch.FloatTensor(data).permute(0, 2, 1)
        self.noise_all = noise_all
        self.fs = fs
        self.noise_snr_range = noise_snr_range

    def __len__(self):
        return len(self.data)

    # ----------------------------------------------------
    # Noise components
    # ----------------------------------------------------

    def gaussian_noise(self, signal, noise_power):
        return torch.randn_like(signal) * torch.sqrt(noise_power)

    def baseline_wander(self, signal, noise_power):
        """
        breathing, 'bad' posture
        """
        t = torch.linspace(
            0, signal.shape[-1] / self.fs,
            signal.shape[-1],
            device=signal.device
        )
        """
        f = np.random.uniform(0.05, 0.5)
        phase = np.random.uniform(0, 2 * np.pi)
        bw = torch.sin(2 * np.pi * f * t + phase)
        return bw * torch.sqrt(noise_power)
        """
        rw = torch.cumsum(torch.randn_like(signal), dim=-1)
        rw = rw / rw.std() * torch.sqrt(noise_power)
        return rw
        
    def powerline_noise(self, signal, noise_power):
        t = torch.linspace(
            0, signal.shape[-1] / self.fs,
            signal.shape[-1],
            device=signal.device
        )
        f = np.random.choice([50, 60]) #50 in EU & 60 in US
        pl = torch.sin(2 * np.pi * f * t)
        return pl * torch.sqrt(noise_power)

    def muscle_artifact(self, signal, noise_power):
        emg = torch.randn_like(signal)
        emg = F.avg_pool1d(
            emg.unsqueeze(0),
            kernel_size=7,
            stride=1,
            padding=3
        ).squeeze(0)

        burst_mask = (torch.rand_like(signal) < 0.03).float() # bursts
        return emg * burst_mask * torch.sqrt(noise_power)

    # ----------------------------------------------------
    # Noise mixer
    # ----------------------------------------------------

    def add_noise(self, signal):
        snr_db = np.random.uniform(*self.noise_snr_range)#(self.noise_snr_range[0], self.noise_snr_range[1])
        signal_power = torch.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))

        noise = self.gaussian_noise(signal, noise_power)

        if torch.rand(1) < 0.9:
            noise += self.baseline_wander(signal, noise_power)
        if torch.rand(1) < 0.5:
            noise += self.powerline_noise(signal, noise_power)
        if torch.rand(1) < 0.4:
            noise += self.muscle_artifact(signal, noise_power)

        return signal + noise

    # ----------------------------------------------------
    # Dataset
    # ----------------------------------------------------

    def __getitem__(self, idx):
        clean_signal = self.data[idx]
        
        # Channel-wise normalization for training stability
        #std = clean_signal.std(dim=-1, keepdim=True) + 1e-8
        #mean = clean_signal.mean(dim=-1, keepdim=True)
        #clean_signal = (clean_signal - mean) / std
        
        if self.noise_all:
            # Add noise to normalized signal
            noisy_signal = self.add_noise(clean_signal)
            return noisy_signal, clean_signal
        else:
            return clean_signal
