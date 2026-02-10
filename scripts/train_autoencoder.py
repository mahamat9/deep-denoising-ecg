import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.autoencoder import ECGUNetDenoiser
from utils.experiment_utils import DataConfig, SplitConfig, build_dataloaders, get_device, set_global_seed
from utils.train_utils import (
    EarlyStopping,
    evaluate_model,
    get_combined_loss,
    plot_combined_results,
    plot_denoising_example,
    snr_improvement,
)


def train_autoencoder(data_cfg: DataConfig, split_cfg: SplitConfig, train_cfg: dict) -> None:
    set_global_seed(split_cfg.seed)
    device = get_device()

    train_loader, val_loader, test_loader, _, _, test_ds = build_dataloaders(data_cfg, split_cfg)

    models_path = train_cfg["models_path"]
    results_path = train_cfg["results_path"]
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    model = ECGUNetDenoiser(input_channels=train_cfg["num_channels"]).to(device)

    signal_loss_fn = nn.MSELoss()
    noise_loss_fn = nn.SmoothL1Loss()

    optimizer = optim.Adam(model.parameters(), lr=train_cfg["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    early_stopping = EarlyStopping(
        patience=train_cfg["patience"],
        verbose=True,
        path=os.path.join(models_path, "best_autoencoder.pth"),
    )

    history_loss = {"train": [], "val": []}
    history_snr = {"before": [], "after": []}

    for epoch in tqdm(range(train_cfg["epochs"]), desc="Training Autoencoder"):
        model.train()
        train_loss = 0.0

        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(noisy)

            loss = get_combined_loss(
                output,
                clean,
                noisy,
                signal_loss_fn,
                noise_loss_fn,
                alpha_noise=train_cfg["alpha_noise"],
                alpha_spectral=train_cfg["alpha_spectral"],
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * noisy.size(0)

        train_loss /= len(train_loader.dataset)
        history_loss["train"].append(train_loss)

        model.eval()
        val_loss = 0.0
        snr_before_vals = []
        snr_after_vals = []

        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                output = model(noisy)
                val_loss_batch = get_combined_loss(
                    output,
                    clean,
                    noisy,
                    signal_loss_fn,
                    noise_loss_fn,
                    alpha_noise=train_cfg["alpha_noise"],
                    alpha_spectral=train_cfg["alpha_spectral"],
                )
                val_loss += val_loss_batch.item() * noisy.size(0)

                snr_before, snr_after = snr_improvement(clean, noisy, output)
                snr_before_vals.append(snr_before)
                snr_after_vals.append(snr_after)

        val_loss /= len(val_loader.dataset)
        history_loss["val"].append(val_loss)

        mean_before = sum(snr_before_vals) / len(snr_before_vals)
        mean_after = sum(snr_after_vals) / len(snr_after_vals)
        history_snr["before"].append(mean_before)
        history_snr["after"].append(mean_after)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"SNR(before): {mean_before:.2f} dB | SNR(after): {mean_after:.2f} dB | "
            f"LR: {current_lr:.2e}"
        )

        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered for Autoencoder")
            break

    if os.path.exists(early_stopping.path):
        model.load_state_dict(torch.load(early_stopping.path, map_location=device))

    plot_combined_results(
        history_loss,
        history_snr,
        model_name="Autoencoder",
        save_path=os.path.join(results_path, "autoencoder_training.png"),
    )

    test_loss = evaluate_model(model, test_loader, device, signal_loss_fn)
    print(f"Autoencoder Test loss: {test_loss:.6f}")

    plot_denoising_example(
        model,
        None,
        test_ds,
        device,
        model_name="Autoencoder",
        save_path=os.path.join(results_path, "autoencoder_example.png"),
    )


if __name__ == "__main__":
    data_cfg = DataConfig()
    split_cfg = SplitConfig(seed=42)
    train_cfg = {
        "num_channels": 12,
        "epochs": 40,
        "lr": 1e-3,
        "patience": 5,
        "alpha_noise": 0.1,
        "alpha_spectral": 1.0,
        "models_path": "./models/",
        "results_path": "./results/",
    }

    train_autoencoder(data_cfg, split_cfg, train_cfg)
