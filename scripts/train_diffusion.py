import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.diffusion import DiffusionModel, ECGDiffusion
from utils.experiment_utils import DataConfig, SplitConfig, build_dataloaders, get_device, set_global_seed
from utils.train_utils import (
    EarlyStopping,
    evaluate_model,
    plot_combined_results,
    plot_denoising_example,
    snr_improvement,
)


def train_diffusion(data_cfg: DataConfig, split_cfg: SplitConfig, train_cfg: dict) -> None:
    set_global_seed(split_cfg.seed)
    device = get_device()

    train_loader, val_loader, test_loader, _, _, test_ds = build_dataloaders(data_cfg, split_cfg)

    models_path = train_cfg["models_path"]
    results_path = train_cfg["results_path"]
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    model = DiffusionModel(input_channels=train_cfg["num_channels"]).to(device)
    diffusion = ECGDiffusion(
        model,
        n_steps=train_cfg["n_steps"],
        beta_start=train_cfg["beta_start"],
        beta_end=train_cfg["beta_end"],
        device=device,
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=train_cfg["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    early_stopping = EarlyStopping(
        patience=train_cfg["patience"],
        verbose=True,
        path=os.path.join(models_path, "best_diffusion.pth"),
    )

    history_loss = {"train": [], "val": []}
    history_snr = {"after": []}

    for epoch in tqdm(range(train_cfg["epochs"]), desc="Training Diffusion"):
        model.train()
        train_loss = 0.0

        for _, clean in train_loader:
            clean = clean.to(device)
            t = torch.randint(0, diffusion.n_steps, (clean.size(0),), device=device)
            noisy, noise = diffusion.add_noise(clean, t)

            optimizer.zero_grad()
            pred_noise = model(noisy, t)
            loss = criterion(pred_noise, noise)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * clean.size(0)

        train_loss /= len(train_loader.dataset)
        history_loss["train"].append(train_loss)

        model.eval()
        val_loss = 0.0
        snr_vals = []

        with torch.no_grad():
            for _, clean in val_loader:
                clean = clean.to(device)
                t = torch.randint(0, diffusion.n_steps, (clean.size(0),), device=device)
                noisy, noise = diffusion.add_noise(clean, t)
                pred_noise = model(noisy, t)
                val_loss += criterion(pred_noise, noise).item() * clean.size(0)

                x_denoised = diffusion.denoise(noisy)
                _, snr_after = snr_improvement(clean, noisy, x_denoised)
                snr_vals.append(snr_after)

        val_loss /= len(val_loader.dataset)
        mean_snr = sum(snr_vals) / len(snr_vals)
        history_loss["val"].append(val_loss)
        history_snr["after"].append(mean_snr)

        print(
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"SNR(after): {mean_snr:.2f} dB"
        )

        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered for Diffusion")
            break

    if os.path.exists(early_stopping.path):
        model.load_state_dict(torch.load(early_stopping.path, map_location=device))

    plot_combined_results(
        history_loss,
        history_snr,
        model_name="Diffusion",
        save_path=os.path.join(results_path, "diffusion_training.png"),
    )

    test_loss = evaluate_model(model, test_loader, device, criterion, is_diffusion=True, handler=diffusion)
    print(f"Diffusion Test loss: {test_loss:.6f}")

    plot_denoising_example(
        model,
        diffusion,
        test_ds,
        device,
        model_name="Diffusion",
        save_path=os.path.join(results_path, "diffusion_example.png"),
    )


if __name__ == "__main__":
    data_cfg = DataConfig()
    split_cfg = SplitConfig(seed=42)
    train_cfg = {
        "num_channels": 12,
        "epochs": 40,
        "lr": 1e-3,
        "patience": 5,
        "n_steps": 100,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "models_path": "./models/",
        "results_path": "./results/",
    }

    train_diffusion(data_cfg, split_cfg, train_cfg)
