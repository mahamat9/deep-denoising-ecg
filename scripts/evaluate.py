import os

import torch
import torch.nn as nn

from models.autoencoder import ECGUNetDenoiser
from models.diffusion import DiffusionModel, ECGDiffusion
from utils.experiment_utils import DataConfig, SplitConfig, build_dataloaders, get_device, set_global_seed
from utils.train_utils import evaluate_model, plot_denoising_example


def evaluate_all(data_cfg: DataConfig, split_cfg: SplitConfig, eval_cfg: dict) -> None:
    set_global_seed(split_cfg.seed)
    device = get_device()

    _, _, test_loader, _, _, test_ds = build_dataloaders(data_cfg, split_cfg)

    models_path = eval_cfg["models_path"]
    results_path = eval_cfg["results_path"]
    os.makedirs(results_path, exist_ok=True)

    criterion = nn.MSELoss()

    ae_path = os.path.join(models_path, "best_autoencoder.pth")
    if os.path.exists(ae_path):
        model_ae = ECGUNetDenoiser(input_channels=eval_cfg["num_channels"]).to(device)
        model_ae.load_state_dict(torch.load(ae_path, map_location=device))
        ae_loss = evaluate_model(model_ae, test_loader, device, criterion)
        print(f"Autoencoder Test MSE: {ae_loss:.6f}")

        plot_denoising_example(
            model_ae,
            None,
            test_ds,
            device,
            model_name="Autoencoder",
            save_path=os.path.join(results_path, "autoencoder_eval_example.png"),
        )
    else:
        print("Autoencoder weights not found. Skipping AE evaluation.")

    diff_path = os.path.join(models_path, "best_diffusion.pth")
    if os.path.exists(diff_path):
        model_diff = DiffusionModel(input_channels=eval_cfg["num_channels"]).to(device)
        model_diff.load_state_dict(torch.load(diff_path, map_location=device))
        diffusion = ECGDiffusion(
            model_diff,
            n_steps=eval_cfg["n_steps"],
            beta_start=eval_cfg["beta_start"],
            beta_end=eval_cfg["beta_end"],
            device=device,
        )
        diff_loss = evaluate_model(
            model_diff,
            test_loader,
            device,
            criterion,
            is_diffusion=True,
            handler=diffusion,
        )
        print(f"Diffusion Test MSE: {diff_loss:.6f}")

        plot_denoising_example(
            model_diff,
            diffusion,
            test_ds,
            device,
            model_name="Diffusion",
            save_path=os.path.join(results_path, "diffusion_eval_example.png"),
        )
    else:
        print("Diffusion weights not found. Skipping diffusion evaluation.")


if __name__ == "__main__":
    data_cfg = DataConfig()
    split_cfg = SplitConfig(seed=42)
    eval_cfg = {
        "num_channels": 12,
        "n_steps": 100,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "models_path": "./models/",
        "results_path": "./results/",
    }

    evaluate_all(data_cfg, split_cfg, eval_cfg)
