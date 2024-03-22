import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from backbones.kplanes import NerfModel
from render import render_rays


def train(
    nerf_model,
    optimizer,
    scheduler,
    data_loader,
    device="cpu",
    hn=0,
    hf=1,
    nb_epochs=int(1e5),
    nb_bins=192,
):
    training_loss = []
    for _ in range(nb_epochs):
        for batch in tqdm(data_loader):
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            ground_truth_px_values = batch[:, 6:].to(device)

            regenerated_px_values = render_rays(
                nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins
            )

            # MSE loss
            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
        scheduler.step()
        torch.save(nerf_model.cpu(), "nerf_model_plex")
        nerf_model.to(device)

    return training_loss


if __name__ == "__main__":
    # GPU Enabled
    device = torch.device("mps")

    # Load data
    training_dataset = torch.from_numpy(
        np.load("../datasets/nerf_pickled_data/training_data_.pkl", allow_pickle=True)
    )

    # Load model
    model = NerfModel(hidden_dim=256).to(device)
    # model = torch.load("./nerf_model_plex", map_location=torch.device("mps"))

    # Set Hyper-parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2, 4, 8], gamma=0.5
    )

    dataloader = DataLoader(training_dataset, batch_size=1024, shuffle=True)

    # hn, hf - near point far point
    train(
        model,
        optimizer,
        scheduler,
        dataloader,
        nb_epochs=14,
        device=device,
        hn=2,
        hf=6,
        nb_bins=192,
    )
