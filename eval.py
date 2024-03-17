import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from skimage.metrics import (
    peak_signal_noise_ratio,
    mean_squared_error,
    structural_similarity,
)
from skimage.color import rgb2gray

from backbones.nerf import NerfModel
from render import render_rays


@torch.no_grad()
def eval(
    hn,
    hf,
    dataset,
    chunk_size=10,
    img_index=0,
    nb_bins=192,
    H=400,
    W=400,
    save_path="./log.csv",
):

    mse_log = []
    psnr_log = []
    ssim_log = []

    ray_origins = dataset[img_index * H * W : (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W : (img_index + 1) * H * W, 3:6]
    target = dataset[img_index * H * W : (img_index + 1) * H * W, 6:]

    data = []
    for i in tqdm(
        range(int(np.ceil(H / chunk_size))), colour="Green", desc="Rendering..."
    ):
        ray_origins_ = ray_origins[i * W * chunk_size : (i + 1) * W * chunk_size].to(
            device
        )

        ray_directions_ = ray_directions[
            i * W * chunk_size : (i + 1) * W * chunk_size
        ].to(device)

        regenerated_px_values = render_rays(
            model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins
        )

        data.append(regenerated_px_values)

    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)
    target_img = target.reshape(400, 400, 3)
    target_img = target_img.detach().cpu().numpy()
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[0].set_title("Synthesized")
    ax[1].imshow(target_img)
    ax[1].set_title("Actual")

    psnr = peak_signal_noise_ratio(target_img, img)
    mse = mean_squared_error(target_img, img)
    ssim = structural_similarity(rgb2gray(target_img), rgb2gray(img), data_range=255)

    fig.suptitle(f"PSNR: {psnr}\nMSE: {mse}\nSSIM: {ssim}")

    plt.savefig(f"./metric_inferences/{img_index}.png", bbox_inches="tight")

    log = {"psnr": psnr_log, "mse": mse_log, "ssim": ssim_log}
    log = pd.DataFrame(log)
    log.to_csv(save_path)


if __name__ == "__main__":

    device = torch.device("mps")
    model = torch.load("./models/nerf_model_plex_6.pth", map_location=device)

    testing_dataset = torch.from_numpy(
        np.load("./data/training_data_.pkl", allow_pickle=True)
    )

    for img_index in range(200):
        eval(
            2,
            6,
            testing_dataset,
            img_index=img_index,
            nb_bins=192,
            H=400,
            W=400,
            save_path="./metric_inferences/log_name.csv",
        )
