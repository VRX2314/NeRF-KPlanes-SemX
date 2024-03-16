import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from backbones.nerf import NerfModel
from render import render_rays


@torch.no_grad()
def test(hn, hf, dataset, chunk_size=10, img_index=0, nb_bins=192, H=400, W=400):
    ray_origins = dataset[img_index * H * W : (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W : (img_index + 1) * H * W, 3:6]

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
    plt.figure()
    plt.imshow(img)
    plt.savefig(f"./inferences/{img_index}.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    device = torch.device("mps")
    model = torch.load("./models/nerf_model_plex_6.pth", map_location=device)

    testing_dataset = torch.from_numpy(
        np.load("./data/testing_data.pkl", allow_pickle=True)
    )

    for img_index in range(200):
        test(2, 6, testing_dataset, img_index=img_index, nb_bins=192, H=400, W=400)
