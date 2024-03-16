import torch
import numpy as np
import os
import imageio

import json
import os
from tqdm import tqdm

################################## ns-process-data output ##################################


def get_rays(dir_path):
    f = open(f"{dir_path}/transforms.json")
    data = json.load(f)

    N = len(np.array(data["frames"]))

    # Make array for poses (placeholder)
    poses = np.zeros((N, 4, 4))

    # Make intrinsics array
    intrinsics = np.zeros((N, 4, 4))
    intrinsics[:, 0, 0] = data["fl_x"]
    intrinsics[:, 0, 2] = data["cx"]
    intrinsics[:, 1, 1] = data["fl_y"]
    intrinsics[:, 1, 2] = data["cy"]
    intrinsics[:, 2, 2] = 1
    intrinsics[:, 3, 3] = 1

    # Make images array
    images = []

    for i in tqdm(range(N), desc="Loading Images...", colour="red"):
        pose = np.array(data["frames"][i]["transform_matrix"], dtype=float)
        poses[i] = pose

        img = imageio.imread(f"{dir_path}{data['frames'][i]['file_path']}") / 255.0
        images.append(img[None, ...])

    print("Running concatenation...")
    images = np.concatenate(images)

    H = images.shape[1]
    W = images.shape[2]

    # For tranparent images
    if images.shape[3] == 4:  # RGBA ->  RGB
        images = images[..., :3] * images[..., -1:] + (1 - images[..., -1])

    # Ray Origins
    rays_o = np.zeros((N, H * W, 3))
    # Ray Directions
    rays_d = np.zeros((N, H * W, 3))
    # Image target
    target_px_values = images.reshape((N, H * W, 3))

    for i in tqdm(range(N), desc="Generating rays...", colour="green"):

        c2w = poses[i]
        f = intrinsics[i, 0, 0]

        u = np.arange(W)
        v = np.arange(H)
        u, v = np.meshgrid(u, v)
        dirs = np.stack((u - W / 2, -(v - H / 2), -np.ones_like(u) * f), axis=-1)

        # Accounting for theta (camera angle)
        dirs = (c2w[:3, :3] @ dirs[..., None]).squeeze(-1)
        dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

        rays_d[i] = dirs.reshape(-1, 3)
        rays_o[i] += c2w[:3, 3]

    return rays_o, rays_d, target_px_values


if __name__ == "__main__":
    a, b, c = get_rays()

    flat_data = torch.cat(
        (
            torch.from_numpy(a).reshape(-1, 3),
            torch.from_numpy(b).reshape(-1, 3),
            torch.from_numpy(c).reshape(-1, 3),
        ),
        dim=1,
    ).float()  # ? -> [n, [ray_o_x, ray_o_y, ray_o_z, ray_d_x, ray_d_y, ray_d_z, px_r, px_g, px_b]]

    flat_np = flat_data.numpy()

    # Save the data as a NumPY array
    np.save("ferrari.npy", flat_np)
