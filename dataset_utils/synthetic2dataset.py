import torch
import numpy as np
import os
import imageio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def get_rays(datapath, mode="train"):

    pose_file_names = [
        f for f in os.listdir(datapath + f"/{mode}/pose") if f.endswith(".txt")
    ]
    intrisics_file_names = [
        f for f in os.listdir(datapath + f"/{mode}/intrinsics") if f.endswith(".txt")
    ]
    img_file_names = [f for f in os.listdir(datapath + "/imgs") if mode in f]

    assert len(pose_file_names) == len(intrisics_file_names)
    assert len(img_file_names) == len(pose_file_names)

    # Read
    N = len(pose_file_names)
    poses = np.zeros((N, 4, 4))
    intrinsics = np.zeros((N, 4, 4))

    images = []

    for i in range(N):
        name = pose_file_names[i]

        pose = open(datapath + f"/{mode}/pose/" + name).read().split()
        poses[i] = np.array(pose, dtype=float).reshape(4, 4)

        intrinsic = open(datapath + f"/{mode}/intrinsics/" + name).read().split()
        intrinsics[i] = np.array(intrinsic, dtype=float).reshape(4, 4)

        # Read images
        img = imageio.imread(datapath + "/imgs/" + name.replace("txt", "png")) / 255.0
        images.append(img[None, ...])
    images = np.concatenate(images)

    H = images.shape[1]
    W = images.shape[2]

    if images.shape[3] == 4:  # RGBA -> RGB
        images = images[..., :3] * images[..., -1:] + (1 - images[..., -1:])

    rays_o = np.zeros((N, H * W, 3))
    rays_d = np.zeros((N, H * W, 3))
    target_px_values = images.reshape((N, H * W, 3))

    for i in range(N):

        c2w = poses[i]
        f = intrinsics[i, 0, 0]

        u = np.arange(W)
        v = np.arange(H)
        u, v = np.meshgrid(u, v)
        dirs = np.stack((u - W / 2, -(v - H / 2), -np.ones_like(u) * f), axis=-1)
        dirs = (c2w[:3, :3] @ dirs[..., None]).squeeze(-1)
        dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

        rays_d[i] = dirs.reshape(-1, 3)
        rays_o[i] += c2w[:3, 3]

    return rays_o, rays_d, target_px_values


if __name__ == "__main__":
    a, b, c = get_rays("../data_raw/fox/")

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
    np.save("fox.npy", flat_np)
