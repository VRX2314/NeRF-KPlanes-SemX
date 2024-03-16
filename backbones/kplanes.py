import torch
import torch.nn as nn
import numpy as np


class NerfModel(nn.Module):
    def __init__(
        self, embedding_dim_direction=4, hidden_dim=64, N=512, F=96, scale=1.5
    ):
        """
        The parameter scale represents the maximum absolute value among all coordinates and is used for scaling the data
        """
        super(NerfModel, self).__init__()

        self.xy_plane = nn.Parameter(torch.rand((N, N, F)))
        self.yz_plane = nn.Parameter(torch.rand((N, N, F)))
        self.xz_plane = nn.Parameter(torch.rand((N, N, F)))

        self.block1 = nn.Sequential(
            nn.Linear(F, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Linear(15 + 3 * 4 * 2 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

        self.embedding_dim_direction = embedding_dim_direction
        self.scale = scale
        self.N = N

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2**j * x))
            out.append(torch.cos(2**j * x))
        return torch.cat(out, dim=1)

    def forward(self, x, d):
        sigma = torch.zeros_like(x[:, 0])
        c = torch.zeros_like(x)

        mask = (
            (x[:, 0].abs() < self.scale)
            & (x[:, 1].abs() < self.scale)
            & (x[:, 2].abs() < self.scale)
        )
        xy_idx = (
            ((x[:, [0, 1]] / (2 * self.scale) + 0.5) * self.N)
            .long()
            .clip(0, self.N - 1)
        )  # [batch_size, 2]
        yz_idx = (
            ((x[:, [1, 2]] / (2 * self.scale) + 0.5) * self.N)
            .long()
            .clip(0, self.N - 1)
        )  # [batch_size, 2]
        xz_idx = (
            ((x[:, [0, 2]] / (2 * self.scale) + 0.5) * self.N)
            .long()
            .clip(0, self.N - 1)
        )  # [batch_size, 2]
        F_xy = self.xy_plane[xy_idx[mask, 0], xy_idx[mask, 1]]  # [batch_size, F]
        F_yz = self.yz_plane[yz_idx[mask, 0], yz_idx[mask, 1]]  # [batch_size, F]
        F_xz = self.xz_plane[xz_idx[mask, 0], xz_idx[mask, 1]]  # [batch_size, F]
        F = F_xy * F_yz * F_xz  # [batch_size, F]

        h = self.block1(F)
        h, sigma[mask] = h[:, :-1], h[:, -1]
        c[mask] = self.block2(
            torch.cat(
                [self.positional_encoding(d[mask], self.embedding_dim_direction), h],
                dim=1,
            )
        )
        return c, sigma
