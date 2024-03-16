import torch
import torch.nn as nn
import numpy as np


# Mapping function for k-planes
def to_cartesian(theta_phi):
    return torch.stack(
        [
            torch.sin(theta_phi[:, 0]) * torch.cos(theta_phi[:, 1]),
            torch.sin(theta_phi[:, 0]) * torch.sin(theta_phi[:, 1]),
            torch.cos(theta_phi[:, 0]),
        ],
        axis=1,
    )


class NerfModel(nn.Module):
    def __init__(self, embedding_dim_pos=20, embedding_dim_direction=8, hidden_dim=128):
        super(NerfModel, self).__init__()

        self.block1 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 3 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1),
        )

        self.block3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.block4 = nn.Sequential(
            nn.Linear(hidden_dim // 2, 75),
        )

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.relu = nn.ReLU()

        self.bandwidth = nn.Parameter(torch.zeros((1, 25)))
        self.p = nn.Parameter(torch.randn((25, 2)))

    @staticmethod
    def positional_encoding(x, L):
        out = torch.empty(x.shape[0], x.shape[1] * 2 * L, device=x.device)
        for i in range(x.shape[1]):
            for j in range(L):
                out[:, i * (2 * L) + 2 * j] = torch.sin(2**j * x[:, i])
                out[:, i * (2 * L) + 2 * j + 1] = torch.cos(2**j * x[:, i])
        return out

    def forward(self, o, d):
        emb_x = self.positional_encoding(o, self.embedding_dim_pos // 2)
        h = self.block1(emb_x)
        tmp = self.block2(torch.cat((h, emb_x), dim=1))
        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1])
        h = self.block3(h)
        k = self.block4(h).reshape(o.shape[0], 25, 3)

        c = (
            k
            * torch.exp(
                (
                    self.bandwidth.unsqueeze(-1)
                    * to_cartesian(self.p).unsqueeze(0)
                    * d.unsqueeze(1)
                )
            )
        ).sum(1)

        return torch.sigmoid(c), sigma
