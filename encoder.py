import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=10):
        super().__init__()
        # Wider capacity: 784 -> 1024 -> 512
        # GELU activation prevents dead neurons and helps gradient flow
        self.enc_shared = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, hidden_dim),
            nn.GELU()
        )
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.enc_shared(x)
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar
