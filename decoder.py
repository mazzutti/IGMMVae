import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, output_dim=784, hidden_dim=512, latent_dim=10):
        super().__init__()
        # Wider capacity: latent_dim -> 512 -> 1024 -> 784
        # GELU activation for smooth gradients
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)
