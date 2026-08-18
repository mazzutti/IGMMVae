import torch
import torch.nn as nn
import torch.nn.functional as F

class FCVAE(nn.Module):
    """
    Fully Convolutional Variational Autoencoder (FCVAE) with Fixed Latent Dimension (D=10)
    and Standard Isotropic Gaussian Prior p(z) = N(0, I).
    """
    def __init__(self, latent_dim=10):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder (Fully Convolutional Backbone)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # -> (B, 32, 14, 14)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 7, 7)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> (B, 128, 4, 4)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc_encoder = nn.Linear(128 * 4 * 4, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder (Fully Transposed Convolutional Backbone)
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0), # -> (B, 64, 7, 7)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (B, 32, 14, 14)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> (B, 1, 28, 28)
            nn.Sigmoid()
        )

    def encode(self, x):
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        h = F.leaky_relu(self.fc_encoder(h), 0.2)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decoder(z)
        h = h.view(h.size(0), 128, 4, 4)
        recon_x = self.decoder_conv(h)
        return recon_x.view(-1, 784)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        if x.dim() == 4:
            x = x.view(-1, 784)
        # Weighted stroke BCE matching GMVAE
        stroke_weight = 1.0 + 0.8 * x
        bce = -(stroke_weight * x * torch.log(recon_x + 1e-8) + (1.0 - x) * torch.log(1.0 - recon_x + 1e-8)).sum(dim=1).mean()
        # Standard Isotropic KL Divergence: KL(q(z|x) || N(0, I))
        kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        total_loss = bce + beta * kl
        return total_loss, bce, kl
