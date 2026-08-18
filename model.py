import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import Encoder
from decoder import Decoder
from igmm import DifferentiableIGMM

class GMVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=10, initial_K=2, beta=1.0, eta=None,
                 tau=0.1, delta=0.2, sp_min=5.0, v_min=200, reg_value=1e-5, max_nc=50,
                 dim_method="none", ard_lambda=1e-3, var_threshold=0.05, pca_threshold=0.01, covariance_type="full"):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dim_method = dim_method
        self.ard_lambda = ard_lambda
        self.var_threshold = var_threshold
        self.pca_threshold = pca_threshold
        self.covariance_type = covariance_type
        
        # Encoder
        self.encoder_module = Encoder(input_dim, hidden_dim, latent_dim)
        
        # Prior (Differentiable IGMM with full original IGMN parameters)
        self.prior = DifferentiableIGMM(
            latent_dim=latent_dim,
            initial_K=initial_K,
            tau=tau,
            delta=delta,
            sp_min=sp_min,
            v_min=v_min,
            reg_value=reg_value,
            max_nc=max_nc,
            beta=beta,
            eta=eta,
            covariance_type=covariance_type
        )
        
        # Decoder
        self.decoder_module = Decoder(input_dim, hidden_dim, latent_dim)
        
        # ARD parameters
        if self.dim_method == "ard":
            self.log_alpha = nn.Parameter(torch.zeros(latent_dim))

    def encode(self, x):
        return self.encoder_module(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder_module(z)

    def forward(self, x):
        device = x.device
        q_mean, q_logvar = self.encode(x)
        z = self.reparameterize(q_mean, q_logvar)
        
        z_processed = z
        active_dims = self.latent_dim
        sparsity_loss = 0.0
        
        if self.dim_method == "ard":
            scaling = torch.sigmoid(self.log_alpha)
            z_processed = z * scaling
            active_dims = torch.sum(scaling > 0.1).item()
            sparsity_loss = self.ard_lambda * torch.sum(scaling)
            
        elif self.dim_method == "kl-pruning":
            if self.training:
                batch_var = torch.var(q_mean, dim=0)
                mask = (batch_var > self.var_threshold).float()
                self.register_buffer("pruning_mask", mask, persistent=False)
            else:
                mask = getattr(self, "pruning_mask", torch.ones(self.latent_dim).to(device))
            
            z_processed = z * mask
            active_dims = torch.sum(mask).item()
            
        elif self.dim_method == "pca":
            if self.training and x.size(0) > 1:
                q_mean_centered = q_mean - torch.mean(q_mean, dim=0, keepdim=True)
                cov = torch.matmul(q_mean_centered.T, q_mean_centered) / (x.size(0) - 1 + 1e-8)
                
                cov_cpu = cov.cpu()
                eigenvalues_cpu, U_cpu = torch.linalg.eigh(cov_cpu)
                eigenvalues = eigenvalues_cpu.to(device)
                U = U_cpu.to(device)
                
                total_var = torch.sum(eigenvalues) + 1e-8
                mask = (eigenvalues > self.pca_threshold * total_var).float()
                
                z_centered = z - torch.mean(z, dim=0, keepdim=True)
                z_proj = torch.matmul(z_centered, U)
                z_proj_masked = z_proj * mask
                z_processed = torch.matmul(z_proj_masked, U.T) + torch.mean(z, dim=0, keepdim=True)
                
                active_dims = torch.sum(mask).item()
                self.register_buffer("pca_U", U, persistent=False)
                self.register_buffer("pca_mask", mask, persistent=False)
            else:
                U = getattr(self, "pca_U", torch.eye(self.latent_dim).to(device))
                mask = getattr(self, "pca_mask", torch.ones(self.latent_dim).to(device))
                z_centered = z - torch.mean(z, dim=0, keepdim=True)
                z_proj = torch.matmul(z_centered, U)
                z_proj_masked = z_proj * mask
                z_processed = torch.matmul(z_proj_masked, U.T) + torch.mean(z, dim=0, keepdim=True)
                active_dims = torch.sum(mask).item()
                
        recon_x = self.decode(z_processed)
        q_y, d_sq, p_new = self.prior(z_processed)
        
        return recon_x, z, q_mean, q_logvar, q_y, d_sq, p_new, active_dims, sparsity_loss

    def get_active_latent_indices(self):
        device = self.prior.means.device
        if self.dim_method == "ard":
            scaling = torch.sigmoid(self.log_alpha)
            return (scaling > 0.1).nonzero(as_tuple=True)[0]
        elif self.dim_method == "kl-pruning":
            mask = getattr(self, "pruning_mask", torch.ones(self.latent_dim).to(device))
            return (mask > 0.5).nonzero(as_tuple=True)[0]
        elif self.dim_method == "pca":
            mask = getattr(self, "pca_mask", torch.ones(self.latent_dim).to(device))
            return (mask > 0.5).nonzero(as_tuple=True)[0]
        else:
            return torch.arange(self.latent_dim).to(device)

    def classify_optimized(self, x):
        """
        Performs ultra-fast classification of input x using only active dimensions and clusters,
        skipping all pruned/dead components to maximize efficiency.
        """
        device = x.device
        B = x.shape[0]
        
        # 1. Encode to get mean
        q_mean, _ = self.encode(x)
        
        # 2. Get active latent dimensions
        active_idx = self.get_active_latent_indices()
        D_active = len(active_idx)
        
        # Slice mean vector to active dimensions only
        q_mean_active = q_mean[:, active_idx]
        
        # 3. Compute active IGMM responsibilities
        active_means = self.prior.means[:, active_idx]
        pi = self.prior.pi
        
        if self.covariance_type == "diagonal":
            clamped_logvars = torch.clamp(self.prior.logvars, min=-5.0, max=2.0)
            active_vars = torch.exp(clamped_logvars[:, active_idx])
            
            # Sliced Mahalanobis distance
            diff = q_mean_active.unsqueeze(1) - active_means.unsqueeze(0)
            d_sq = torch.sum((diff ** 2) / (active_vars.unsqueeze(0) + 1e-8), dim=2)
            
            logvars_sum = torch.sum(clamped_logvars[:, active_idx], dim=1, keepdim=True).T
            log_pdf = -0.5 * (D_active * math.log(2 * math.pi) + logvars_sum + d_sq)
        else:
            # Full Covariance: Project covariance matrix to active dimensions and do Cholesky
            L_tril = torch.tril(self.prior.L_params, diagonal=-1)
            diag_val = torch.diagonal(self.prior.L_params, dim1=1, dim2=2)
            clamped_diag = torch.clamp(diag_val, min=-3.0, max=0.0)
            L = L_tril + torch.diag_embed(torch.exp(clamped_diag))
            
            d_sq_list = []
            log_pdf_list = []
            
            for k in range(self.prior.K):
                L_k = L[k]
                Sigma_k = torch.matmul(L_k, L_k.T)
                # Slice covariance to active dimensions only
                Sigma_active_k = Sigma_k[active_idx][:, active_idx]
                
                # Perform Cholesky on the smaller sliced covariance matrix
                L_active_k = torch.linalg.cholesky(Sigma_active_k + 1e-6 * torch.eye(D_active).to(device))
                
                diff_k = q_mean_active - active_means[k].unsqueeze(0)
                v = torch.linalg.solve_triangular(L_active_k, diff_k.T, upper=False)
                d_sq_k = torch.sum(v ** 2, dim=0)
                d_sq_list.append(d_sq_k)
                
                log_det_cov_k = 2.0 * torch.sum(torch.log(torch.diagonal(L_active_k) + 1e-8))
                log_pdf_k = -0.5 * (D_active * math.log(2 * math.pi) + log_det_cov_k + d_sq_k)
                log_pdf_list.append(log_pdf_k)
                
            d_sq = torch.stack(d_sq_list, dim=1)
            log_pdf = torch.stack(log_pdf_list, dim=1)
            
        pdf = torch.exp(log_pdf)
        pi_exp = pi.unsqueeze(0)
        unnorm_resp = pi_exp * pdf
        
        # Predicted cluster index (arg max responsibility)
        _, predicted_classes = torch.max(unnorm_resp, dim=1)
        
        return predicted_classes
import math
