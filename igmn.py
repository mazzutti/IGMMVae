import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DifferentiableIGMN(nn.Module):
    def __init__(self, latent_dim, initial_K=2, beta=1.0, eta=18.0, covariance_type="full"):
        super().__init__()
        self.latent_dim = latent_dim
        self.K = initial_K
        self.beta = beta  # Temperature scale for novelty sigmoid
        self.eta = eta    # Mahalanobis threshold
        self.covariance_type = covariance_type
        
        # GMM parameters
        self.means = nn.Parameter(torch.randn(self.K, latent_dim))
        self.pi_logits = nn.Parameter(torch.zeros(self.K))
        
        # GMM components start small (standard deviation = 0.2, variance = 0.04)
        # to match IGMN original behavior where clusters are spawned tightly and grow to fit
        if self.covariance_type == "diagonal":
            self.logvars = nn.Parameter(torch.ones(self.K, latent_dim) * -3.2) # exp(-1.6) approx 0.2
        else:  # "full"
            init_L = 0.2 * torch.eye(latent_dim).unsqueeze(0).repeat(self.K, 1, 1)
            self.L_params = nn.Parameter(init_L)

    @property
    def pi(self):
        return F.softmax(self.pi_logits, dim=0)

    def forward(self, z):
        B, D = z.shape
        device = z.device
        
        if self.covariance_type == "diagonal":
            clamped_logvars = torch.clamp(self.logvars, min=-5.0, max=2.0)
            
            # Expand dimensions
            z_exp = z.unsqueeze(1)
            means_exp = self.means.unsqueeze(0)
            vars_exp = torch.exp(clamped_logvars).unsqueeze(0)
            
            # Mahalanobis distance
            diff = z_exp - means_exp
            d_sq = torch.sum((diff ** 2) / (vars_exp + 1e-8), dim=2)
            
            # Probability densities
            logvars_sum = torch.sum(clamped_logvars, dim=1, keepdim=True).T
            log_pdf = -0.5 * (D * math.log(2 * math.pi) + logvars_sum + d_sq)
            
        else:  # "full" (Cholesky parametrization)
            L_tril = torch.tril(self.L_params, diagonal=-1)
            diag_val = torch.diagonal(self.L_params, dim1=1, dim2=2)
            # Clamp GMM component standard deviation log-values to max=0.0 (std <= 1.0)
            clamped_diag = torch.clamp(diag_val, min=-5.0, max=0.0)
            L = L_tril + torch.diag_embed(torch.exp(clamped_diag))
            
            d_sq_list = []
            log_pdf_list = []
            
            for k in range(self.K):
                diff_k = z - self.means[k].unsqueeze(0) # (B, D)
                L_k = L[k] # (D, D)
                
                # solve L_k v = diff_k.T -> v is (D, B)
                v = torch.linalg.solve_triangular(L_k, diff_k.T, upper=False)
                d_sq_k = torch.sum(v ** 2, dim=0) # (B,)
                d_sq_list.append(d_sq_k)
                
                # log det of covariance matrix
                log_det_cov_k = 2.0 * torch.sum(torch.log(torch.diagonal(L_k) + 1e-8))
                
                log_pdf_k = -0.5 * (D * math.log(2 * math.pi) + log_det_cov_k + d_sq_k)
                log_pdf_list.append(log_pdf_k)
                
            d_sq = torch.stack(d_sq_list, dim=1) # (B, K)
            log_pdf = torch.stack(log_pdf_list, dim=1) # (B, K)
            
        # 3. Novelty probability p_new: (B,)
        min_d_sq, _ = torch.min(d_sq, dim=1)
        p_new = torch.sigmoid(self.beta * (min_d_sq - self.eta))
        
        # 4. Responsibilities
        pdf = torch.exp(log_pdf)
        pi_exp = self.pi.unsqueeze(0)
        unnorm_resp = pi_exp * pdf
        sum_unnorm = torch.sum(unnorm_resp, dim=1, keepdim=True)
        resp_existing_norm = unnorm_resp / (sum_unnorm + 1e-8)
        
        # Combine
        r_k = (1.0 - p_new.unsqueeze(1)) * resp_existing_norm
        r_new = p_new.unsqueeze(1)
        q_y = torch.cat([r_k, r_new], dim=1)
        
        return q_y, d_sq, p_new

    def spawn_component(self, new_mean, new_logvar=None):
        device = self.means.device
        new_mean = new_mean.detach().to(device).view(1, self.latent_dim)
        means_new = torch.cat([self.means.data, new_mean], dim=0)
        
        if self.covariance_type == "diagonal":
            if new_logvar is None:
                new_logvar = (torch.ones(1, self.latent_dim) * -3.2).to(device)
            else:
                new_logvar = new_logvar.detach().to(device).view(1, self.latent_dim)
            self.logvars = nn.Parameter(torch.cat([self.logvars.data, new_logvar], dim=0))
        else:  # "full"
            new_L = (0.2 * torch.eye(self.latent_dim)).to(device).unsqueeze(0)
            self.L_params = nn.Parameter(torch.cat([self.L_params.data, new_L], dim=0))
            
        # Update prior logits
        pi_current = self.pi.data
        new_pi = torch.tensor([0.05]).to(device)
        pi_new = torch.cat([pi_current * 0.95, new_pi], dim=0)
        logits_new = torch.log(pi_new + 1e-8)
        
        self.means = nn.Parameter(means_new)
        self.pi_logits = nn.Parameter(logits_new)
        
        self.K += 1

    def prune_components(self, threshold=0.02):
        if self.K <= 2:
            return False
            
        pi_vals = self.pi.data
        keep_indices = (pi_vals >= threshold).nonzero(as_tuple=True)[0]
        
        if len(keep_indices) < self.K:
            old_K = self.K
            self.means = nn.Parameter(self.means.data[keep_indices])
            
            if self.covariance_type == "diagonal":
                self.logvars = nn.Parameter(self.logvars.data[keep_indices])
            else:
                self.L_params = nn.Parameter(self.L_params.data[keep_indices])
                
            pi_new = pi_vals[keep_indices]
            pi_new = pi_new / pi_new.sum()
            self.pi_logits = nn.Parameter(torch.log(pi_new + 1e-8))
            
            self.K = len(keep_indices)
            print(f"\n[Pruning] Removed {old_K - self.K} components. Remaining: {self.K}")
            return True
            
        return False
