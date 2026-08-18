import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DifferentiableIGMM(nn.Module):
    """
    Differentiable Incremental Gaussian Mixture Model (IGMM)
    Faithfully implementing the original IGMN (Incremental Gaussian Mixture Network) architecture:
    - tau: Novelty/likelihood threshold for creating new components
    - delta: Fraction of data range used for initial covariance scaling (L_init = delta * I)
    - sp_min: Minimum accumulated activation threshold for pruning noisy components
    - v_min: Minimum age/sample threshold before a component is eligible for pruning
    - reg_value: Regularization jitter for numerical stability
    - max_nc: Maximum number of Gaussian components allowed
    """
    def __init__(
        self,
        latent_dim,
        initial_K=2,
        tau=0.1,
        delta=0.2,
        sp_min=5.0,
        v_min=200,
        reg_value=1e-5,
        max_nc=50,
        beta=1.0,
        eta=None,
        covariance_type="full"
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.K = initial_K
        self.tau = tau
        self.delta = delta
        self.sp_min = sp_min
        self.v_min = v_min
        self.reg_value = reg_value
        self.max_nc = max_nc
        self.beta = beta
        self.covariance_type = covariance_type
        
        # If eta is not explicitly provided, derive from tau
        if eta is not None:
            self.eta = eta
        else:
            # eta = -2 * ln(tau)
            self.eta = -2.0 * math.log(max(1e-8, min(0.999, tau)))
            
        # IGMM Parameters
        self.means = nn.Parameter(torch.randn(self.K, latent_dim))
        self.pi_logits = nn.Parameter(torch.zeros(self.K))
        
        # Initial covariance scaled by delta
        if self.covariance_type == "diagonal":
            init_logvar = 2.0 * math.log(self.delta)
            self.logvars = nn.Parameter(torch.ones(self.K, latent_dim) * init_logvar)
        else:  # "full"
            init_L = self.delta * torch.eye(latent_dim).unsqueeze(0).repeat(self.K, 1, 1)
            self.L_params = nn.Parameter(init_L)
            
        # IGMN Accumulators for component age (v) and accumulated activation (sp)
        self.register_buffer('v', torch.zeros(self.K))
        self.register_buffer('sp', torch.zeros(self.K))

    @property
    def pi(self):
        return F.softmax(self.pi_logits, dim=0)

    def forward(self, z):
        B, D = z.shape
        device = z.device
        
        if self.covariance_type == "diagonal":
            clamped_logvars = torch.clamp(self.logvars, min=-6.0, max=3.0)
            
            # Expand dimensions
            z_exp = z.unsqueeze(1)
            means_exp = self.means.unsqueeze(0)
            vars_exp = torch.exp(clamped_logvars).unsqueeze(0) + self.reg_value
            
            # Mahalanobis distance
            diff = z_exp - means_exp
            d_sq = torch.sum((diff ** 2) / vars_exp, dim=2)
            
            # Probability densities
            logvars_sum = torch.sum(torch.log(vars_exp.squeeze(0)), dim=1, keepdim=True).T
            log_pdf = -0.5 * (D * math.log(2 * math.pi) + logvars_sum + d_sq)
            
        else:  # "full" (Cholesky parametrization)
            L_tril = torch.tril(self.L_params, diagonal=-1)
            diag_val = torch.diagonal(self.L_params, dim1=1, dim2=2)
            clamped_diag = torch.clamp(diag_val, min=-6.0, max=1.0)
            # Regularize diagonal
            diag_reg = torch.exp(clamped_diag) + self.reg_value
            L = L_tril + torch.diag_embed(diag_reg)
            
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

    def update_statistics(self, z, q_y):
        """
        Original IGMN Online Recursive Parameter Updates:
        sp_k <- sp_k + sum(w_k)
        v_k  <- v_k + B
        mu_k <- (1 - alpha_k) * mu_k + alpha_k * z_bar_k  where alpha_k = sum(w_k) / sp_k
        """
        if not self.training:
            return
        with torch.no_grad():
            B = z.shape[0]
            resp = q_y[:, :self.K]
            z_det = z.detach()
            for k in range(self.K):
                w_k = resp[:, k]
                sum_w = torch.sum(w_k)
                if sum_w > 1e-4:
                    old_sp = self.sp[k]
                    new_sp = old_sp + sum_w
                    self.sp[k] = new_sp
                    self.v[k] += B
                    
                    weighted_z = torch.sum(w_k.unsqueeze(1) * z_det, dim=0)
                    lr_mu = sum_w / (new_sp + 1e-5)
                    self.means.data[k] = (1.0 - lr_mu) * self.means.data[k] + lr_mu * (weighted_z / sum_w)

    def spawn_component(self, new_mean, new_logvar=None):
        if self.K >= self.max_nc:
            return False
            
        device = self.means.device
        new_mean = new_mean.detach().to(device).view(1, self.latent_dim)
        means_new = torch.cat([self.means.data, new_mean], dim=0)
        
        if self.covariance_type == "diagonal":
            if new_logvar is None:
                init_logvar = 2.0 * math.log(self.delta)
                new_logvar = (torch.ones(1, self.latent_dim) * init_logvar).to(device)
            else:
                new_logvar = new_logvar.detach().to(device).view(1, self.latent_dim)
            self.logvars = nn.Parameter(torch.cat([self.logvars.data, new_logvar], dim=0))
        else:  # "full"
            new_L = (self.delta * torch.eye(self.latent_dim)).to(device).unsqueeze(0)
            self.L_params = nn.Parameter(torch.cat([self.L_params.data, new_L], dim=0))
            
        # Update prior logits
        pi_current = self.pi.data
        new_pi = torch.tensor([0.05]).to(device)
        pi_new = torch.cat([pi_current * 0.95, new_pi], dim=0)
        logits_new = torch.log(pi_new + 1e-8)
        
        self.means = nn.Parameter(means_new)
        self.pi_logits = nn.Parameter(logits_new)
        
        # Initialize accumulators for new component
        v_new = torch.tensor([1.0], device=device)
        sp_new = torch.tensor([1.0], device=device)
        self.v = torch.cat([self.v, v_new])
        self.sp = torch.cat([self.sp, sp_new])
        
        self.K += 1
        return True

    def prune_components(self, threshold=None):
        """
        Original IGMN Pruning rule:
        A component is only pruned if its age v_k >= v_min (grace period) AND its activation sp_k < sp_min.
        """
        if self.K <= 2:
            return False
            
        # Eligible only if mature
        is_mature = self.v >= self.v_min
        
        if threshold is not None:
            # Threshold on relative activation / prior
            pi_vals = self.pi.data
            is_noise = pi_vals < threshold
        else:
            is_noise = self.sp < self.sp_min
            
        prune_mask = is_mature & is_noise
        keep_indices = (~prune_mask).nonzero(as_tuple=True)[0]
        
        if len(keep_indices) < self.K and len(keep_indices) >= 2:
            old_K = self.K
            self.means = nn.Parameter(self.means.data[keep_indices])
            
            if self.covariance_type == "diagonal":
                self.logvars = nn.Parameter(self.logvars.data[keep_indices])
            else:
                self.L_params = nn.Parameter(self.L_params.data[keep_indices])
                
            pi_vals = self.pi.data
            pi_new = pi_vals[keep_indices]
            pi_new = pi_new / pi_new.sum()
            self.pi_logits = nn.Parameter(torch.log(pi_new + 1e-8))
            
            self.v = self.v[keep_indices]
            self.sp = self.sp[keep_indices]
            
            self.K = len(keep_indices)
            print(f"\n[Pruning] Removed {old_K - self.K} noisy components (v >= {self.v_min}, sp < {self.sp_min}). Remaining: {self.K}")
            return True
            
        return False

    def get_chi2_threshold(self, p_val=0.03):
        """
        Analytical Chi-Squared core-overlap critical value for latent dimension D.
        p_val represents the lower-tail percentile: two components overlap if their
        Mahalanobis distance d_M^2 is smaller than the inner core critical value.
        """
        try:
            return float(stats.chi2.ppf(p_val, df=self.latent_dim))
        except Exception:
            # Wilson-Hilferty approximation for lower-tail quantiles
            z = -1.28155 if p_val <= 0.10 else -1.64485
            k = self.latent_dim
            return max(0.5, float(k * (1.0 - 2.0 / (9.0 * k) + z * math.sqrt(2.0 / (9.0 * k))) ** 3))

    def merge_components(self, alpha=0.03, eta_merge=None, merge_dist_threshold=None):
        """
        Analytical Mahalanobis Chi-Squared Overlap Merging:
        Calculates the joint-covariance Mahalanobis distance between all pairs of Gaussians:
        d_M^2(mu_i, mu_j) = (mu_i - mu_j)^T ((Sigma_i + Sigma_j)/2)^(-1) (mu_i - mu_j).
        Merges components if d_M^2 < chi2_D(1 - alpha).
        """
        if self.K <= 2:
            return False
            
        device = self.means.device
        means = self.means.data
        K = self.K
        D = self.latent_dim
        
        if eta_merge is None:
            eta_merge = self.get_chi2_threshold(p_val=alpha)
            
        # 1. Compute full or diagonal covariances
        if self.covariance_type == "diagonal":
            variances = torch.exp(self.logvars.data)  # (K, D)
            dists = torch.full((K, K), 1e9, device=device)
            for i in range(K):
                for j in range(i + 1, K):
                    joint_var = 0.5 * (variances[i] + variances[j]) + 1e-4
                    diff = means[i] - means[j]
                    d_sq = torch.sum((diff ** 2) / joint_var).item()
                    dists[i, j] = d_sq
                    dists[j, i] = d_sq
        else:
            # Full Covariance via Cholesky factor L_params
            L_tril = torch.tril(self.L_params.data, diagonal=-1)
            diag_val = torch.diagonal(self.L_params.data, dim1=1, dim2=2)
            clamped_diag = torch.clamp(diag_val, min=-3.0, max=-0.2)
            L = L_tril + torch.diag_embed(torch.exp(clamped_diag))
            Sigmas = torch.matmul(L, L.transpose(1, 2)) + 1e-4 * torch.eye(D, device=device).unsqueeze(0)
            
            dists = torch.full((K, K), 1e9, device=device)
            for i in range(K):
                for j in range(i + 1, K):
                    joint_sigma = 0.5 * (Sigmas[i] + Sigmas[j]) + 1e-4 * torch.eye(D, device=device)
                    diff = (means[i] - means[j]).unsqueeze(1)
                    try:
                        L_joint = torch.linalg.cholesky(joint_sigma)
                        v = torch.linalg.solve_triangular(L_joint, diff, upper=False)
                        d_sq = torch.sum(v ** 2).item()
                    except Exception:
                        d_sq = torch.norm(means[i] - means[j]).item() ** 2
                    dists[i, j] = d_sq
                    dists[j, i] = d_sq
                    
        min_val, min_idx = torch.min(dists.view(-1), dim=0)
        
        # Check threshold (Euclidean fallback if merge_dist_threshold is explicitly given, otherwise Mahalanobis chi2)
        should_merge = (min_val.item() < merge_dist_threshold) if merge_dist_threshold is not None else (min_val.item() < eta_merge)
        
        if should_merge:
            i = (min_idx.item() // K)
            j = (min_idx.item() % K)
            
            sp_i = self.sp[i]
            sp_j = self.sp[j]
            total_sp = sp_i + sp_j + 1e-5
            
            # 1. Update Mean (Support-weighted combination)
            mu_i_old = self.means.data[i].clone()
            mu_j_old = self.means.data[j].clone()
            merged_mu = (sp_i * mu_i_old + sp_j * mu_j_old) / total_sp
            self.means.data[i] = merged_mu
            self.sp[i] = total_sp
            self.v[i] = max(self.v[i], self.v[j])
            
            # 2. Update Covariance (Within-cluster + Between-cluster scatter)
            if self.covariance_type == "diagonal":
                var_i = variances[i]
                var_j = variances[j]
                diff_sq = (mu_i_old - mu_j_old) ** 2
                merged_var = (sp_i * var_i + sp_j * var_j) / total_sp + (sp_i * sp_j / (total_sp ** 2)) * diff_sq
                self.logvars.data[i] = torch.log(torch.clamp(merged_var, min=1e-4, max=10.0))
            else:
                sigma_i = Sigmas[i]
                sigma_j = Sigmas[j]
                diff_mu = (mu_i_old - mu_j_old).unsqueeze(1)
                scatter = torch.matmul(diff_mu, diff_mu.T)
                merged_sigma = (sp_i * sigma_i + sp_j * sigma_j) / total_sp + (sp_i * sp_j / (total_sp ** 2)) * scatter
                merged_sigma = merged_sigma + 1e-4 * torch.eye(D, device=device)
                try:
                    merged_L = torch.linalg.cholesky(merged_sigma)
                    self.L_params.data[i] = merged_L
                except Exception:
                    pass
            
            # 3. Remove component j
            keep_indices = [idx for idx in range(K) if idx != j]
            keep = torch.tensor(keep_indices, device=device, dtype=torch.long)
            
            self.means = nn.Parameter(self.means.data[keep])
            if self.covariance_type == "diagonal":
                self.logvars = nn.Parameter(self.logvars.data[keep])
            else:
                self.L_params = nn.Parameter(self.L_params.data[keep])
                
            pi_vals = self.pi.data[keep]
            pi_new = pi_vals / pi_vals.sum()
            self.pi_logits = nn.Parameter(torch.log(pi_new + 1e-8))
            
            self.v = self.v[keep]
            self.sp = self.sp[keep]
            self.K = len(keep_indices)
            print(f"\n[Mahalanobis χ² Merge] Overlapping components ({i}, {j}) → Remaining K = {self.K} (d_M² = {min_val.item():.2f} < η = {eta_merge:.2f})")
            return True
        return False

    def set_K(self, new_K, device=None):
        """Dynamically resize all parameters and statistical buffers to match new_K."""
        if device is None:
            device = self.means.device
        self.K = new_K
        self.means = nn.Parameter(torch.zeros(new_K, self.latent_dim, device=device))
        if self.covariance_type == "diagonal":
            self.logvars = nn.Parameter(torch.zeros(new_K, self.latent_dim, device=device))
        else:
            self.L_params = nn.Parameter(torch.zeros(new_K, self.latent_dim, self.latent_dim, device=device))
        self.pi_logits = nn.Parameter(torch.zeros(new_K, device=device))
        self.v = torch.zeros(new_K, device=device)
        self.sp = torch.zeros(new_K, device=device)
