import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA as sklearn_PCA
from scipy.stats import chi2
import argparse
import math
import os

from model import GMVAE

def compute_loss(recon_x, x, q_mean, q_logvar, q_y, prior, sparsity_loss, device):
    B, D = q_mean.shape
    K = prior.K
    
    q_logvar = torch.clamp(q_logvar, min=-10.0, max=10.0)
    
    # 1. Reconstruction Loss
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # 2. KL Divergence for z
    q_mean_exp = q_mean.unsqueeze(1)
    q_var_exp = torch.exp(q_logvar).unsqueeze(1)
    
    if prior.covariance_type == "diagonal":
        clamped_prior_logvars = torch.clamp(prior.logvars, min=-5.0, max=5.0)
        p_mean_exp = prior.means.unsqueeze(0)
        p_var_exp = torch.exp(clamped_prior_logvars).unsqueeze(0)
        p_logvar_exp = clamped_prior_logvars.unsqueeze(0)
        
        kl_element = 0.5 * (
            p_logvar_exp - q_logvar.unsqueeze(1) - 1.0 + 
            (q_var_exp + (q_mean_exp - p_mean_exp) ** 2) / (p_var_exp + 1e-8)
        )
        kl_k = torch.sum(kl_element, dim=2)
    else:  # "full"
        L_tril = torch.tril(prior.L_params, diagonal=-1)
        diag_val = torch.diagonal(prior.L_params, dim1=1, dim2=2)
        clamped_diag = torch.clamp(diag_val, min=-3.0, max=0.0)
        L = L_tril + torch.diag_embed(torch.exp(clamped_diag))
        
        kl_k_list = []
        for k in range(K):
            L_k = L[k]
            log_det_p = 2.0 * torch.sum(torch.log(torch.diagonal(L_k) + 1e-8))
            
            diff_mean = q_mean - prior.means[k].unsqueeze(0)
            v_mean = torch.linalg.solve_triangular(L_k, diff_mean.T, upper=False)
            mahalanobis_term = torch.sum(v_mean ** 2, dim=0)
            
            I = torch.eye(D).to(device)
            M = torch.linalg.solve_triangular(L_k, I, upper=False)
            diag_inv_cov = torch.sum(M ** 2, dim=0)
            
            q_var = torch.exp(q_logvar)
            trace_term = torch.sum(q_var * diag_inv_cov.unsqueeze(0), dim=1)
            
            log_det_q = torch.sum(q_logvar, dim=1)
            
            kl_k_val = 0.5 * (log_det_p - log_det_q - D + trace_term + mahalanobis_term)
            kl_k_list.append(kl_k_val)
            
        kl_k = torch.stack(kl_k_list, dim=1)
        
    # New component prior N(0, I)
    kl_new = -0.5 * torch.sum(1.0 + q_logvar - q_mean ** 2 - torch.exp(q_logvar), dim=1)
    
    kl_all = torch.cat([kl_k, kl_new.unsqueeze(1)], dim=1)
    kl_z = torch.sum(q_y * kl_all, dim=1).sum()
    
    # 3. KL Divergence for y
    p_new_prior = 0.05
    pi_existing = prior.pi * (1.0 - p_new_prior)
    p_y_prior = torch.cat([pi_existing, torch.tensor([p_new_prior]).to(device)])
    p_y_prior = p_y_prior.unsqueeze(0)
    
    kl_y = torch.sum(q_y * (torch.log(q_y + 1e-8) - torch.log(p_y_prior + 1e-8)), dim=1).sum()
    
    total_loss = (recon_loss + kl_z + kl_y) / B + sparsity_loss
    
    return total_loss, recon_loss / B, kl_z / B, kl_y / B


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.view(-1, 784).to(device)
            recon_x, z, q_mean, q_logvar, q_y, d_sq, p_new, active_dims, sparsity_loss = model(x)
            loss, _, _, _ = compute_loss(recon_x, x, q_mean, q_logvar, q_y, model.prior, sparsity_loss, device)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def plot_latent_space(model, dataloader, device, filename="latent_space.png"):
    model.eval()
    all_z = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.view(-1, 784).to(device)
            _, z, _, _, _, _, _, _, _ = model(x)
            all_z.append(z.cpu().numpy())
            all_labels.append(y.numpy())
            if len(all_z) * x.size(0) >= 3000:
                break
                
    z_pts = np.concatenate(all_z, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    means = model.prior.means.data.cpu().numpy()
    latent_dim = z_pts.shape[1]
    if latent_dim > 2:
        print(f"Projecting {latent_dim}D space to 2D for plotting...")
        pca_proj = sklearn_PCA(n_components=2)
        z_pts_2d = pca_proj.fit_transform(z_pts)
        means_2d = pca_proj.transform(means)
        title_extra = f" (Projected {latent_dim}D -> 2D via PCA)"
    else:
        z_pts_2d = z_pts
        means_2d = means
        title_extra = ""
        
    plt.figure(figsize=(10, 8), facecolor='#121212')
    ax = plt.gca()
    ax.set_facecolor('#1e1e1e')
    
    scatter = ax.scatter(z_pts_2d[:, 0], z_pts_2d[:, 1], c=labels, cmap='tab10', s=10, alpha=0.6, edgecolors='none')
    
    # Plot means
    for k in range(len(means_2d)):
        ax.plot(means_2d[k, 0], means_2d[k, 1], 'x', color='white', markersize=10, markeredgewidth=2)
        ax.text(means_2d[k, 0] + 0.1, means_2d[k, 1] + 0.1, f"C{k}", color='white', fontsize=12, fontweight='bold')
        
    plt.title(f"GMVAE Latent Space - {len(means)} Clusters Found{title_extra}", color='white', fontsize=14)
    plt.xlabel("Component 1", color='white')
    plt.ylabel("Component 2", color='white')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label("True MNIST Digit", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.yaxis.label.set_color('white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    plt.tight_layout()
    plt.savefig(filename, facecolor=plt.gcf().get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Saved plot to {filename}")


def main():
    parser = argparse.ArgumentParser(description="GMVAE + Differentiable IGMN MNIST Example")
    parser.add_argument("--dim_method", type=str, default="none", choices=["none", "ard", "kl-pruning", "pca"],
                        help="Dynamic dimension selection method")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--latent_dim", type=int, default=10, help="Initial maximum latent dimension")
    parser.add_argument("--ard_lambda", type=float, default=1e-3, help="Sparsity penalty weight for ARD")
    parser.add_argument("--var_threshold", type=float, default=0.05, help="Variance collapse threshold")
    parser.add_argument("--pca_threshold", type=float, default=0.01, help="Eigenvalue percentage threshold for PCA")
    parser.add_argument("--covariance_type", type=str, default="full", choices=["full", "diagonal"],
                        help="GMM Covariance type for Differentiable IGMN")
    args = parser.parse_args()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    if args.covariance_type == "full" and device.type == "mps":
        print("[Info] MPS autograd has numerical issues with solve_triangular backpropagation. Falling back to CPU for stability.")
        device = torch.device("cpu")
        
    print(f"--- Running GMVAE + Differentiable IGMN ---")
    print(f"Method: {args.dim_method}")
    print(f"Initial Latent Dim: {args.latent_dim}")
    print(f"Covariance Type: {args.covariance_type}")
    print(f"Device: {device}")
    
    # Load MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Dynamic Chi-squared Thresholding for Eta initialization
    initial_eta = chi2.ppf(0.99, args.latent_dim)
    print(f"Initial Chi-squared Eta threshold (99% confidence for {args.latent_dim}D): {initial_eta:.2f}")
    
    model = GMVAE(
        input_dim=784,
        hidden_dim=512,  # Set wide capacity hidden dimension
        latent_dim=args.latent_dim,
        initial_K=2,
        beta=0.5,
        eta=initial_eta,
        dim_method=args.dim_method,
        ard_lambda=args.ard_lambda,
        var_threshold=args.var_threshold,
        pca_threshold=args.pca_threshold,
        covariance_type=args.covariance_type
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    step_count = 0
    spawn_cooldown = 150
    last_spawn_step = -spawn_cooldown
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 2
    allow_spawning = True
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl_z = 0
        total_kl_y = 0
        active_dims_sum = 0
        
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, 784).to(device)
            
            # Forward
            recon_x, z, q_mean, q_logvar, q_y, d_sq, p_new, active_dims, sparsity_loss = model(x)
            
            # Loss
            loss, recon, kl_z, kl_y = compute_loss(recon_x, x, q_mean, q_logvar, q_y, model.prior, sparsity_loss, device)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            has_nan = False
            for p in model.parameters():
                if torch.isnan(p).any():
                    has_nan = True
                    break
            if has_nan:
                print("\n[Warning] NaN detected in model parameters! Aborting batch.")
                break
                
            total_loss += loss.item()
            total_recon += recon.item()
            total_kl_z += kl_z.item()
            total_kl_y += kl_y.item()
            active_dims_sum += active_dims
            
            # Dynamic eta recalculation
            current_active_dim = max(1.0, active_dims)
            dynamic_eta = chi2.ppf(0.99, current_active_dim)
            model.prior.eta = dynamic_eta
            
            step_count += 1
            
            # IGMN Dynamic component spawning
            if allow_spawning and (step_count - last_spawn_step >= spawn_cooldown):
                max_p_new, max_idx = torch.max(p_new, dim=0)
                if max_p_new.item() > 0.85 and model.prior.K < 10:
                    new_mean = z[max_idx].detach()
                    model.prior.spawn_component(new_mean)
                    last_spawn_step = step_count
                    print(f"\n[Step {step_count}] Spawning cluster {model.prior.K} at {new_mean.cpu().numpy()} (Current active dims: {active_dims:.1f}, eta: {dynamic_eta:.2f})")
                    
                    # Recreate optimizer and scheduler preserving current learning rate
                    current_lr = optimizer.param_groups[0]['lr']
                    optimizer = optim.Adam(model.parameters(), lr=current_lr)
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
                    
        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon / len(train_loader)
        avg_kl_z = total_kl_z / len(train_loader)
        avg_kl_y = total_kl_y / len(train_loader)
        avg_active_dims = active_dims_sum / len(train_loader)
        
        # Evaluate validation loss
        val_loss = evaluate(model, test_loader, device)
        
        # Step LR scheduler based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {avg_loss:.2f} (Recon: {avg_recon:.2f}) | Val Loss: {val_loss:.2f} | LR: {current_lr:.6f} | Clusters: {model.prior.K} | Active Dims: {avg_active_dims:.1f}")
        
        # Validation loss early-stopping for spawning (optimize K)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience and allow_spawning:
                print(f"[Optimize K] Validation loss plateaud. Freezing GMM spawning at K={model.prior.K}.")
                allow_spawning = False
        
        # End of epoch: prune components with low prior weight (outliers)
        if model.prior.prune_components(threshold=0.03):
            current_lr = optimizer.param_groups[0]['lr']
            optimizer = optim.Adam(model.parameters(), lr=current_lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
        
    plot_latent_space(model, test_loader, device, filename=f"/Users/mazzutti/Downloads/IGMNVae/latent_space_{args.dim_method}.png")

if __name__ == "__main__":
    main()
