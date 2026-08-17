import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA as sklearn_PCA
from scipy.stats import chi2
import io
from PIL import Image
import math
import os

from model import GMVAE
from train import evaluate

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

covariance_type = "full"
if covariance_type == "full" and device.type == "mps":
    device = torch.device("cpu")
print("Using device:", device)

def compute_loss_balanced(recon_x, x, q_mean, q_logvar, q_y, prior, sparsity_loss, device):
    B, D = q_mean.shape
    K = prior.K
    
    q_logvar = torch.clamp(q_logvar, min=-10.0, max=10.0)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
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
        
    kl_new = -0.5 * torch.sum(1.0 + q_logvar - q_mean ** 2 - torch.exp(q_logvar), dim=1)
    kl_all = torch.cat([kl_k, kl_new.unsqueeze(1)], dim=1)
    kl_z = torch.sum(q_y * kl_all, dim=1).sum()
    
    p_new_prior = 0.05
    pi_existing = prior.pi * (1.0 - p_new_prior)
    p_y_prior = torch.cat([pi_existing, torch.tensor([p_new_prior]).to(device)])
    p_y_prior = p_y_prior.unsqueeze(0)
    
    kl_y = torch.sum(q_y * (torch.log(q_y + 1e-8) - torch.log(p_y_prior + 1e-8)), dim=1).sum()
    
    # Prior Entropy Regularization
    pi_clean = prior.pi
    entropy_pi = -torch.sum(pi_clean * torch.log(pi_clean + 1e-8))
    entropy_penalty = -2.0 * entropy_pi
    
    total_loss = (recon_loss + 0.5 * kl_z + 0.5 * kl_y) / B + sparsity_loss + entropy_penalty
    
    return total_loss, recon_loss / B, kl_z / B, kl_y / B


def evaluate_balanced(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.view(-1, 784).to(device)
            recon_x, z, q_mean, q_logvar, q_y, d_sq, p_new, active_dims, sparsity_loss = model(x)
            loss, _, _, _ = compute_loss_balanced(recon_x, x, q_mean, q_logvar, q_y, model.prior, sparsity_loss, device)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def get_plot_frame(model, test_loader, device, step, epoch, val_loss, allow_spawning, lr):
    model.eval()
    all_z = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.view(-1, 784).to(device)
            _, z, _, _, _, _, _, _, _ = model(x)
            all_z.append(z.cpu().numpy())
            all_labels.append(y.numpy())
            if len(all_z) * x.size(0) >= 1500:
                break
                
    z_pts = np.concatenate(all_z, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    means = model.prior.means.data.cpu().numpy()
    latent_dim = z_pts.shape[1]
    
    # Predict clusters for empirical ellipses using 10D Euclidean distance to GMM centroids
    distances_to_centroids = np.linalg.norm(z_pts[:, None, :] - means[None, :, :], axis=2)
    predicted_clusters = np.argmin(distances_to_centroids, axis=1)
        
    # Fit t-SNE to project 10D space to 2D
    from sklearn.manifold import TSNE
    combined_pts = np.concatenate([z_pts, means], axis=0)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    combined_tsne = tsne.fit_transform(combined_pts)
    
    # Normalize coordinates to $[-4, 4]$
    tsne_min = combined_tsne.min(axis=0)
    tsne_max = combined_tsne.max(axis=0)
    combined_tsne_norm = -4.0 + 8.0 * (combined_tsne - tsne_min) / (tsne_max - tsne_min + 1e-8)
    
    z_pts_2d = combined_tsne_norm[:-len(means)]
    means_2d = combined_tsne_norm[-len(means):]
    
    fig = plt.figure(figsize=(8, 6), facecolor='#121212')
    ax = plt.gca()
    ax.set_facecolor('#1e1e1e')
    
    # Scatter plot in projected space
    scatter = ax.scatter(z_pts_2d[:, 0], z_pts_2d[:, 1], c=labels, cmap='tab10', s=5, alpha=0.5, edgecolors='none')
    
    # Draw GMM ellipses using empirical covariance of projected points in 2D
    emp_means_list = {}
    for k in range(len(means)):
        cluster_points = z_pts_2d[predicted_clusters == k]
        if len(cluster_points) >= 1:
            if len(cluster_points) > 1:
                emp_mean = np.mean(cluster_points, axis=0)
                emp_means_list[k] = emp_mean
                emp_cov = np.cov(cluster_points.T)
                eigenvalues, eigenvectors = np.linalg.eigh(emp_cov)
                order = eigenvalues.argsort()[::-1]
                eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
                angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
                width, height = 4 * np.sqrt(np.clip(eigenvalues, a_min=1e-8, a_max=None))
            else:
                emp_mean = cluster_points[0]
                emp_means_list[k] = emp_mean
                angle = 0
                width, height = 0.5, 0.5
            
            # Plot centroid marker at emp_mean
            ax.plot(emp_mean[0], emp_mean[1], 'x', color='white', markersize=8, markeredgewidth=2)
            
            ellipse = Ellipse(
                xy=(emp_mean[0], emp_mean[1]),
                width=width,
                height=height,
                angle=angle,
                edgecolor='white',
                fc='none',
                lw=1.5,
                ls='--',
                alpha=0.7
            )
            ax.add_patch(ellipse)
        else:
            emp_means_list[k] = means_2d[k]
            ax.plot(means_2d[k, 0], means_2d[k, 1], 'x', color='white', markersize=8, markeredgewidth=2)

    # Decode and overlay digit predictions right above the GMM cluster centers!
    with torch.no_grad():
        for k in range(len(means)):
            # Pass cluster mean (in 10D) to the Decoder
            mu_k_tensor = model.prior.means[k].unsqueeze(0)
            recon_digit = model.decode(mu_k_tensor)
            digit_img = recon_digit.squeeze().view(28, 28).data.cpu().numpy()
            digit_img = np.clip(digit_img, 0.0, 1.0)
            
            # Create a transparent RGBA image where digit glows in gold/yellow
            rgba = np.zeros((28, 28, 4))
            rgba[:, :, 0] = 1.0  # Red
            rgba[:, :, 1] = 0.9  # Green
            rgba[:, :, 2] = 0.1  # Blue
            rgba[:, :, 3] = digit_img  # Alpha is pixel intensity
            
            # Get empirical mean coordinates for this cluster
            emp_mean = emp_means_list.get(k, means_2d[k])
            
            im = OffsetImage(rgba, zoom=0.7)
            # Offset the floating digit slightly above the center marker (emp_mean[1] + 0.5)
            ab = AnnotationBbox(im, (emp_mean[0], emp_mean[1] + 0.5), xycoords='data', frameon=False)
            ax.add_artist(ab)
            
    status_str = "Spawning: Active" if allow_spawning else "Spawning: Frozen"
    val_loss_str = f"Val Loss: {val_loss:.2f}" if val_loss is not None else "Val Loss: N/A"
    lr_str = f"LR: {lr:.5f}" if lr is not None else "LR: N/A"
    plt.title(f"Dynamic GMM Structuring | Epoch {epoch} | Step {step} | K={len(means)} Clusters\n{status_str} | {val_loss_str} | {lr_str} (Projected 10D -> 2D via t-SNE)", color='white', fontsize=11)
    
    xlim_min, xlim_max = np.percentile(z_pts_2d[:, 0], [1, 99])
    ylim_min, ylim_max = np.percentile(z_pts_2d[:, 1], [1, 99])
    plt.xlim(xlim_min - 1.0, xlim_max + 1.0)
    plt.ylim(ylim_min - 1.0, ylim_max + 1.0)
    ax.tick_params(colors='white')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    img.load()
    plt.close(fig)
    return img

def main():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    latent_dim = 10
    eta = 11.5
    
    model = GMVAE(
        input_dim=784,
        hidden_dim=512,
        latent_dim=latent_dim,
        initial_K=2,
        beta=0.5,
        eta=eta,
        dim_method="none",
        covariance_type=covariance_type
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    frames = []
    step_count = 0
    spawn_cooldown = 100
    last_spawn_step = -spawn_cooldown
    
    best_val_loss = float('inf')
    epochs_no_improve_spawning = 0
    epochs_no_improve_total = 0
    
    spawning_patience = 3
    early_stopping_patience = 5
    allow_spawning = True
    val_loss = None
    
    print("Capturing initial state...")
    frames.append(get_plot_frame(model, test_loader, device, step_count, 0, val_loss, allow_spawning, 1e-3))
    
    max_epochs = 25
    best_epoch = 0
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, 784).to(device)
            
            recon_x, z, q_mean, q_logvar, q_y, d_sq, p_new, _, _ = model(x)
            loss, _, _, _ = compute_loss_balanced(recon_x, x, q_mean, q_logvar, q_y, model.prior, 0.0, device)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step_count += 1
            
            # Dynamic Spawning
            if allow_spawning and (step_count - last_spawn_step >= spawn_cooldown):
                max_p_new, max_idx = torch.max(p_new, dim=0)
                if max_p_new.item() > 0.80 and model.prior.K < 12:
                    new_mean = z[max_idx].detach()
                    model.prior.spawn_component(new_mean)
                    last_spawn_step = step_count
                    print(f"Step {step_count}: Spawned Cluster {model.prior.K}")
                    
                    current_lr = optimizer.param_groups[0]['lr']
                    optimizer = optim.Adam(model.parameters(), lr=current_lr)
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
            
            # Save frame every 100 steps
            if step_count % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                frames.append(get_plot_frame(model, test_loader, device, step_count, epoch, val_loss, allow_spawning, current_lr))
                
        # Evaluate validation loss
        val_loss = evaluate_balanced(model, test_loader, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch} | Val Loss: {val_loss:.2f} | LR: {current_lr:.6f} | K={model.prior.K} (Best Val Loss: {best_val_loss:.2f})")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve_spawning = 0
            epochs_no_improve_total = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            epochs_no_improve_spawning += 1
            epochs_no_improve_total += 1
            
            if epochs_no_improve_spawning >= spawning_patience and allow_spawning:
                print(f"[Optimize K] Validation loss plateaud. Freezing GMM spawning at K={model.prior.K}.")
                allow_spawning = False
                
            if epochs_no_improve_total >= early_stopping_patience:
                print(f"\n[Early Stopping] Lowest validation loss reached at epoch {best_epoch}. Stopping.")
                break
                
        # Lower pruning threshold
        if model.prior.prune_components(threshold=0.005):
            current_lr = optimizer.param_groups[0]['lr']
            optimizer = optim.Adam(model.parameters(), lr=current_lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
            
    # Load the best model parameters
    checkpoint = torch.load("best_model.pt")
    best_K = checkpoint["prior.means"].shape[0]
    
    if model.prior.K != best_K:
        print(f"Re-shaping GMM parameters to match checkpoint size K={best_K}")
        model.prior.means = nn.Parameter(torch.zeros(best_K, model.prior.latent_dim).to(device))
        if model.prior.covariance_type == "diagonal":
            model.prior.logvars = nn.Parameter(torch.zeros(best_K, model.prior.latent_dim).to(device))
        else:
            model.prior.L_params = nn.Parameter(torch.zeros(best_K, model.prior.latent_dim, model.prior.latent_dim).to(device))
        model.prior.pi_logits = nn.Parameter(torch.zeros(best_K).to(device))
        model.prior.K = best_K
        
    model.load_state_dict(checkpoint)
    print(f"Loaded best model from epoch {best_epoch} with Val Loss {best_val_loss:.2f}")
    
    # Save the final animation GIF
    gif_path = "/Users/mazzutti/Downloads/IGMNVae/training_evolution.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=300,
        loop=0
    )
    print(f"Complete training evolution animation saved to {gif_path}")

if __name__ == "__main__":
    main()
