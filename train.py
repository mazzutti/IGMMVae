import argparse
import os
import io
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.manifold import TSNE
from scipy.stats import chi2
from PIL import Image

from model import GMVAE

def get_plot_frame(model, test_loader, device, step, epoch, val_loss, lr, action_text=""):
    model.eval()
    all_z = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.view(-1, 784).to(device)
            q_mean, _ = model.encode(x)
            all_z.append(q_mean.cpu().numpy())
            all_labels.append(y.numpy())
            if len(all_z) * x.size(0) >= 1000:
                break
                
    z_pts = np.concatenate(all_z, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    means = model.prior.means.data.cpu().numpy()
    latent_dim = z_pts.shape[1]
    K = len(means)
    
    combined_pts = np.concatenate([z_pts, means], axis=0)
    perplexity = min(30, max(5, len(combined_pts) // 15))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    combined_tsne = tsne.fit_transform(combined_pts)
    
    tsne_min = combined_tsne.min(axis=0)
    tsne_max = combined_tsne.max(axis=0)
    combined_tsne_norm = -4.0 + 8.0 * (combined_tsne - tsne_min) / (tsne_max - tsne_min + 1e-8)
    
    z_pts_2d = combined_tsne_norm[:-K]
    means_2d = combined_tsne_norm[-K:]
    
    dists_2d = np.linalg.norm(z_pts_2d[:, None, :] - means_2d[None, :, :], axis=2)
    assign_2d = np.argmin(dists_2d, axis=1)
    
    fig = plt.figure(figsize=(9, 7), facecolor='#121212')
    ax = plt.gca()
    ax.set_facecolor('#1e1e1e')
    
    scatter = ax.scatter(z_pts_2d[:, 0], z_pts_2d[:, 1], c=labels, cmap='tab10', s=6, alpha=0.55, edgecolors='none')
    
    emp_means_list = {}
    cmap_clusters = plt.colormaps.get_cmap('tab10').resampled(max(10, K))
    
    for k in range(K):
        cluster_points = z_pts_2d[assign_2d == k]
        color_k = cmap_clusters(k % 10)
        
        if len(cluster_points) > 1:
            emp_mean = np.mean(cluster_points, axis=0)
            emp_means_list[k] = emp_mean
            emp_cov = np.cov(cluster_points.T)
            eigenvalues, eigenvectors = np.linalg.eigh(emp_cov)
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            width, height = 3.0 * np.sqrt(np.clip(eigenvalues, a_min=1e-8, a_max=None))
        else:
            emp_mean = means_2d[k]
            emp_means_list[k] = emp_mean
            angle = 0
            width, height = 0.5, 0.5
        
        ax.plot(emp_mean[0], emp_mean[1], 'x', color='white', markersize=9, markeredgewidth=2, zorder=5)
        ax.text(emp_mean[0] + 0.12, emp_mean[1] + 0.12, f"C{k}", color='white', fontsize=9, fontweight='bold', zorder=5)
        
        ellipse_fill = Ellipse(
            xy=(emp_mean[0], emp_mean[1]),
            width=width,
            height=height,
            angle=angle,
            edgecolor=color_k,
            facecolor=color_k,
            lw=1.5,
            ls='--',
            alpha=0.18,
            zorder=3
        )
        ax.add_patch(ellipse_fill)
        ellipse_border = Ellipse(
            xy=(emp_mean[0], emp_mean[1]),
            width=width,
            height=height,
            angle=angle,
            edgecolor=color_k,
            facecolor='none',
            lw=1.5,
            ls='--',
            alpha=0.85,
            zorder=3
        )
        ax.add_patch(ellipse_border)
            
    val_loss_str = f"Loss: {val_loss:.2f}" if val_loss is not None else "Loss: N/A"
    status_str = f" | {action_text}" if action_text else ""
    plt.title(f"GMVAE + IGMM Evolution | Step {step:4d} | Epoch {epoch} | K={K} Clusters\n{val_loss_str}{status_str} (t-SNE 16D -> 2D)", color='white', fontsize=11)
    
    xlim_min, xlim_max = np.percentile(z_pts_2d[:, 0], [1, 99])
    ylim_min, ylim_max = np.percentile(z_pts_2d[:, 1], [1, 99])
    plt.xlim(xlim_min - 1.2, xlim_max + 1.2)
    plt.ylim(ylim_min - 1.2, ylim_max + 1.2)
    ax.tick_params(colors='white')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    img.load()
    plt.close(fig)
    return img

def main():
    parser = argparse.ArgumentParser(description="Train GMVAE with Differentiable IGMM Prior")
    parser.add_argument("--epochs", type=int, default=14, help="Number of training epochs")
    parser.add_argument("--latent_dim", type=int, default=16, help="Initial maximum latent dimension")
    parser.add_argument("--initial_K", type=int, default=2, help="Initial number of IGMM components")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--elbow_threshold", type=float, default=3.5,
                        help="Reconstruction marginal gain threshold (nats/cluster) for autonomous Elbow/Knee detection")
    parser.add_argument("--merge_dist", type=float, default=3.18,
                        help="Centroid separation distance threshold for agglomerative statistical merging")
    parser.add_argument("--min_spawn_dist", type=float, default=2.4,
                        help="Minimum latent distance to existing centroids for spawning a new cluster")
    parser.add_argument("--covariance_type", type=str, default="full", choices=["full", "diagonal"],
                        help="IGMM Covariance type for Differentiable IGMM")
    args = parser.parse_args()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    if args.covariance_type == "full" and device.type == "mps":
        print("[Info] MPS autograd has numerical issues with solve_triangular backpropagation. Falling back to CPU for stability.")
        device = torch.device("cpu")
        
    print(f"--- Running GMVAE + Differentiable IGMM (Autonomous Discovery) ---")
    print(f"Initial Latent Dim: {args.latent_dim}, Initial K: {args.initial_K}")
    print(f"Device: {device}")
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    spawn_cooldown = max(35, len(train_loader) // 10)
    print(f"Automatic Gradient-Adaptive Spawn Cooldown: {spawn_cooldown} steps")
    
    model = GMVAE(
        input_dim=784,
        hidden_dim=512,
        latent_dim=args.latent_dim,
        initial_K=args.initial_K,
        covariance_type=args.covariance_type
    ).to(device)
    
    ae_params = list(model.encoder_module.parameters()) + list(model.decoder_module.parameters()) + [model.prior.L_params]
    optimizer = optim.Adam(ae_params, lr=1.2e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    step_count = 0
    last_spawn_step = -spawn_cooldown
    allow_spawning = True
    prev_val_loss = None
    prev_K = args.initial_K
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, 784).to(device)
            B, D = x.shape[0], args.latent_dim
            
            recon_x, z, q_mean, q_logvar, q_y, d_sq, p_new, _, _ = model(x)
            
            log_resp = -0.5 * d_sq / 0.40
            sharp_resp = F.softmax(log_resp, dim=1)
            
            stroke_weight = 1.0 + 0.8 * x
            recon_bce = -(stroke_weight * x * torch.log(recon_x + 1e-8) + (1.0 - x) * torch.log(1.0 - recon_x + 1e-8)).sum(dim=1)
            recon_loss = recon_bce.mean()
            
            K = model.prior.K
            L_tril = torch.tril(model.prior.L_params, diagonal=-1)
            diag_val = torch.diagonal(model.prior.L_params, dim1=1, dim2=2)
            clamped_diag = torch.clamp(diag_val, min=-3.0, max=-0.2)
            L = L_tril + torch.diag_embed(torch.exp(clamped_diag))
            
            kl_list = []
            for k in range(K):
                L_k = L[k]
                log_det_p = 2.0 * torch.sum(torch.log(torch.diagonal(L_k) + 1e-8))
                diff_mean = q_mean - model.prior.means[k].unsqueeze(0)
                v_mean = torch.linalg.solve_triangular(L_k, diff_mean.T, upper=False)
                mahal_term = torch.sum(v_mean ** 2, dim=0)
                
                I = torch.eye(D).to(device)
                M = torch.linalg.solve_triangular(L_k, I, upper=False)
                diag_inv_cov = torch.sum(M ** 2, dim=0)
                q_var = torch.exp(q_logvar)
                trace_term = torch.sum(q_var * diag_inv_cov.unsqueeze(0), dim=1)
                log_det_q = torch.sum(q_logvar, dim=1)
                
                kl_val = 0.5 * (log_det_p - log_det_q - D + trace_term + mahal_term)
                kl_list.append(kl_val)
                
            kl_k = torch.stack(kl_list, dim=1)
            kl_z = torch.sum(sharp_resp * kl_k, dim=1).mean()
            
            avg_qy = torch.mean(sharp_resp, dim=0)
            balance_loss = torch.sum(avg_qy * (torch.log(avg_qy + 1e-8) - np.log(1.0 / K)))
            
            if K > 1:
                diffs = model.prior.means.unsqueeze(0) - model.prior.means.unsqueeze(1)
                dist_sq = torch.sum(diffs ** 2, dim=2)
                mask = 1.0 - torch.eye(K).to(device)
                repulsion = torch.clamp(12.0 - dist_sq, min=0.0)
                sep_loss = torch.sum(mask * repulsion) / (K * (K - 1))
            else:
                sep_loss = torch.tensor(0.0).to(device)
                
            loss = recon_loss + 0.20 * kl_z + 6.0 * balance_loss + 2.0 * sep_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae_params, max_norm=5.0)
            optimizer.step()
            
            with torch.no_grad():
                z_detached = q_mean.detach()
                for k in range(K):
                    w_k = sharp_resp[:, k]
                    sum_w = torch.sum(w_k)
                    if sum_w > 1e-4:
                        old_sp = model.prior.sp[k]
                        new_sp = old_sp + sum_w
                        model.prior.sp[k] = new_sp
                        model.prior.v[k] += B
                        
                        weighted_z = torch.sum(w_k.unsqueeze(1) * z_detached, dim=0)
                        lr_mu = sum_w / (new_sp + 1e-5)
                        model.prior.means.data[k] = (1.0 - lr_mu) * model.prior.means.data[k] + lr_mu * (weighted_z / sum_w)
                        
            step_count += 1
            
            if allow_spawning and (step_count - last_spawn_step >= spawn_cooldown):
                combined_novelty = p_new * (stroke_weight.mean(dim=1))
                max_score, max_idx = torch.max(combined_novelty, dim=0)
                if max_score.item() > 0.75:
                    new_mean = q_mean[max_idx].detach()
                    min_dist = torch.min(torch.norm(model.prior.means - new_mean.unsqueeze(0), dim=1))
                    if min_dist > args.min_spawn_dist:
                        model.prior.spawn_component(new_mean)
                        last_spawn_step = step_count
                        print(f"[Step {step_count}] Spawning cluster {model.prior.K} at {new_mean.cpu().numpy()[:3]}... (dist={min_dist:.2f})")
                        ae_params = list(model.encoder_module.parameters()) + list(model.decoder_module.parameters()) + [model.prior.L_params]
                        current_lr = optimizer.param_groups[0]['lr']
                        optimizer = optim.Adam(ae_params, lr=current_lr)
                        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Validation & Elbow Detection
        model.eval()
        with torch.no_grad():
            test_x, test_y = next(iter(test_loader))
            test_x = test_x.view(-1, 784).to(device)
            t_recon, _, t_mean, _, _, _, _, _, _ = model(test_x)
            val_loss = F.binary_cross_entropy(t_recon, test_x, reduction='sum').item() / test_x.shape[0]
            
        scheduler.step(val_loss)
        current_K = model.prior.K
        print(f"Epoch {epoch:2d}/{args.epochs} | Train Loss: {avg_loss:.2f} | Val Loss: {val_loss:.2f} | LR: {current_lr:.6f} | Clusters: {current_K}")
        
        if prev_val_loss is not None and allow_spawning:
            delta_K = current_K - prev_K
            delta_loss = prev_val_loss - val_loss
            if delta_K > 0:
                gain_per_k = delta_loss / delta_K
                print(f"  [Elbow Metric] Marginal Gain = {gain_per_k:.2f} nats/cluster (Delta K={delta_K}, Delta Loss={delta_loss:.2f})")
                if gain_per_k < args.elbow_threshold and current_K >= 8:
                    print(f"  [Elbow Knee Detected!] Freezing spawning at optimal K={current_K}.")
                    allow_spawning = False
                    
        prev_val_loss = val_loss
        prev_K = current_K
        
        # Merge overlapping components during active exploration
        if epoch <= 6:
            while model.prior.merge_components(merge_dist_threshold=args.merge_dist):
                ae_params = list(model.encoder_module.parameters()) + list(model.decoder_module.parameters()) + [model.prior.L_params]
                current_lr = optimizer.param_groups[0]['lr']
                optimizer = optim.Adam(ae_params, lr=current_lr)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
                
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  ==> Saved best model checkpoint to best_model.pt (Val Loss: {best_val_loss:.2f})")
            
    print(f"\nTraining complete! Optimal Discovered K: {model.prior.K}")

if __name__ == "__main__":
    main()
