import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.manifold import TSNE
import os

from model import GMVAE

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
# CPU fallback for full covariance stability
if device.type == "mps":
    device = torch.device("cpu")
print("Using device for inference:", device)

def main():
    checkpoint = torch.load("best_model.pt", map_location=device)
    latent_dim = checkpoint["prior.means"].shape[1]
    best_K = checkpoint["prior.means"].shape[0]
    
    # Load VAE model
    model = GMVAE(
        input_dim=784,
        hidden_dim=512,
        latent_dim=latent_dim,
        initial_K=best_K,
        covariance_type="full"
    ).to(device)
    
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Set seeds for 100% exact reproducibility across all scripts/video
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load batch of test data
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1500, shuffle=False)
    x, y = next(iter(test_loader))
    x = x.view(-1, 784).to(device)
    
    # Encode test data (use deterministic mean q_mean without random sampling noise)
    with torch.no_grad():
        q_mean, _ = model.encode(x)
    z_np = q_mean.cpu().numpy()
    y_np = y.numpy()
    means_latent = model.prior.means.data.cpu().numpy()

    # Get Cholesky covariance factors for all IGMM components
    L_tril = torch.tril(model.prior.L_params, diagonal=-1)
    diag_val = torch.diagonal(model.prior.L_params, dim1=1, dim2=2)
    clamped_diag = torch.clamp(diag_val, min=-3.0, max=0.0)
    L = L_tril + torch.diag_embed(torch.exp(clamped_diag))
    L_np = L.data.cpu().numpy()

    print("Fitting PCA (Linear Projection)...")
    pca = sklearn_PCA(n_components=2)
    z_pca = pca.fit_transform(z_np)
    means_pca = pca.transform(means_latent)
    W = pca.components_.T

    print("Fitting t-SNE (Topological Projection)...")
    # Project real data and IGMM centroids together (NO synthetic samples, which warp t-SNE manifold)
    combined_pts = np.concatenate([z_np, means_latent], axis=0)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    combined_tsne = tsne.fit_transform(combined_pts)

    # Normalize to [-4, 4]
    tsne_min = combined_tsne.min(axis=0)
    tsne_max = combined_tsne.max(axis=0)
    norm = -4.0 + 8.0 * (combined_tsne - tsne_min) / (tsne_max - tsne_min + 1e-8)

    z_tsne     = norm[:-best_K]     # real latent codes
    means_tsne = norm[-best_K:]     # projected IGMM centroids

    # 2D Voronoi assignment: partition 2D t-SNE plane by nearest projected centroid
    dists_2d = np.linalg.norm(z_tsne[:, None, :] - means_tsne[None, :, :], axis=2)
    assign_2d = np.argmin(dists_2d, axis=1)

    # Compute empirical 2D mean and covariance for each visual cluster
    means_tsne_emp = []
    cov_tsne_list = []
    for k in range(best_K):
        pts_k = z_tsne[assign_2d == k]
        if len(pts_k) > 1:
            emp_mean = np.mean(pts_k, axis=0)
            cov_k = np.cov(pts_k.T)
        else:
            emp_mean = means_tsne[k]
            cov_k = np.eye(2) * 0.1
        means_tsne_emp.append(emp_mean)
        cov_tsne_list.append(cov_k)
    means_tsne_emp = np.array(means_tsne_emp)


    # Create Side-by-Side Comparison Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), facecolor='#121212')

    colors = ['#E11D48', '#2563EB', '#16A34A', '#D97706', '#9333EA',
              '#0891B2', '#DB2777', '#4F46E5', '#CA8A04', '#059669']

    # ------------------ PLOT 1: PCA ------------------
    ax1.set_facecolor('#1e1e1e')
    scatter1 = ax1.scatter(
        z_pca[:, 0], z_pca[:, 1], c=y_np, cmap='tab10', s=15, alpha=0.5, edgecolors='none'
    )

    for k in range(best_K):
        ax1.plot(means_pca[k, 0], means_pca[k, 1], 'x', color='white', markersize=10, markeredgewidth=2)
        ax1.text(means_pca[k, 0] + 0.1, means_pca[k, 1] + 0.1, f"C{k}", color='white', fontsize=10, fontweight='bold')

        # Analytical projection of IGMM 10D covariance through PCA: Σ_2d = W^T Σ_k W
        L_k = L_np[k]
        Sigma_k = np.dot(L_k, L_k.T)
        Sigma_2d = np.dot(W.T, np.dot(Sigma_k, W))

        eigenvalues, eigenvectors = np.linalg.eigh(Sigma_2d)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 4 * np.sqrt(np.clip(eigenvalues, a_min=1e-8, a_max=None))

        el1 = Ellipse(
            xy=(means_pca[k, 0], means_pca[k, 1]),
            width=width, height=height, angle=angle,
            edgecolor='white', fc='none', lw=1.5, ls='--', alpha=0.8
        )
        ax1.add_patch(el1)

    ax1.set_title("PCA Linear Projection (Preserves Global Variances)", color='white', fontsize=14, pad=10)
    ax1.tick_params(colors='white')
    ax1.grid(color='#2d2d2d', linestyle=':', linewidth=0.5)

    # ------------------ PLOT 2: t-SNE ------------------
    ax2.set_facecolor('#1e1e1e')

    # Real data scatter colored by true MNIST label
    scatter2 = ax2.scatter(
        z_tsne[:, 0], z_tsne[:, 1], c=y_np, cmap='tab10', s=15, alpha=0.6, edgecolors='none',
        zorder=2
    )

    cmap12 = plt.colormaps.get_cmap('tab20').resampled(best_K)
    for k in range(best_K):
        cx, cy  = means_tsne_emp[k]
        emp_cov = cov_tsne_list[k]

        eigenvalues, eigenvectors = np.linalg.eigh(emp_cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        angle  = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 3.0 * np.sqrt(np.clip(eigenvalues, a_min=1e-8, a_max=None))

        color_k = cmap12(k)
        ax2.plot(cx, cy, 'x', color='white', markersize=10, markeredgewidth=2, zorder=4)
        ax2.text(cx + 0.05, cy + 0.05, f"C{k}", color='white', fontsize=9, fontweight='bold', zorder=4)

        el2 = Ellipse(
            xy=(cx, cy), width=width, height=height, angle=angle,
            edgecolor=color_k, facecolor=color_k, lw=1.5, ls='--',
            alpha=0.15, zorder=3
        )
        ax2.add_patch(el2)
        el2_border = Ellipse(
            xy=(cx, cy), width=width, height=height, angle=angle,
            edgecolor=color_k, facecolor='none', lw=1.5, ls='--',
            alpha=0.8, zorder=3
        )
        ax2.add_patch(el2_border)

    ax2.set_title("t-SNE Topological Projection (Preserves Local Manifolds)", color='white', fontsize=14, pad=10)
    ax2.tick_params(colors='white')
    ax2.grid(color='#2d2d2d', linestyle=':', linewidth=0.5)

    
    # Legend
    cbar = fig.colorbar(scatter2, ax=[ax1, ax2], location='bottom', pad=0.1, aspect=40)
    cbar.set_label("True MNIST Digit Label", color='white', fontsize=12)
    cbar.ax.xaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='white')
    
    plt.suptitle("GMVAE Latent Space Visualization Comparison", color='white', fontsize=18, y=0.98, fontweight='bold')
    
    # Save files
    out_filename = "latent_space_comparison.png"
    plt.savefig(out_filename, facecolor=fig.get_facecolor(), edgecolor='none', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {out_filename}")

if __name__ == "__main__":
    main()
