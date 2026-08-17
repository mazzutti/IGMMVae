import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
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
    latent_dim = 10
    
    # Load VAE model
    model = GMVAE(
        input_dim=784,
        hidden_dim=512,
        latent_dim=latent_dim,
        initial_K=2,
        covariance_type="full"
    ).to(device)
    
    checkpoint = torch.load("best_model.pt", map_location=device)
    best_K = checkpoint["prior.means"].shape[0]
    
    if model.prior.K != best_K:
        model.prior.means = torch.nn.Parameter(torch.zeros(best_K, latent_dim).to(device))
        model.prior.L_params = torch.nn.Parameter(torch.zeros(best_K, latent_dim, latent_dim).to(device))
        model.prior.pi_logits = torch.nn.Parameter(torch.zeros(best_K).to(device))
        model.prior.K = best_K
        
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Load batch of test data
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1500, shuffle=False)
    x, y = next(iter(test_loader))
    x = x.view(-1, 784).to(device)
    
    # Encode test data
    with torch.no_grad():
        recon_x, z, q_mean, q_logvar, q_y, _, _, _, _ = model(x)
        predicted_clusters = torch.argmax(q_y[:, :model.prior.K], dim=1).cpu().numpy()
        
    z_np = z.cpu().numpy()
    y_np = y.numpy()
    means_10d = model.prior.means.data.cpu().numpy()
    
    print("Fitting PCA (Linear Projection)...")
    pca = sklearn_PCA(n_components=2)
    z_pca = pca.fit_transform(z_np)
    means_pca = pca.transform(means_10d)
    W = pca.components_.T
    
    # Get Cholesky covariances for PCA GMM projection
    L_tril = torch.tril(model.prior.L_params, diagonal=-1)
    diag_val = torch.diagonal(model.prior.L_params, dim1=1, dim2=2)
    clamped_diag = torch.clamp(diag_val, min=-3.0, max=0.0)
    L = L_tril + torch.diag_embed(torch.exp(clamped_diag))
    L_np = L.data.cpu().numpy()
    
    print("Fitting t-SNE (Topological Projection)...")
    # Combine latent points and GMM cluster centers to project them together
    combined_pts = np.concatenate([z_np, means_10d], axis=0)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    combined_tsne = tsne.fit_transform(combined_pts)
    
    z_tsne = combined_tsne[:-best_K]
    means_tsne = combined_tsne[-best_K:]
    
    # Create Side-by-Side Comparison Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), facecolor='#121212')
    
    # Common premium plot styling
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
        
        # Exact GMM Ellipse projection in linear space
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
            width=width,
            height=height,
            angle=angle,
            edgecolor='white',
            fc='none',
            lw=1.5,
            ls='--',
            alpha=0.8
        )
        ax1.add_patch(el1)
        
    ax1.set_title("PCA Linear Projection (Preserves Global Variances)", color='white', fontsize=14, pad=10)
    ax1.tick_params(colors='white')
    ax1.grid(color='#2d2d2d', linestyle=':', linewidth=0.5)
    
    # ------------------ PLOT 2: t-SNE ------------------
    ax2.set_facecolor('#1e1e1e')
    scatter2 = ax2.scatter(
        z_tsne[:, 0], z_tsne[:, 1], c=y_np, cmap='tab10', s=15, alpha=0.5, edgecolors='none'
    )
    
    for k in range(best_K):
        ax2.plot(means_tsne[k, 0], means_tsne[k, 1], 'x', color='white', markersize=10, markeredgewidth=2)
        ax2.text(means_tsne[k, 0] + 0.5, means_tsne[k, 1] + 0.5, f"C{k}", color='white', fontsize=10, fontweight='bold')
        
        # Calculate empirical 2D covariance of mapped points assigned to cluster k
        cluster_points = z_tsne[predicted_clusters == k]
        if len(cluster_points) > 5:
            # Empirical 2D mean and covariance
            emp_mean = np.mean(cluster_points, axis=0)
            emp_cov = np.cov(cluster_points.T)
            
            # Eigendecomposition of empirical 2D covariance
            eigenvalues, eigenvectors = np.linalg.eigh(emp_cov)
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            width, height = 4 * np.sqrt(np.clip(eigenvalues, a_min=1e-8, a_max=None))
            
            el2 = Ellipse(
                xy=(means_tsne[k, 0], means_tsne[k, 1]),
                width=width,
                height=height,
                angle=angle,
                edgecolor='white',
                fc='none',
                lw=1.5,
                ls='--',
                alpha=0.8
            )
            ax2.add_patch(el2)
            
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
    
    # Copy to artifacts folder
    artifact_path = "/Users/mazzutti/.gemini/antigravity-cli/brain/6cc502c9-3dcf-4f62-88dd-78d793a2ff6e/latent_space_comparison.png"
    import shutil
    shutil.copy(out_filename, artifact_path)
    print(f"Comparison plot saved to {out_filename} and copied to artifacts!")

if __name__ == "__main__":
    main()
