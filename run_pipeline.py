import os
import sys
import subprocess
import time
import io
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

from model import GMVAE

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title.upper()}")
    print("=" * 70)

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

    with torch.no_grad():
        for k in range(K):
            mu_k_tensor = model.prior.means[k].unsqueeze(0)
            recon_digit = model.decode(mu_k_tensor)
            digit_img = recon_digit.squeeze().view(28, 28).data.cpu().numpy()
            digit_img = np.clip(digit_img, 0.0, 1.0)
            
            rgba = np.zeros((28, 28, 4))
            rgba[:, :, 0] = 1.0
            rgba[:, :, 1] = 0.85
            rgba[:, :, 2] = 0.2
            rgba[:, :, 3] = digit_img
            
            emp_mean = emp_means_list.get(k, means_2d[k])
            im = OffsetImage(rgba, zoom=0.75)
            ab = AnnotationBbox(im, (emp_mean[0], emp_mean[1] + 0.6), xycoords='data', frameon=False)
            ax.add_artist(ab)
            
    val_loss_str = f"Loss: {val_loss:.2f}" if val_loss is not None else "Loss: N/A"
    status_str = f" | {action_text}" if action_text else ""
    plt.title(f"GMVAE + IGMM Evolution | Step {step:4d} | Epoch {epoch} | K={K} Clusters\n{val_loss_str}{status_str} (t-SNE {model.latent_dim}D -> 2D)", color='white', fontsize=11)
    
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

def step1_train_best_model(force_k=None, epochs=14, latent_dim=16):
    mode_str = f"Forced Canonical K={force_k}" if force_k is not None else "Autonomous K Discovery"
    print_header(f"Step 1/5: Training GMVAE + Differentiable IGMM ({mode_str})")
    device = torch.device("cpu")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    initial_K = force_k if force_k is not None else 2
    spawn_cooldown = max(35, len(train_loader) // 10)
    epsilon_knee = 3.5
    # Dimension-scaled geometric thresholds (Curse of Dimensionality / Hypersphere volume scaling)
    dim_scale = (latent_dim / 16.0) ** 0.5
    merge_dist_threshold = 3.18 * dim_scale
    min_spawn_dist = 2.4 * dim_scale
    
    # Step capture interval: capture snapshots every 60 batches during training
    capture_step_interval = 60
    
    model = GMVAE(
        input_dim=784,
        hidden_dim=512,
        latent_dim=latent_dim,
        initial_K=initial_K,
        covariance_type="full"
    ).to(device)
    
    if force_k is not None:
        with torch.no_grad():
            for k in range(force_k):
                vec = torch.zeros(latent_dim)
                if k < latent_dim:
                    vec[k] = 3.5
                model.prior.means.data[k] = vec
    
    ae_params = list(model.encoder_module.parameters()) + list(model.decoder_module.parameters()) + [model.prior.L_params]
    optimizer = optim.Adam(ae_params, lr=1.2e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    anim_frames = []
    print("Capturing Step 0 (Initial State before training)...")
    anim_frames.append(get_plot_frame(model, test_loader, device, 0, 0, None, 1.2e-3, "Initial State (K=2)"))
    
    step_count = 0
    last_spawn_step = -spawn_cooldown
    allow_spawning = True if force_k is None else False
    prev_val_loss = None
    prev_K = initial_K
    
    epochs = 14
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, 784).to(device)
            B, D = x.shape[0], latent_dim
            
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
            
            # Exact Recursive IGMN Updates
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
            
            # Autonomous Spawning
            spawned = False
            if allow_spawning and (step_count - last_spawn_step >= spawn_cooldown):
                combined_novelty = p_new * (stroke_weight.mean(dim=1))
                max_score, max_idx = torch.max(combined_novelty, dim=0)
                if max_score.item() > 0.75:
                    new_mean = q_mean[max_idx].detach()
                    min_dist = torch.min(torch.norm(model.prior.means - new_mean.unsqueeze(0), dim=1))
                    if min_dist > min_spawn_dist:
                        model.prior.spawn_component(new_mean)
                        last_spawn_step = step_count
                        spawned = True
                        print(f"  [Auto-Spawn] Step {step_count}: Discovered Cluster K -> {model.prior.K} (dist={min_dist:.2f})")
                        ae_params = list(model.encoder_module.parameters()) + list(model.decoder_module.parameters()) + [model.prior.L_params]
                        current_lr = optimizer.param_groups[0]['lr']
                        optimizer = optim.Adam(ae_params, lr=current_lr)
                        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
                        
                        # Capture frame right when a cluster is spawned!
                        anim_frames.append(get_plot_frame(model, test_loader, device, step_count, epoch, loss.item(), current_lr, f"Spawned C{model.prior.K-1}"))

            # Intermediate step frame capture (every capture_step_interval steps)
            if not spawned and (step_count % capture_step_interval == 0):
                current_lr = optimizer.param_groups[0]['lr']
                anim_frames.append(get_plot_frame(model, test_loader, device, step_count, epoch, loss.item(), current_lr, f"Training Step {step_count}"))
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Validation & Elbow Knee Detection
        model.eval()
        with torch.no_grad():
            test_x, _ = next(iter(test_loader))
            test_x = test_x.view(-1, 784).to(device)
            t_recon, _, _, _, _, _, _, _, _ = model(test_x)
            val_loss = F.binary_cross_entropy(t_recon, test_x, reduction='sum').item() / test_x.shape[0]
            
        current_K = model.prior.K
        print(f"Epoch {epoch:2d}/{epochs} | Discovered K = {current_K:2d} | Avg Loss: {avg_loss:.2f} | Val Loss: {val_loss:.2f}")
        
        if prev_val_loss is not None and allow_spawning:
            delta_K = current_K - prev_K
            delta_loss = prev_val_loss - val_loss
            if delta_K > 0:
                gain_per_k = delta_loss / delta_K
                print(f"  [Elbow Metric] Marginal Gain: {gain_per_k:.2f} nats/cluster (Delta K={delta_K}, Delta Loss={delta_loss:.2f})")
                if gain_per_k < epsilon_knee and current_K >= 8:
                    print(f"  [Elbow Knee Detected!] Freezing structural growth at optimal K={current_K}.")
                    allow_spawning = False
                    
        prev_val_loss = val_loss
        prev_K = current_K
        
        # Merge duplicates only during active growth phase
        if force_k is None and epoch <= 6:
            merged_any = False
            while model.prior.merge_components(merge_dist_threshold=merge_dist_threshold):
                merged_any = True
                ae_params = list(model.encoder_module.parameters()) + list(model.decoder_module.parameters()) + [model.prior.L_params]
                current_lr = optimizer.param_groups[0]['lr']
                optimizer = optim.Adam(ae_params, lr=current_lr)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
            if merged_any:
                anim_frames.append(get_plot_frame(model, test_loader, device, step_count, epoch, val_loss, current_lr, f"Merged Overlapping (K={model.prior.K})"))
                
        # End of epoch frame
        anim_frames.append(get_plot_frame(model, test_loader, device, step_count, epoch, val_loss, current_lr, f"Epoch {epoch} Complete"))

    torch.save(model.state_dict(), "best_model.pt")
    final_K = model.prior.K
    print(f"✓ Saved best_model.pt (Autonomously Discovered K* = {final_K})")
    
    cols = min(6, final_K)
    rows = (final_K + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).flatten()
    with torch.no_grad():
        for k in range(final_K):
            mu = model.prior.means[k:k+1]
            img = model.decode(mu).view(28, 28).numpy()
            axes[k].imshow(img, cmap="magma")
            axes[k].set_title(f"Discovered C{k}")
            axes[k].axis("off")
        for k in range(final_K, len(axes)):
            axes[k].axis("off")
            
    plt.tight_layout()
    plt.savefig("centroids_preview.png", dpi=150)
    plt.close()
    print(f"✓ Saved centroids_preview.png ({final_K} Discovered Clusters)")
    
    print(f"Saving high-resolution training_evolution.gif ({len(anim_frames)} step-by-step frames)...")
    anim_frames[0].save(
        "training_evolution.gif",
        save_all=True,
        append_images=anim_frames[1:],
        optimize=False,
        duration=180,
        loop=0
    )
    print(f"✓ Saved training_evolution.gif with {len(anim_frames)} continuous execution frames")

def step2_visualize_latent_space():
    print_header("Step 2/5: Generating Latent Space Comparison (PCA vs t-SNE)")
    subprocess.run([sys.executable, "visualize_tsne.py"], check=True)
    print("✓ Generated latent_space_comparison.png")

def step3_generate_gifs():
    print_header("Step 3/5: Generating Latent Digits Walk Interpolation GIF")
    subprocess.run([sys.executable, "animate.py"], check=True)
    print("✓ Generated digits_interpolation.gif")

def step4_generate_manim_video():
    print_header("Step 4/5: Rendering Manim Video (GMVAE_Demo.mp4 & manim_demo.gif)")
    subprocess.run(["manim", "-qm", "manim_demo.py", "GMVAE_Demo"], check=True)
    
    mp4_path = "media/videos/manim_demo/720p30/GMVAE_Demo.mp4"
    if os.path.exists(mp4_path):
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", mp4_path,
            "-vf", "fps=15,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            "-loop", "0", "manim_demo.gif"
        ]
        subprocess.run(ffmpeg_cmd, check=True)
        print("✓ Generated media/videos/manim_demo/720p30/GMVAE_Demo.mp4")
        print("✓ Generated manim_demo.gif")

def step5_benchmark():
    print_header("Step 5/5: Running Benchmark Validation")
    subprocess.run([sys.executable, "benchmark.py"], check=True)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Full GMVAE + IGMM Pipeline")
    parser.add_argument("--force_k", type=int, default=None, help="Force a fixed number of clusters (e.g. 10)")
    parser.add_argument("--epochs", type=int, default=14, help="Number of training epochs")
    parser.add_argument("--latent_dim", type=int, default=16, help="Latent bottleneck dimension (e.g. 10 or 16)")
    args = parser.parse_args()

    start_time = time.time()
    mode_title = f"FORCED K={args.force_k} CANONICAL" if args.force_k is not None else "AUTONOMOUS K DISCOVERY"
    print(f"\n🚀 STARTING FULL GMVAE + IGMM PIPELINE ({mode_title} | Latent D={args.latent_dim})")
    
    step1_train_best_model(force_k=args.force_k, epochs=args.epochs, latent_dim=args.latent_dim)
    step2_visualize_latent_space()
    step3_generate_gifs()
    step4_generate_manim_video()
    step5_benchmark()
    
    elapsed = time.time() - start_time
    print_header("Pipeline Completed Successfully!")
    print(f"⏱ Total Execution Time: {elapsed:.1f}s")
    print("📁 Output Artifacts:")
    print("  - best_model.pt")
    print("  - centroids_preview.png")
    print("  - latent_space_comparison.png")
    print("  - digits_interpolation.gif")
    print("  - training_evolution.gif (Step-by-Step Recording)")
    print("  - manim_demo.gif")
    print("  - media/videos/manim_demo/720p30/GMVAE_Demo.mp4")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
