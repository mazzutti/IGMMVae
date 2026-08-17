import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
import os
import shutil
from PIL import Image

# Manim import
from manim import *

# Pre-generate frames for Manim using t-SNE
def pregenerate_frames(model_path="best_model.pt"):
    device = torch.device("cpu") # Keep on CPU for stability
    latent_dim = 10
    
    from model import GMVAE
    model = GMVAE(
        input_dim=784,
        hidden_dim=512,
        latent_dim=latent_dim,
        initial_K=2,
        covariance_type="full"
    )
    
    # Load model and dynamically adjust parameter sizes
    checkpoint = torch.load(model_path, map_location=device)
    best_K = checkpoint["prior.means"].shape[0]
    
    if model.prior.K != best_K:
        model.prior.means = torch.nn.Parameter(torch.zeros(best_K, latent_dim))
        model.prior.L_params = torch.nn.Parameter(torch.zeros(best_K, latent_dim, latent_dim))
        model.prior.pi_logits = torch.nn.Parameter(torch.zeros(best_K))
        model.prior.K = best_K
        
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Get test data to fit t-SNE projection
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1500, shuffle=False)
    x, y = next(iter(test_loader))
    x = x.view(-1, 784)
    
    with torch.no_grad():
        _, z, _, _, q_y, _, _, _, _ = model(x)
        predicted_clusters = torch.argmax(q_y, dim=1).numpy()
        
    z_np = z.numpy()
    means = model.prior.means.data.numpy()
    
    print("Fitting t-SNE (Topological Projection 10D -> 2D)...")
    # Combine latent points and GMM cluster centers to project them together
    combined_pts = np.concatenate([z_np, means], axis=0)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    combined_tsne = tsne.fit_transform(combined_pts)
    
    # Normalize t-SNE coordinates to fit beautifully in the axes range [-4.0, 4.0]
    tsne_min = combined_tsne.min(axis=0)
    tsne_max = combined_tsne.max(axis=0)
    combined_tsne_norm = -4.0 + 8.0 * (combined_tsne - tsne_min) / (tsne_max - tsne_min + 1e-8)
    
    z_tsne = combined_tsne_norm[:-best_K]
    means_2d = combined_tsne_norm[-best_K:]
    
    # Compute empirical 2D covariances of normalized coordinates assigned to each cluster
    cov_2d_list = []
    for k in range(best_K):
        pts_k = z_tsne[predicted_clusters == k]
        if len(pts_k) > 5:
            cov_k = np.cov(pts_k.T)
        else:
            cov_k = np.eye(2) * 0.1
        cov_2d_list.append(cov_k)
        
    # Determine the order of visiting cluster centers (sorted coordinate path)
    unvisited = list(range(best_K))
    walk_order = [0]
    unvisited.remove(0)
    
    while unvisited:
        curr = walk_order[-1]
        distances = [np.linalg.norm(means_2d[curr] - means_2d[u]) for u in unvisited]
        next_idx = unvisited[np.argmin(distances)]
        walk_order.append(next_idx)
        unvisited.remove(next_idx)
    walk_order.append(0) # Loop back to start
    
    # Interpolate path in the 10D space along the walk order
    steps_per_segment = 25
    path_10d = []
    path_2d_points = []
    
    for i in range(len(walk_order) - 1):
        start_10d = means[walk_order[i]]
        end_10d = means[walk_order[i+1]]
        
        start_2d = means_2d[walk_order[i]]
        end_2d = means_2d[walk_order[i+1]]
        
        for t in np.linspace(0.0, 1.0, steps_per_segment, endpoint=False):
            # 10D interpolation for Decoder generation
            pt_10d = start_10d * (1.0 - t) + end_10d * t
            path_10d.append(pt_10d)
            
            # 2D interpolation for visual walking dot
            pt_2d = start_2d * (1.0 - t) + end_2d * t
            path_2d_points.append(pt_2d)
            
    # Add final frame
    path_10d.append(means[walk_order[-1]])
    path_2d_points.append(means_2d[walk_order[-1]])
    
    # Reconstruct images for each path coordinate
    shutil.rmtree("scratch/manim_frames", ignore_errors=True)
    os.makedirs("scratch/manim_frames", exist_ok=True)
    
    with torch.no_grad():
        for idx, pt in enumerate(path_10d):
            pt_tensor = torch.tensor(pt, dtype=torch.float32).unsqueeze(0)
            recon = model.decode(pt_tensor)
            img_arr = recon.squeeze().view(28, 28).numpy()
            img_arr = np.clip(img_arr, 0.0, 1.0)
            
            # Upscale and save image
            img_pil = Image.fromarray((img_arr * 255).astype(np.uint8))
            img_pil = img_pil.resize((128, 128), Image.Resampling.LANCZOS)
            img_pil.save(f"scratch/manim_frames/frame_{idx}.png")
            
    return means_2d, cov_2d_list, path_2d_points, len(path_10d)


# Run pre-generation before scene rendering
print("Pre-generating digit reconstruction frames using t-SNE...")
means_2d, cov_2d_list, path_2d, num_frames = pregenerate_frames()

class GMVAE_Demo(Scene):
    def construct(self):
        # Set dark premium theme
        self.camera.background_color = "#121212"
        
        # Title of scene
        title = Text("GMVAE + Differentiable IGMN", font_size=32, color=WHITE)
        title.to_edge(UP, buff=0.4)
        subtitle = Text("Dynamic Latent Space Structuring & Digit Generation", font_size=20, color=GRAY)
        subtitle.next_to(title, DOWN, buff=0.2)
        
        self.play(FadeIn(title), Write(subtitle))
        self.wait(0.5)
        
        # 1. Left side: Latent Space axes (t-SNE topological projection)
        axes = NumberPlane(
            x_range=[-5, 5, 2],
            y_range=[-5, 5, 2],
            x_length=5,
            y_length=5,
            background_line_style={"stroke_color": "#262626", "stroke_width": 1}
        )
        axes.shift(LEFT * 3)
        
        latent_label = Text("Latent Space (Projected 2D via t-SNE)", font_size=14, color=LIGHT_GRAY)
        latent_label.next_to(axes, DOWN, buff=0.2)
        
        self.play(Create(axes), FadeIn(latent_label))
        
        # Draw clusters and empirical ellipses
        cluster_dots = VGroup()
        ellipses = VGroup()
        
        colors = [RED, BLUE, GREEN, YELLOW, ORANGE, PURPLE, PINK, TEAL, GOLD, MAROON]
        
        for k in range(len(means_2d)):
            pt_2d = means_2d[k]
            screen_pos = axes.c2p(pt_2d[0], pt_2d[1])
            
            # Mean cross
            dot = Cross(stroke_width=2, scale_factor=0.15).move_to(screen_pos)
            dot.set_color(colors[k % len(colors)])
            
            # Compute visual ellipse from empirical covariance
            Sigma_2d = cov_2d_list[k]
            eigenvalues, eigenvectors = np.linalg.eigh(Sigma_2d)
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
            
            angle = np.arctan2(*eigenvectors[:, 0][::-1])
            width, height = 2.0 * np.sqrt(np.clip(eigenvalues, a_min=1e-8, a_max=None)) # 1 std radius
            
            scale_x = axes.x_length / (axes.x_range[1] - axes.x_range[0])
            scale_y = axes.y_length / (axes.y_range[1] - axes.y_range[0])
            
            el = Ellipse(
                width=width * scale_x * 2.0, 
                height=height * scale_y * 2.0, 
                color=colors[k % len(colors)],
                stroke_width=1.5,
                stroke_opacity=0.6,
                fill_opacity=0.08
            ).move_to(screen_pos)
            el.rotate(angle)
            
            cluster_dots.add(dot)
            ellipses.add(el)
            
        self.play(FadeIn(ellipses, lag_ratio=0.1), FadeIn(cluster_dots, lag_ratio=0.1))
        
        # 2. Right side: Reconstruction Panel
        recon_box = Rectangle(width=4.0, height=4.0, stroke_color=BLUE_D, stroke_width=2, fill_color="#181818", fill_opacity=0.8)
        recon_box.shift(RIGHT * 3)
        
        recon_label = Text("Decoder Output Image", font_size=16, color=LIGHT_GRAY)
        recon_label.next_to(recon_box, DOWN, buff=0.2)
        
        self.play(Create(recon_box), FadeIn(recon_label))
        
        # 3. Dynamic updater: walking dot and updating image
        frame_tracker = ValueTracker(0)
        
        first_pos = axes.c2p(path_2d[0][0], path_2d[0][1])
        walk_dot = Dot(color=YELLOW, radius=0.12).move_to(first_pos)
        walk_dot_halo = Dot(color=YELLOW, radius=0.25, fill_opacity=0.3).move_to(first_pos)
        walk_dot_halo.add_updater(lambda m: m.move_to(walk_dot.get_center()))
        
        self.play(FadeIn(walk_dot), FadeIn(walk_dot_halo))
        self.add(walk_dot_halo)
        
        arrow = Arrow(
            start=axes.get_right() + RIGHT * 0.2, 
            end=recon_box.get_left() - RIGHT * 0.2, 
            stroke_width=3, 
            color=BLUE
        )
        decoder_text = Text("VAE Decoder", font_size=12, color=BLUE_A).next_to(arrow, UP, buff=0.1)
        self.play(GrowArrow(arrow), FadeIn(decoder_text))
        
        def get_digit_mobject():
            idx = int(frame_tracker.get_value())
            idx = min(max(0, idx), num_frames - 1)
            img = ImageMobject(f"scratch/manim_frames/frame_{idx}.png")
            img.scale_to_fit_width(3.2)
            img.move_to(recon_box.get_center())
            return img
            
        digit_mobject = always_redraw(get_digit_mobject)
        self.add(digit_mobject)
        
        # Move dot along the 2D t-SNE path
        def update_dot_position(m):
            idx = int(frame_tracker.get_value())
            idx = min(max(0, idx), num_frames - 1)
            pt = path_2d[idx]
            m.move_to(axes.c2p(pt[0], pt[1]))
            
        walk_dot.add_updater(update_dot_position)
        
        self.play(
            frame_tracker.animate.set_value(num_frames - 1),
            run_time=18, # Slightly longer walk for 10 clusters
            rate_func=linear
        )
        
        walk_dot.remove_updater(update_dot_position)
        self.wait(1.0)
        
        # Cleanup
        self.play(
            FadeOut(title), FadeOut(subtitle),
            FadeOut(axes), FadeOut(latent_label),
            FadeOut(cluster_dots), FadeOut(ellipses),
            FadeOut(recon_box), FadeOut(recon_label),
            FadeOut(walk_dot), FadeOut(walk_dot_halo),
            FadeOut(arrow), FadeOut(decoder_text),
            FadeOut(digit_mobject)
        )
        self.wait(0.5)
