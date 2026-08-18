import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
from PIL import Image, ImageDraw
import os
import shutil

from model import GMVAE

device = torch.device("cpu") # CPU for inference stability

def main():
    checkpoint = torch.load("best_model.pt", map_location=device)
    latent_dim = checkpoint["prior.means"].shape[1]
    best_K = checkpoint["prior.means"].shape[0]
    print(f"Loading checkpoint with K={best_K} clusters (latent_dim={latent_dim})")
    
    # 1. Load the trained best model
    model = GMVAE(
        input_dim=784,
        hidden_dim=512,
        latent_dim=latent_dim,
        initial_K=best_K,
        covariance_type="full"
    )
    
    model.load_state_dict(checkpoint)
    model.eval()
    
    # 2. Get test data to fit t-SNE for walk order sorting
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1500, shuffle=False)
    x, _ = next(iter(test_loader))
    x = x.view(-1, 784)
    
    with torch.no_grad():
        _, z, _, _, _, _, _, _, _ = model(x)
    z_np = z.numpy()
    means = model.prior.means.data.numpy()
    
    print("Fitting t-SNE to sort cluster visit path...")
    combined_pts = np.concatenate([z_np, means], axis=0)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    combined_tsne = tsne.fit_transform(combined_pts)
    means_2d = combined_tsne[-best_K:]
    
    # Nearest neighbor walk starting from C0
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
    
    # 3. Interpolate frames in 10D space
    print(f"Generating interpolation along walk path: {walk_order}")
    frames = []
    steps_per_segment = 20
    
    with torch.no_grad():
        for i in range(len(walk_order) - 1):
            start_10d = means[walk_order[i]]
            end_10d = means[walk_order[i+1]]
            
            for step in range(steps_per_segment):
                t = step / steps_per_segment
                # 10D Linear interpolation
                z_val = (1.0 - t) * start_10d + t * end_10d
                z_tensor = torch.tensor(z_val, dtype=torch.float32).unsqueeze(0)
                
                # Decode to image
                recon = model.decode(z_tensor).numpy().reshape(28, 28)
                recon = np.clip(recon, 0.0, 1.0)
                
                # Render frame
                img_data = (recon * 255).astype(np.uint8)
                img = Image.fromarray(img_data)
                img = img.resize((280, 280), Image.Resampling.BILINEAR)
                
                # Draw text annotation
                draw = ImageDraw.Draw(img)
                draw.text((10, 10), f"Centroid C{walk_order[i]} -> C{walk_order[i+1]}", fill=255)
                
                frames.append(img)
                
    # 4. Save and copy the looping GIF
    out_filename = "digits_interpolation.gif"
    frames[0].save(
        out_filename,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=70, # ms per frame
        loop=0
    )
    print(f"Looping interpolation GIF saved to {out_filename}")

if __name__ == "__main__":
    main()
