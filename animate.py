import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

from model import GMVAE
from train import compute_loss

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def train_and_get_model():
    print("Training a model to generate the latent space interpolation...")
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # 2D latent space so we can visualize and interpolate easily
    model = GMVAE(
        input_dim=784,
        hidden_dim=256,
        latent_dim=2,
        initial_K=2,
        beta=0.5,
        eta=9.0, # 2D threshold
        dim_method="none"
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    step_count = 0
    spawn_cooldown = 150
    last_spawn_step = -spawn_cooldown
    
    # Train for 4 epochs (fast and sufficient for clear digit shapes in 2D)
    for epoch in range(1, 5):
        model.train()
        for x, _ in train_loader:
            x = x.view(-1, 784).to(device)
            recon_x, z, q_mean, q_logvar, q_y, d_sq, p_new, _, _ = model(x)
            loss, _, _, _ = compute_loss(recon_x, x, q_mean, q_logvar, q_y, model.prior, 0.0, device)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step_count += 1
            if step_count - last_spawn_step >= spawn_cooldown:
                max_p_new, max_idx = torch.max(p_new, dim=0)
                if max_p_new.item() > 0.85 and model.prior.K < 8:
                    new_mean = z[max_idx].detach()
                    model.prior.spawn_component(new_mean)
                    last_spawn_step = step_count
                    optimizer = optim.Adam(model.parameters(), lr=1e-3)
                    
        print(f"Epoch {epoch}/4 completed. Clusters: {model.prior.K}")
        
    return model

def create_interpolation_gif(model, filename="digits_interpolation.gif"):
    model.eval()
    means = model.prior.means.data.cpu().numpy()
    K = model.prior.K
    
    print(f"Discovered GMM means: {K}")
    
    frames = []
    steps_per_segment = 15
    
    # Generate the path: 0 -> 1 -> 2 -> ... -> K-1 -> 0
    path_indices = list(range(K)) + [0]
    
    with torch.no_grad():
        for i in range(len(path_indices) - 1):
            start_mean = means[path_indices[i]]
            end_mean = means[path_indices[i+1]]
            
            for step in range(steps_per_segment):
                t = step / steps_per_segment
                # Linear interpolation
                z_val = (1 - t) * start_mean + t * end_mean
                z_tensor = torch.tensor(z_val, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Decode to get image
                recon = model.decode(z_tensor).cpu().numpy().reshape(28, 28)
                
                # Convert to PIL Image
                img_data = (recon * 255).astype(np.uint8)
                img = Image.fromarray(img_data)
                
                # Resize (upscale) using nearest neighbor to preserve pixel art style or bilinear for smooth
                img = img.resize((280, 280), Image.Resampling.BILINEAR)
                
                # Add text label of current cluster source
                draw = ImageDraw.Draw(img)
                # Drawing a small marker/text indicating transition
                draw.text((10, 10), f"From Cluster {path_indices[i]} to {path_indices[i+1]}", fill=255)
                
                frames.append(img)
                
    # Save the animation
    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=80, # ms per frame
        loop=0
    )
    print(f"Animation saved as {filename}")

if __name__ == "__main__":
    model = train_and_get_model()
    create_interpolation_gif(model, "/Users/mazzutti/Downloads/IGMNVae/digits_interpolation.gif")
