import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

from model import GMVAE

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
# CPU fallback for full covariance stability
if device.type == "mps":
    device = torch.device("cpu")
print("Benchmarking on device:", device)

def main():
    checkpoint = torch.load("best_model.pt", map_location=device)
    latent_dim = checkpoint["prior.means"].shape[1]
    best_K = checkpoint["prior.means"].shape[0]
    print(f"Loading checkpoint with K={best_K} IGMM components (latent_dim={latent_dim})")
    
    model = GMVAE(
        input_dim=784,
        hidden_dim=512,
        latent_dim=latent_dim,
        initial_K=best_K,
        covariance_type="full"
    ).to(device)
    
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Load batch of test data
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    x, _ = next(iter(test_loader))
    x = x.view(-1, 784).to(device)
    
    # Run multiple times to warm up CPU caches
    for _ in range(5):
        _ = model(x)
        _ = model.classify_optimized(x)
        
    # Benchmark 1: Standard Forward Pass (Standard Classification)
    start_time = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            _, _, _, _, q_y, _, _, _, _ = model(x)
            standard_preds = torch.argmax(q_y[:, :model.prior.K], dim=1)
    standard_time = (time.perf_counter() - start_time) * 1000 / 100
    
    # Benchmark 2: Optimized Sliced Classification (Active dimensions and IGMM submatrices)
    start_time = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            opt_preds = model.classify_optimized(x)
    opt_time = (time.perf_counter() - start_time) * 1000 / 100
    
    # Verify predictions match
    match_percentage = (standard_preds == opt_preds).float().mean().item() * 100
    
    active_dims = len(model.get_active_latent_indices())
    print("\n--- BENCHMARK RESULTS (1000 items batch) ---")
    print(f"Active Latent Dimensions: {active_dims} / {latent_dim}")
    print(f"Active IGMM Clusters: {best_K}")
    print(f"Standard Classification Time: {standard_time:.2f} ms")
    print(f"Optimized Classification Time: {opt_time:.2f} ms")
    print(f"Speedup: {standard_time / opt_time:.2f}x faster")
    print(f"Predictions Match Accuracy: {match_percentage:.2f}%")

if __name__ == "__main__":
    main()
