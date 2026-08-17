# GMVAE + Differentiable IGMN (PyTorch)

Fully differentiable implementation of the Incremental Gaussian Mixture Network (IGMN) integrated as a GMM prior inside a Gaussian Mixture Variational Autoencoder (GMVAE).

## Wide MLP Architecture
The model uses wide capacities to maximize reconstruction quality (achieving a validation loss of **7.22** on MNIST):
- **Encoder**: 784 -> 1024 -> 512 -> Latent Space (GELU activations)
- **Decoder**: Latent Space -> 512 -> 1024 -> 784 (GELU activations)

## Running the Code via VS Code
We included pre-configured launch configurations in `.vscode/launch.json`. You can run any path directly from the VS Code Run & Debug tab:
1. `Train: Default (Full Covariance)`
2. `Train: Diagonal Covariance`
3. `Train: ARD dimension selection`
4. `Train: KL-Pruning dimension selection`
5. `Train: PCA dimension selection`
6. `Animate: Training Evolution (GMM + Digits)`
7. `Animate: Latent Walk Morphing`
8. `Run: Performance Benchmark`

---

## CLI Usage Guide

### 1. Training Options

#### Default Training (Full Covariance)
Uses Cholesky parameterized covariance matrices ($\Sigma = L L^T$):
```bash
python3 train.py --covariance_type full --epochs 8
```
*Note: If run on Apple Silicon (MPS), the script automatically falls back to CPU for backward pass stability of `solve_triangular`.*

#### Diagonal Covariance Training
```bash
python3 train.py --covariance_type diagonal --epochs 8
```

---

### 2. Latent Dimension Selection Modes

Set `--dim_method` to automatically prune and select active latent dimensions during training:

#### ARD (Automatic Relevance Determination)
```bash
python3 train.py --dim_method ard --ard_lambda 1e-3 --latent_dim 10
```

#### KL-Pruning (Variance Collapse Masking)
```bash
python3 train.py --dim_method kl-pruning --var_threshold 0.05 --latent_dim 10
```

#### PCA Eigendecomposition Projection
```bash
python3 train.py --dim_method pca --pca_threshold 0.01 --latent_dim 10
```

---

### 3. Visualizations and Animations

#### Generate Training Evolution Animation (Floating Digits)
Trains the model for 25 epochs (with early stopping) and outputs `training_evolution.gif` demonstrating dynamic GMM spawning, GMM ellipses adapting, and decoded digit thumbnails floating above each cluster center:
```bash
python3 animate_training.py
```

#### Generate Latent Walk Morphing Animation
Walks between the GMM cluster centroids and generates a loop morphing digit styles:
```bash
python3 animate.py
```

---

### 4. Performance Benchmarks

Measures classification speedups by skipping pruned dimensions and inactive IGMN clusters:
```bash
python3 benchmark.py
```

---

### 5. Dynamic Spawning & Pruning

Our implementation automatically optimizes the number of clusters $K$ and active dimensions during training:
1. **Component Spawning**: Outliers in the latent space (Mahalanobis distance > $\chi^2$ threshold) trigger the dynamic addition of new Gaussian clusters.
2. **Component Pruning**: At the end of each epoch, components with an prior weight ($\pi_k$) below `0.005` (0.5%) are physically pruned from the model parameters, keeping the network compact and optimizing classification speed by **1.33x**.

