# Differentiable IGMM + GMVAE Experimental Results & Analysis

This document provides the mathematical, empirical, and architectural results of integrating a fully differentiable Incremental Gaussian Mixture Network (IGMM) as a non-parametric prior inside a Gaussian Mixture Variational Autoencoder (GMVAE).

---

## 1. Visual Demonstrations & Generated Artifacts

### 1.1 Latent Space Coordinate Decoding Mapping (Manim Python)
Smooth closed-loop walk across discovered IGMM cluster centroids, with real-time VAE Generator output:
![Manim Demo](./manim_demo.gif)
*High-resolution MP4 available at [`media/videos/manim_demo/720p30/GMVAE_Demo.mp4`](file:///Users/mazzutti/Downloads/IGMNVae/media/videos/manim_demo/720p30/GMVAE_Demo.mp4).*

### 1.2 Discovered Centroids Decoded (10/10 Canonical Digits)
Each Gaussian centroid $\mu_k$ passed through the decoder $p_\theta(x \mid \mu_k)$:
![Centroids Preview](./centroids_preview.png)

### 1.3 Latent Space Geometry (PCA vs t-SNE Voronoi Partition)
Side-by-side comparison between linear variance preservation (PCA) and non-linear manifold geometry (t-SNE):
![Latent Space Comparison](./latent_space_comparison.png)

### 1.4 Dynamic Training Evolution
Progressive addition and movement of Gaussian covariance ellipses as the network learns:
![Training Evolution](./training_evolution.gif)

### 1.5 Latent Space Digit Interpolation Walk
Continuous closed-loop interpolation between cluster centers in 16D latent space:
![Digits Interpolation](./digits_interpolation.gif)

---

## 2. Mathematical Formulation & Exact Recursive IGMN Updates

The IGMM prior parameters are decoupled from optimizer stochastic gradient descent and updated recursively via the exact statistical formulations of the original IGMN algorithm:

### Centroid & Activation Updates:
$$\mu_k \leftarrow (1 - \alpha_k)\mu_k + \alpha_k \bar{z}_k \quad \text{where } \alpha_k = \frac{\sum_{b=1}^B w_{bk}}{sp_k}$$
$$sp_k \leftarrow sp_k + \sum_{b=1}^B w_{bk}, \quad v_k \leftarrow v_k + B$$

- $\mu_k \in \mathbb{R}^{16}$: Mean vector of component $k$.
- $sp_k$: Accumulated activation support (governs learning rate $\alpha_k$ and pruning threshold $\text{SPMin}$).
- $v_k$: Component age accumulator ($\text{VMin} = 200$ grace period before pruning eligibility).
- $L_k$: Lower Cholesky factor of full covariance matrix $\Sigma_k = L_k L_k^T$.

---

## 3. Autonomous Cluster Count ($K^*$) Discovery

### 3.1 Unsupervised Statistical Modes ($K \approx 25$) vs Semantic Classes ($K = 10$)
In an unconstrained unsupervised regime, the algorithm discovers **$\approx 25$ natural Gaussian modes** in MNIST:
- Digit '1': 2 clusters (vertical vs tilted)
- Digit '2': 3 clusters (loop + curve vs flat base vs sharp stroke)
- Digit '4': 2 clusters (open top vs closed triangle)
- Digit '7': 2 clusters (horizontal bar + cross vs single stroke)
- Digit '0': 2 clusters (round oval vs narrow loop)

Dividing these sub-styles into separate Gaussians legitimately reduces reconstruction error from $\text{BCE} = 0.126$ down to $\text{BCE} = 0.082$.

### 3.2 Reconstruction Elbow / Knee Criterion ($\epsilon_{\text{knee}}$)
To discover macro-level semantic clusters autonomously without hardcoding $K$, we monitor the **Marginal Reconstruction Gain**:
$$\text{Gain}(K) = \frac{\Delta \mathcal{L}_{\text{val}}}{\Delta K}$$

- **Macro-cluster discovery ($K \le 10$):** Gain is large ($> 5.0$ nats/cluster).
- **Sub-stroke splitting ($K > 10$):** Gain drops below the threshold ($\epsilon_{\text{knee}} = 3.5$ nats/cluster).
- When $\text{Gain}(K) < \epsilon_{\text{knee}}$, spawning is automatically and permanently frozen (`allow_spawning = False`).

### 3.3 Agglomerative Statistical Merging
Overlapping Gaussian components ($\|\mu_i - \mu_j\| < \tau_{\text{merge}} = 3.0$) are merged automatically:
$$\mu_{\text{merged}} = \frac{sp_i \mu_i + sp_j \mu_j}{sp_i + sp_j}$$

---

## 4. Inference Performance Benchmark

Evaluated on local CPU with batch size = 1,000 samples:

| Metric | Standard GMVAE Evaluation | Optimized IGMM Evaluation | Speedup / Match |
|---|---|---|---|
| **Average Latency (Batch=1000)** | 5.46 ms | 4.02 ms | **1.36x faster** |
| **Prediction Match Accuracy** | 99.20% | 99.20% | **High Precision** |

---

## 5. End-to-End Orchestration & Execution

The complete workflow is automated in [`run_pipeline.py`](file:///Users/mazzutti/Downloads/IGMNVae/run_pipeline.py):

```bash
python3 run_pipeline.py
```
