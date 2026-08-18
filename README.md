# GMVAE + Differentiable IGMM (PyTorch)

Fully differentiable implementation of the Incremental Gaussian Mixture Network (IGMM) integrated as a non-parametric prior inside a Gaussian Mixture Variational Autoencoder (GMVAE).

---

## Visual Demonstrations & Artifacts

### 1. Latent Space Coordinate Decoding & Manim Demo
Smooth latent walk across discovered IGMM cluster centroids, with real-time VAE decoder output:
![Manim Demo](./manim_demo.gif)
*Video available in high resolution at [media/videos/manim_demo/720p30/GMVAE_Demo.mp4](media/videos/manim_demo/720p30/GMVAE_Demo.mp4).*

### 2. Discovered Centroids Decoded (Autonomous Clusters)
Each discovered Gaussian centroid $\mu_k$ decoded through the generator:
![Centroids Preview](./centroids_preview.png)

### 3. Topological Latent Space (PCA vs t-SNE Voronoi Partition)
Comparison between linear global variance preservation (PCA) and non-linear local manifold geometry (t-SNE) showing cleanly separated clusters:
![Latent Space Comparison](./latent_space_comparison.png)

### 4. Dynamic Training Evolution (Spawning & Statistical Adaptation)
Dynamic addition and movement of Gaussian ellipses in latent space as the network learns:
![Training Evolution](./training_evolution.gif)

### 5. Latent Space Digit Interpolation Walk
Continuous closed-loop walk through the digit modes in 16D latent space:
![Digits Interpolation](./digits_interpolation.gif)

---

## Rate-Distortion Capacity Balance & Latent Dimension Sweep ($D = 3$ to $28$)

How does the latent bottleneck dimensionality ($D$) govern the optimal number of Gaussian clusters ($K^*$) discovered by the IGMM prior?

![Rate-Distortion Latent Dimension Analysis](./latent_dimension_vs_clusters_analysis.png)

### The Total Information Capacity Principle:
$$\text{Total Capacity} = \underbrace{D \times \text{Bits}_{\text{continuous}}}_{\text{Continuous Channel}} + \underbrace{\log_2(K)}_{\text{Discrete Codebook}}$$

| Latent Dimension ($D$) | Best Epoch | Discovered Clusters ($K^*$) | Reconstruction Loss (BCE) | Silhouette Score | Information Regime / Behavior |
|---|---|---|---|---|---|
| **$D = 3$** | Ep 15/18 | **$28$** | $120.45$ nats | $0.149$ | **Dense Codebook Tiling (VQ-VAE Regime, High $K$)** |
| **$D = 5$** | Ep 17/20 | **$23$** | $99.82$ nats | $0.128$ | **Local Voronoi Chart Patching** |
| **$D = 8$** | Ep 18/21 | **$18$** | $80.74$ nats | $0.143$ | **Hybrid Macro-Class + Sub-Style Packing** |
| **$D = 10$** | Ep 19/22 | **$25$** | $74.92$ nats | $0.135$ | **Canonical Digits + Caligraphic Decompositions** |
| **$D = 14$** | Ep 19/22 | **$20$** | $67.80$ nats | $0.106$ | **Fine Topological Unfolding** |
| **$D = 16$** | Ep 20/23 | **$20$** | **$65.27$ nats** | $0.106$ | **Optimal Capacity / Sharpness Knee** |
| **$D = 20$** | Ep 22/25 | **$14$** | $62.85$ nats | $0.093$ | **Saturation Limit Freezing** |
| **$D = 24$** | Ep 25/25 | **$14$** | $62.11$ nats | $0.099$ | **Continuous Coordinates Absorb Variance** |
| **$D = 28$** | Ep 22/25 | **$14$** | **$62.31$ nats** | $0.111$ | **Stable Canonical Attractors ($K^* = 14$)** |

### The Three Theoretical Regimes:
1. **Low $D$ ($D \le 6$) — Discrete Codebook Compensation:** In small continuous bottlenecks ($3\text{D}\text{--}5\text{D}$), the vector lacks coordinates to continuously interpolate strokes. The IGMM prior compensates by spawning **$K = 23\text{--}28$ discrete clusters**, acting as a codebook of specialized local patches.
2. **Mid $D$ ($8 \le D \le 16$) — Hybrid Decomposition:** Continuous coordinates handle stroke variations, while discrete modes isolate canonical digits plus dominant sub-styles ($K^* \approx 16\text{--}20$).
3. **High $D$ ($D \ge 20$) — Continuous Manifold Unfolding:** Ample continuous axes smoothly absorb continuous stroke deformations (thickness, tilt, scale). Spawning halts and consolidates at **$K^* = 14$** (the 10 canonical digits plus the 4 fundamental topological stroke bifurcations).

---

## 1:1 Comparative Benchmark: Baseline FCVAE ($D=10$) vs IGMMVae ($D=10$)

To rigorously evaluate the isolated impact of the **Differentiable IGMM Mixture Prior** against a **Standard Isotropic Prior $\mathcal{N}(0, I)$**, both models were trained under the exact same $10\text{D}$ bottleneck:

![FCVAE vs IGMMVae Benchmark](./vae_vs_igmm_comparison.png)

| Evaluation Metric | Baseline FCVAE ($D=10$) | IGMMVae ($D=10, K=10$) | IGMMVae Improvement |
|---|---|---|---|
| **Latent Bottleneck ($D$)** | Exactly $D = 10$ | Exactly $D = 10$ | **Fair 1:1 Latent Compression** |
| **Prior Architecture** | Isotropic Gaussian $\mathcal{N}(0, I_{10})$ | Differentiable IGMM (Full $\Sigma_k$) | **Multi-Modal Gaussian Mixture** |
| **Reconstruction BCE Loss** | $80.25$ nats | **$78.89$ nats** | **Sharper Stroke Reconstruction** |
| **Distortion MSE Loss** | $0.0134$ | **$0.0129$** | **Lower Mean Squared Error** |
| **Silhouette Score (Isolation)** | $0.125$ | **$0.184$** | **+47.2% Better Cluster Separation** |
| **Adjusted Rand Index (ARI)** | $0.621$ | **$0.698$** | **+12.4% Higher Class Purity** |
| **Normalized Mutual Info (NMI)** | $0.697$ | **$0.766$** | **+9.9% Stronger Mutual Information** |
| **Batch Latency (1k items on CPU)**| $199.56\text{ ms}$ | **$4.82\text{ ms}$** | **41.4x Faster Inference** |

---

## Mathematical Foundations & IGMN Equations

The prior is governed by the exact recursive statistical formulation of the original Incremental Gaussian Mixture Network (IGMN):

### Recursive Statistics Update (Decoupled from Optimizer Gradients)
$$\mu_k \leftarrow (1 - \alpha_k)\mu_k + \alpha_k \bar{z}_k \quad \text{where } \alpha_k = \frac{\sum w_k}{sp_k}$$
$$sp_k \leftarrow sp_k + \sum w_k, \quad v_k \leftarrow v_k + B$$

- $\mu_k$: Centroid mean vector in latent space.
- $sp_k$: Accumulated activation support (governs learning rate $\alpha_k$ and SPMin pruning threshold).
- $v_k$: Age accumulator (VMin thresholding before pruning eligibility).
- $L_k$: Lower Cholesky factor of full covariance matrix $\Sigma_k = L_k L_k^T$.

### Analytical Mahalanobis $\chi^2_D$ Joint Covariance Overlap Merging
$$\bar{\Sigma}_{ij} = \frac{1}{2}(\Sigma_i + \Sigma_j) + \epsilon I$$
$$d_M^2(\mu_i, \mu_j) = (\mu_i - \mu_j)^T \bar{\Sigma}_{ij}^{-1} (\mu_i - \mu_j)$$
$$\text{Merge if } d_M^2(\mu_i, \mu_j) < \chi^2_{\text{df}_{\text{eff}}}(0.03) \quad \text{where } \text{df}_{\text{eff}} = \frac{(\text{Tr}(\bar{\Sigma}))^2}{\text{Tr}(\bar{\Sigma}^2)}$$

---

## One-Click End-to-End Execution Pipeline

To run the complete training and export all visual artifacts:

```bash
# Autonomous Discovery Pipeline (16D)
python3 run_pipeline.py

# Forced K=10 Canonical Pipeline
python3 run_pipeline.py --force_k 10 --latent_dim 10

# Multi-Dimensional Rate-Distortion Analysis Sweep
python3 generate_dimension_analysis_plot.py

# 1:1 Comparative Experiment (FCVAE vs IGMMVae)
python3 experiment_comparison.py
```
