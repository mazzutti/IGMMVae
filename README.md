# GMVAE + Differentiable IGMM (PyTorch)

Fully differentiable implementation of the Incremental Gaussian Mixture Network (IGMM) integrated as a non-parametric prior inside a Gaussian Mixture Variational Autoencoder (GMVAE).

---

## Visual Demonstrations & Artifacts

### 1. Latent Space Coordinate Decoding & Manim Demo
Smooth latent walk across discovered IGMM cluster centroids, with real-time VAE decoder output:
![Manim Demo](./manim_demo.gif)
*Video available in high resolution at [media/videos/manim_demo/720p30/GMVAE_Demo.mp4](file:///Users/mazzutti/Downloads/IGMNVae/media/videos/manim_demo/720p30/GMVAE_Demo.mp4).*

### 2. Discovered Centroids Decoded (10/10 Canonical Digits)
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

## Mathematical Foundations & IGMN Equations

The prior is governed by the exact recursive statistical formulation of the original Incremental Gaussian Mixture Network (IGMN):

### Recursive Statistics Update (Decoupled from Optimizer Gradients)
$$\mu_k \leftarrow (1 - \alpha_k)\mu_k + \alpha_k \bar{z}_k \quad \text{where } \alpha_k = \frac{\sum w_k}{sp_k}$$
$$sp_k \leftarrow sp_k + \sum w_k, \quad v_k \leftarrow v_k + B$$

- $\mu_k$: Centroid mean vector in latent space ($D=16$).
- $sp_k$: Accumulated activation support (governs learning rate $\alpha_k$ and SPMin pruning threshold).
- $v_k$: Age accumulator (VMin thresholding before pruning eligibility).
- $L_k$: Lower Cholesky factor of full covariance matrix $\Sigma_k = L_k L_k^T$.

---

## Theoretical & Empirical Insights: Cluster Discovery Dynamics

### 1. Visual Statistical Modes ($K \approx 20\text{--}25$) vs Semantic Classes ($K = 10$)

When trained in a purely unsupervised statistical mode without class labels, the model discovers **$K \approx 20\text{--}25$ distinct Gaussian modes** on MNIST.

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │  STATISTICAL VIEW (Pixel Space)       vs  SEMANTIC VIEW (Human Classes)│
  ├────────────────────────────────────────────────────────────────────────┤
  │  Model discovers autonomously            There are 10 digit classes    │
  │  K = 20-25 Gaussian modes:               (0 through 9).                │
  │                                                                        │
  │  • '1' tilted vs '1' vertical            (2 clusters)                  │
  │  • '2' with loop vs '2' flat base        (3 clusters)                  │
  │  • '4' open top vs '4' closed top        (2 clusters)                  │
  │  • '7' with crossbar vs '7' straight     (2 clusters)                  │
  │  • '0' narrow vs '0' round               (2 clusters)                  │
  │  • '3', '5', '6', '8', '9' variations   (10 clusters)                 │
  │                                                                        │
  │  Total natural sub-styles: ~21 to 25     Total human classes: 10       │
  └────────────────────────────────────────────────────────────────────────┘
```

#### The Gaussian Unimodality & Covariance Inflation Problem:
A Gaussian distribution $\mathcal{N}(\mu, \Sigma)$ is strictly **unimodal and convex**. It mathematically cannot cover two distinct handwriting sub-styles (such as a vertical '1' and a $30^\circ$-tilted '1') without artificially inflating its covariance matrix $\Sigma$. 
- Forcing both styles into a single Gaussian component results in **covariance inflation**, causing the VAE decoder to generate blurred, ambiguous digits ($\text{BCE} \approx 72.14$).
- Splitting them into distinct Gaussian modes allows each component to model sharp, tight variances, reducing reconstruction loss to **$\text{BCE} = 66.43$**.

---

### 2. Agglomerative Merging & Exploration vs Consolidation Phases

The IGMM implements two distinct phases during training:

1. **Active Growth Phase (Exploration):**
   - New components are spawned whenever an outlier exhibits high statistical novelty:
     $$S(x) = p_{\text{new}}(z) \times \frac{\mathcal{L}_{\text{recon}}(x)}{\overline{\mathcal{L}}_{\text{recon}}} > 0.75$$
   - Agglomerative merging (`merge_components`) actively eliminates duplicate/overlapping Gaussians ($\|\mu_i - \mu_j\| < 2.6$) by computing their support-weighted mean:
     $$\mu_{\text{merged}} = \frac{sp_i \mu_i + sp_j \mu_j}{sp_i + sp_j}$$

2. **Structural Consolidation Phase:**
   - Once the **Reconstruction Elbow Criterion** detects diminishing returns ($\Delta \mathcal{L}_{\text{val}} / \Delta K < \epsilon_{\text{knee}} = 3.5$), both spawning and broad merging are **frozen** (`allow_spawning = False`).
   - *Why freezing merging is essential:* As the autoencoder optimizes latent representations in later epochs, distinct digit clusters naturally compress closer together. If broad geometric merging remained active indefinitely, adjacent clusters would be over-merged, causing cluster count to collapse from $K=10 \to 7$.

---

### 3. Automatic Parameter Determination

1. **Automatic Mahalanobis Eta ($\eta$):**
   $$\eta = F_{\chi^2}^{-1}(0.85; \; D)$$
2. **Gradient-Adaptive Spawning Cooldown ($\tau_{\text{cooldown}}$):**
   $$\tau_{\text{cooldown}} = \max\left(35, \; \left\lfloor \frac{N_{\text{batches}}}{10} \right\rfloor\right)$$
   Ensures the optimizer has sufficient parameter update steps ($\approx 10\%$ of an epoch) to adapt encoder/decoder manifolds before evaluating novelty.
3. **Centroid Repulsion Radius:**
   $$\min_{i \ne j} \|\mu_i - \mu_j\| \ge 3.0$$

---

## One-Click End-to-End Execution Pipeline

To run the complete training and export all visual artifacts (`latent_space_comparison.png`, `*.gif`, `*.mp4`):

```bash
python3 run_pipeline.py
```

### VS Code Run & Debug Configurations (`.vscode/launch.json`):
1. `🚀 Full Pipeline: Train -> Visuals -> Video & GIFs` *(Runs full end-to-end pipeline)*
2. `🏋️ Train Model (Generate best_model.pt)`
3. `📊 Visualize Latent Space (PCA vs t-SNE)`
4. `🎬 Render Manim Demo Video & GIF`
5. `🌀 Latent Digits Walk Animation`
6. `📈 Dynamic Training Evolution GIF`
7. `⚡ Run Inference Benchmark`

---

## Benchmark Results (Local CPU)

| Metric | Standard Classification | Optimized IGMM Classification | Speedup / Match |
|---|---|---|---|
| **Average Compute Time (Batch=1000)** | 5.46 ms | 4.02 ms | **1.36x faster** |
| **Prediction Match Accuracy** | 99.20% | 99.20% | **High Precision** |

---

## Project Structure & Files
- [run_pipeline.py](file:///Users/mazzutti/Downloads/IGMNVae/run_pipeline.py): Master end-to-end orchestration script.
- [train.py](file:///Users/mazzutti/Downloads/IGMNVae/train.py): Training loop with IGMM statistical updates.
- [model.py](file:///Users/mazzutti/Downloads/IGMNVae/model.py): Core GMVAE network architecture.
- [igmm.py](file:///Users/mazzutti/Downloads/IGMNVae/igmm.py): Differentiable IGMM prior module (Cholesky, covariance, merging & spawning logic).
- [visualize_tsne.py](file:///Users/mazzutti/Downloads/IGMNVae/visualize_tsne.py): Generates PCA vs t-SNE topological comparison plots.
- [manim_demo.py](file:///Users/mazzutti/Downloads/IGMNVae/manim_demo.py): Generates Manim video animation.
- [animate.py](file:///Users/mazzutti/Downloads/IGMNVae/animate.py): Generates latent walk interpolation GIF.
- [animate_training.py](file:///Users/mazzutti/Downloads/IGMNVae/animate_training.py): Generates dynamic training evolution GIF.
- [benchmark.py](file:///Users/mazzutti/Downloads/IGMNVae/benchmark.py): Measures inference latency and accuracy.
