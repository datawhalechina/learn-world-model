import sys
sys.path.insert(0, "docs/superpowers/plans")
from split_p01_cells import split_notebook

narrative_1 = """## 2. Model: Encoder

The `Encoder` is the CNN from [L02 Observation Encoding](../lectures/lecture-02-encode-and-dynamics/01-encoding): four stride-2 convolutions halve the spatial resolution each time (64 -> 32 -> 16 -> 8 -> 4), so the final feature map is `256 x 4 x 4`. Flattening that map and passing it through two linear heads gives `mu` and `log_var`, the mean and log-variance of the approximate posterior $q(z \\mid x)$ introduced on that page. We predict `log_var` rather than `var` or `sigma` directly so the network's raw output can be any real number; exponentiating it later guarantees the variance stays positive without needing a constrained activation function."""

narrative_2 = """The `Decoder` mirrors the encoder with transposed convolutions (see the "Transposed Convolution" callout on the encoding page) that each double spatial resolution, undoing the encoder's compression: `4 -> 8 -> 16 -> 32 -> 64`. The final `Sigmoid` keeps output pixels in `[0, 1]`, matching how the synthetic images were generated."""

narrative_3 = """`VAE` combines the encoder and decoder. `reparameterize` is the reparameterization trick from the lecture, `z = mu + sigma * epsilon`, implemented with `std = exp(0.5 * log_var)` because `log_var = log(sigma^2)`, so `0.5 * log_var = log(sigma)` and exponentiating recovers `sigma`. `torch.randn_like(std)` draws `epsilon ~ N(0, I)` independently of the network parameters, which is exactly what keeps this operation differentiable: gradients flow through `mu` and `std`, never through the random draw itself."""

split_notebook("docs/en/projects/p01_vae_encoder.ipynb", narrative_1, narrative_2, narrative_3)
print("en done")
