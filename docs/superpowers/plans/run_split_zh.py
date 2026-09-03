import sys
sys.path.insert(0, "docs/superpowers/plans")
from split_p01_cells import split_notebook

narrative_1 = """## 2. 模型：编码器

`Encoder` 就是 [L02 观测编码](../lectures/lecture-02-encode-and-dynamics/01-encoding) 里讲的 CNN：四层步幅为 2 的卷积每次把空间分辨率减半（64 -> 32 -> 16 -> 8 -> 4），最终特征图是 `256 x 4 x 4`。展平后经过两个线性层，分别输出 `mu` 和 `log_var`，对应该页引入的近似后验 $q(z \\mid x)$ 的均值和对数方差。之所以预测 `log_var` 而不是直接预测 `var` 或 `sigma`，是因为这样网络的原始输出可以是任意实数，后面取指数就能保证方差恒为正，不需要额外的约束激活函数。"""

narrative_2 = """`Decoder` 与编码器对称，用转置卷积（讲义里"转置卷积"深挖框讲过）把空间分辨率逐次翻倍，撤销编码器的压缩过程：`4 -> 8 -> 16 -> 32 -> 64`。最后的 `Sigmoid` 把输出像素限制在 `[0, 1]`，与合成图像的生成方式一致。"""

narrative_3 = """`VAE` 把编码器和解码器组合起来。`reparameterize` 就是讲义里的重参数化技巧，`z = mu + sigma * epsilon`，用 `std = exp(0.5 * log_var)` 实现，因为 `log_var = log(sigma^2)`，所以 `0.5 * log_var = log(sigma)`，取指数就还原出 `sigma`。`torch.randn_like(std)` 独立于网络参数采样出 `epsilon ~ N(0, I)`，这正是让这个操作保持可微的关键：梯度流过 `mu` 和 `std`，永远不流过随机采样本身。"""

split_notebook("docs/zh/projects/p01_vae_encoder.ipynb", narrative_1, narrative_2, narrative_3)
print("zh done")
