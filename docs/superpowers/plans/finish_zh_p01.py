import json

path = "docs/zh/projects/p01_vae_encoder.ipynb"
with open(path) as f:
    nb = json.load(f)
cells = nb["cells"]

def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}

def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

demo_md = md_cell(
    "**核对讲义里的手算例子**：讲义用 $\\mu = 0.50$，$\\sigma = 0.20$，$\\varepsilon = 1.30$ 手算出 "
    "$z = 0.76$。下面用同样的数字跑一遍 `reparameterize`，确认代码和公式完全对得上。"
)
demo_code = code_cell(
    "demo_mu = torch.tensor([[0.50]])\n"
    "demo_std = torch.tensor([[0.20]])\n"
    "demo_eps = torch.tensor([[1.30]])\n"
    "demo_z = demo_mu + demo_std * demo_eps\n"
    "print(f'z = mu + std * eps = {demo_mu.item()} + {demo_std.item()} * {demo_eps.item()} = {demo_z.item():.2f}')\n"
    "assert abs(demo_z.item() - 0.76) < 1e-6, '与讲义手算结果不一致'\n"
)

elbo_md = md_cell(
"""## 3. ELBO 损失

`elbo_loss` 实现的正是[讲义里推导的 ELBO 两项](../lectures/lecture-02-encode-and-dynamics/01-encoding#elbo-损失两个目标的平衡)：

$$
\\mathcal{L} = \\underbrace{\\mathbb{E}_{q(z|x)}[-\\log p(x|z)]}_{\\text{重建}} + \\underbrace{D_{KL}(q(z|x) \\| p(z))}_{\\text{KL 散度}}
$$

**重建项。** `F.mse_loss(recon_x, x, reduction='mean')` 代替了 $-\\log p(x \\mid z)$。这个替换是标准做法，但值得说清楚原因：如果假设解码器输出的分布是以重建像素为中心、方差固定的高斯分布，那么 $-\\log p(x \\mid z)$ 化简后就是 `recon_x` 和 `x` 之间的平方误差（相差一个常数项），正好就是 MSE。所以用 MSE 不是图方便的近似，而是这个高斯假设下精确的负对数似然。

**KL 项的解析形式从哪来。** 对于两个高斯分布，$q(z \\mid x) = \\mathcal{N}(\\mu, \\sigma^2)$ 和标准正态先验 $p(z) = \\mathcal{N}(0, 1)$，KL 散度有精确的解析形式（不需要蒙特卡洛采样），把 $D_{KL}(q \\| p) = \\int q(z) \\log \\frac{q(z)}{p(z)} \\, dz$ 的定义代入高斯密度函数展开即可得到：

$$
D_{KL}\\big(\\mathcal{N}(\\mu, \\sigma^2) \\,\\|\\, \\mathcal{N}(0, 1)\\big) = -\\frac{1}{2}\\left(1 + \\log \\sigma^2 - \\mu^2 - \\sigma^2\\right)
$$

对全部 32 个潜在维度求和（假设各维度独立，联合 KL 就是各维度 KL 之和），正好得到代码里 `torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)` 这一项（取负、除以 2）。`log_var.exp()` 从 $\\log \\sigma^2$ 还原出 $\\sigma^2$，和上面 `reparameterize` 里用的关系一致。

**为什么要对 KL 项做缩放。** 代码额外把 `kl_loss` 除以 `IMG_CH * IMG_SIZE * IMG_SIZE`（即 12,288，和[「为什么要压缩」](../lectures/lecture-02-encode-and-dynamics/01-encoding#为什么要压缩)里提到的像素数一致）。重建 MSE 本来就是对 12,288 个像素取平均，而 KL 只是对 32 个潜在维度求和，如果不缩放，KL 项会比重建项小几个数量级，训练时几乎不起作用。除以同样的像素数，就能让两项在应用 `kl_weight` 之前处于可比的量级。

下面的健全性检查在训练开始前跑一次前向传播，用来确认损失函数能正确执行、输出形状正确；这里打印出的 KL 损失应该接近 0，因为未训练的编码器的 `mu` 和 `log_var` 都在初始化附近的 `(0, 0)`，恰好接近标准正态先验，这只是巧合，不是学到的结果。"""
)

training_loop_md = md_cell(
"""## 4. 训练循环

除了标准的 `EPOCHS`/`LR` 之外，这里有三个超参数值得注意：

- **`KL_WEIGHT = 3.0`** 是 β-VAE 里的 β 系数：把 KL 项乘以这个权重后再加到重建损失上。大于 1 的权重会把潜在分布往 $\\mathcal{N}(0, I)$ 推得比朴素 ELBO 更紧，代价是重建清晰度会打一点折扣。这直接影响第 5 节：如果潜在空间没有充分接近标准正态，从 `z ~ N(0, I)` 采样再解码就会得到不连贯的图像，因为解码器训练时从没见过那个区域的点。代码里的注释解释了为什么这个数据集上选 3.0：因为 KL 项已经除以了像素数，权重给到 3 仍然不会明显影响重建质量，同时能显著收紧潜在分布。
- **混合精度**（`torch.amp.autocast`、`GradScaler`）在 CUDA 上用 float16 做前向传播来加速训练，`GradScaler` 在反向传播前对梯度做缩放，防止 float16 下溢。这是与 VAE 统计性质无关的标准训练加速手段；在 CPU 上或 `USE_CUDA` 为 `False` 时，`autocast` 和 scaler 都是空操作，训练照常以 float32 进行。
- **分别追踪 `recon_loss` 和 `kl_loss`**（而不只是它们的和）是让下一个 cell 的双轴图真正有诊断价值的原因：只看 KL 曲线就能发现**后验坍缩**（posterior collapse）——这是一种失败模式，KL 项在训练早期就跌到接近 0，因为编码器学会了忽略输入，对每张图像都输出 $\\mu \\approx 0, \\log\\sigma^2 \\approx 0$，正好匹配先验，却同时毁掉了重建这条路径。

> **📖 后验坍缩（Posterior Collapse）**：如果看到 `Recon` 在训练最初几轮就停在一个较高的值不再下降，而 `KL` 几乎为零，说明发生了后验坍缩：编码器"偷懒"直接输出接近先验的分布，不再依赖输入图像。常见原因是 `KL_WEIGHT` 设得太高，或者训练早期重建信号太弱。

训练预算和损失权重都已确定，下面把数据集接到 epoch 迭代器和 DataLoader 上，开始按 minibatch 做优化。"""
)

post_loop_md = md_cell(
"训练结束后，把两个损失分量按 epoch 画出来。期望的形状：`重建损失`随着编码器-解码器学会压缩和还原合成形状而稳定下降，`KL 散度`应该更缓慢地下降，最终稳定在一个较小的正值，而不是坍缩为零。如果 KL 曲线在最初几轮就跌到零，那是上面说的后验坍缩的警示信号，不是训练提前收敛的好消息。"
)

recon_md = md_cell(
"损失曲线上的数字只能说明模型在优化某个目标，只有肉眼检查才能确认它优化对了。把保留批次的原始图像和重建图像放在一起比较：健康的编码器-解码器组合应该保留形状、颜色和大致位置，尽管 32 维的潜在瓶颈（相对于 12,288 维的输入像素）注定无法做到像素级的完美还原。"
)

latent_viz_md = md_cell(
"""## 5. 潜在空间可视化

**潜在遍历（latent traversal）**。把 `z` 的所有维度固定为 0，只让其中一维从 -2 扫到 2，逐个解码，测试的正是 [VAE 直觉](../lectures/lecture-02-encode-and-dynamics/01-encoding#vae-直觉学会压缩与重建)一节里说的**连续性**：潜在空间中相邻的点应该解码出视觉上相似的图像，`z` 平滑变化时不应该出现突变。这正是 KL 项的作用：如果没有它把后验拉向同一个密集分布的区域（$\\mathcal{N}(0, I)$），编码器完全可以把不同图像散布到潜在空间里互不相连、形状随意的区域，在它们之间插值就可能经过解码出乱码的地带。

观察每个被扫描的维度是否只改变一个可辨识的属性（位置、大小或颜色），同时让其他属性大致不变。朴素的 ELBO 目标并不保证完美的**解耦**（每个维度都精确对应一个独立因素，这是 β-VAE、FactorVAE 等后续工作要解决的问题），所以在这个简单数据集上，维度 0-2 出现部分或混合的效果是预期之中的，不是 bug。"""
)

prior_sample_md = md_cell(
"""**先验采样（prior sampling）**。这个 cell 完全绕开编码器，直接采样 `z ~ N(0, I)` 再解码。这是比上面的遍历更严格的检验：它检查的是**整个**训练后的潜在分布是否匹配标准正态先验，而不只是前面扫描过的三个维度。如果第 4 节的重建结果看起来不错，但这里的随机采样解码出来一团糟，说明编码器学到了对真实图像有效的压缩，但并没有让 $p(z) = \\mathcal{N}(0, I)$ 在所有区域都成立，也就是说 KL 项相对重建项来说太弱了。直接的修法是调高 `KL_WEIGHT` 重新训练，代价是重建清晰度会下降；这种「重建 vs. 匹配先验」的张力正是第 3 节里介绍的那对目标的平衡。"""
)

checkpoint_md = md_cell(
"""## 保存权重文件

将权重文件保存为 `vae_encoder.pt`，供 P02 和 P03 复用编码器权重。一次良好的训练应以较低的重建损失、视觉上连贯的重建结果，以及对解码形状有可预测影响的潜在遍历作为收尾。P02 通过 `model.encoder` 加载这份权重，把它当作一个冻结（不参与训练）的特征提取器：把每个原始观测 `o_t` 映射成潜在向量 `z_t`，动力学模型（GRU、MDN-RNN、RSSM）再学习预测它随时间的演化，这正是 [Dreamer 流程小结](../lectures/lecture-02-encode-and-dynamics/03-dynamics-dreamer-series#编码器在-dreamer-中的桥梁角色) 里描述的编码器到动力学模型的交接。"""
)

# Build new cell list, inserting the demo pair after index 12 and replacing the rest by content match.
new_cells = cells[:13]  # 0..12 unchanged (up through VAE class cell)
new_cells += [demo_md, demo_code]
new_cells += [elbo_md, cells[14]]                # ELBO markdown + elbo_loss code (unchanged)
new_cells += [training_loop_md, cells[16]]       # training loop intro + training loop code
new_cells += [post_loop_md, cells[18]]           # post loop plot intro + plot code
new_cells += [recon_md, cells[20]]               # reconstruction compare intro + code
new_cells += [latent_viz_md, cells[22]]          # latent viz header + code
new_cells += [prior_sample_md, cells[24]]        # prior sampling intro + code
new_cells += [checkpoint_md, cells[26]]          # checkpoint save + code

nb["cells"] = new_cells
with open(path, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print("done, total cells:", len(new_cells))
