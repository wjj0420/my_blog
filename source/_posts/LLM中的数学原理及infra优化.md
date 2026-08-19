---
title: LLM中的数学原理及infra优化
date: 2026-08-09 19:05:17
categories: [学习]
tags: [学习, LLM, ai infra]
index_img: /img/covers/llm-math.svg
---

## 引言

本文将拆解大模型中几个核心操作（RMSNorm、Softmax、Causal Mask、Sampling）背后的数学与 Infra 优化逻辑。看完你会发现，Infra 优化，本质上就是在用**数学上的等价变换**，或者**对精度的适度妥协**，去换取更高的硬件利用率和极致的推理速度。

## 1.RMSNorm - 均方根归一化

大语言模型（Transformer 结构）通常包含数十甚至上百个堆叠的隐藏层（如 Transformer 结构）。输入张量（Tensor）在经过连续的矩阵乘法和加法操作后，其数值的分布范围会发生剧烈的变化。

这种数值大小的不可控会导致两个严重的工程和算法问题：

1. 算法收敛困难：数值变得过大或过小会影响训练的稳定性：极端数值要么会落入激活函数的饱和区导致梯度消失，要么会顺着网络不受控地放大引发梯度爆炸或硬件溢出。
2. 硬件层面的溢出与截断：在当前主流的 GPU 推理和训练中，为了追求极致的吞吐，底层计算会使用低精度浮点格式（如 FP16 或 BF16）。FP16的数值范围小，容易溢出，BF数值范围大，但是精度低（大数与小数加和时容易因为四舍五入将小数截断）。

![RMSNorm 归一化示意图](llm-rmsnorm.png)

这两者都会导致模型输出乱码或训练彻底崩溃。为了保证大模型在深层网络中的数值稳定性，研究人员在架构中引入了特征归一化（Normalization）机制（例如 LayerNorm、RMSNorm）。其核心目的，是在数据的层间传递过程中，对其数值分布进行强制的缩放与平移，将其约束在一个标准、安全的物理尺度内, 防止方差膨胀引发的溢出。

神经网络真正关心的并非数值的绝对大小，而是特征之间的**相对差异**，故可以平移。

### 1.1 LayerNorm

LayerNorm 的核心思想是对同一个 Token 内的所有特征维度（hidden size, *d*）进行标准化，使其均值为 0，方差为 1。

与LayerNorm对应的是BatchNorm, LayerNorm/RMSNorm 是严格在 **Token 级别（Hidden Size 维度）** 闭环的。无论外部的 Batch Size 怎么变，无论旁边并行的请求是写诗还是写代码，每一个 Token 自身的归一化结果不会发生变化。训推一致。

 git add .git commit -m "新文章"git pushbash

#### 数学公式

$$
y = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} \odot \gamma + \beta
$$

符号说明：

- $\odot$：**哈达玛积（逐元素相乘）**
- $\mu$：均值，$\sigma^2$：方差，$\epsilon$：防止分母为 0 的极小常数
- $\gamma$：缩放参数，$\beta$：偏移参数

标准化并做**仿射变换**（乘以可学习参数 *γ*，加上偏置 *β* ，为了**增强特征表达能力**）。

#### Infra视角

LayerNorm 是一个典型的 **Memory-bound（访存密集型）** 算子。它的计算包含了两次 Global Reduction（全局规约）操作。

最致命的是**数据依赖**：你必须先完整遍历一次数据算出均值 $\mu$，然后才能用 $\mu$ 去遍历第二次算方差 $\sigma^2$。

在 GPU 上，这意味着：
- 更复杂的线程同步逻辑；
- 在 Kernel 未极致融合时，需要多次往返读写 HBM（显存），极大地浪费了宝贵的内存带宽。

## 1.2 RMSNorm (Root Mean Square Normalization)

RMSNorm 的作者（Biao Zhang 等人，2019）通过实验发现：LayerNorm 之所以有效，主要是因为缩放（Scaling，即除以标准差）的作用，而平移（Mean-centering，即减去均值 μ）对模型收敛的贡献微乎其微。

既然均值没用，那就直接砍掉它。放弃了计算均值，只保留对向量 RMS 尺度的归一化。当然也有说法：LayerNorm 强行减均值（取平均归零）这个动作，其实是在人为地阉割模型的表达能力。

### 数学公式

其中均方根（RMS）的计算：

$$
\text{RMS}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2}
$$

$$
\hat{x}_i = \frac{x_i}{\text{RMS}(x) + \epsilon} \cdot \gamma
$$

其中 $\epsilon = 10^{-6}$，即代码默认值：`eps = 1e-6`。实际运行值取决于具体模型的 `config.rms_norm_eps` 配置，通常为 `1e-5` 或 `1e-6`。$\epsilon$ 的作用是防止除以零，保证数值稳定性，其值非常小，对最终计算结果的影响微乎其微。

### Infra 视角

- **打破数据依赖**：RMSNorm 不需要算 $\mu$，直接计算每个元素的平方和即可。这意味着只需要一次单向的 Reduction 操作。
- **极致的访存优化**：在编写 Triton 或 CUDA Kernel 时，RMSNorm 可以非常丝滑地在一个 Block 内完成「数据加载 (SRAM) → 平方求和 → 广播 → 缩放」的流水线，中间变量极少。
- **计算量减少**：省去了大量减法操作。

主流模型在使用 RMSNorm 时，通常连后处理的偏置项 $\beta$ 也一并去掉了（即无 Bias 线性层），进一步减少了参数加载和 element-wise 加法的开销。

> 值得注意的是：相对 LayerNorm，Bias 也被去掉了：RMSNorm 常常只保留 $\gamma$ 而去掉 $\beta$，这不是数学上必然要求，而是现代 LLM 架构中的经验选择。它通常与无 bias Linear、Pre-Norm 残差结构、SwiGLU 等设计共同出现，整体上减少参数与访存，同时保持效果。

也有类似说法：LayerNorm 包含减均值，因此后面紧跟的线性层加 Bias 是有意义的。而 RMSNorm 砍掉了平移，只做纯粹的尺度缩放，如果它后面紧跟的 Linear 层仍保留 Bias，就破坏了 RMSNorm 抛弃绝对中心、只维持相对尺度的初衷。

当今的主流开源大模型不仅仅 RMSNorm 去掉了 $\beta$，而是几乎所有的 Linear 层都去掉了 Bias：

- `q_proj`, `k_proj`, `v_proj`, `o_proj` 没有 Bias
- MLP 的 `gate_proj`, `up_proj`, `down_proj` 也没有 Bias

对此有很多解释：

- **训练更稳定**：  
  > "No biases were used in any of the dense kernels or layer norms. We found this to result in increased training stability for large models." —— *PaLM 论文*
- **架构冗余**：RMSNorm 后紧接 Linear 时，bias 的位移作用会被下一个 Norm 的 $\gamma/\beta$ 吸收；SwiGLU 的门控本身也提供了类似 bias 的自由度。bias 在现代架构里已被架构本身替代。
- **Infra 友好**：少一次 add 与 bias load。

但要我说，就是实验有效 + Infra 友好——专业的说法，哈哈：从工程视角看，这类设计往往并非单一数学原则的必然结果，而是效果、稳定性、实现成本和硬件效率共同权衡后的经验选择。

---

### LayerNorm vs RMSNorm（底层实现视角）

LayerNorm 可借助 $\mathbb{E}[X^2] - (\mathbb{E}[X])^2$ 公式与 Kernel 融合，实现只需访问一次 HBM。而 RMSNorm 进一步斩断了均值计算，压缩了 SRAM 占用和 ALU 指令周期。

LayerNorm 虽然理论上需要均值和方差两步，但在高性能实现中并不必然需要两次 HBM 访问：可以在一次 HBM load 中同时累计 $\sum x$ 和 $\sum x^2$，甚至使用 Welford 算法提升数值稳定性。但相较 RMSNorm，LayerNorm 仍需要维护均值相关统计量，并在归一化阶段执行额外的减均值操作，因此寄存器压力、规约状态、ALU 指令数都更高。

**RMSNorm 相对于 LayerNorm 的收益：**

- 减少了寄存器 / SRAM 的占用
- 节省了大量的 ALU（逻辑运算单元）指令，特别是消除了对全部元素的减法（减去均值，element-wise）操作

其实在当今主流的 Fused CUDA / Triton Kernel 中，LayerNorm 也是可以做到单次 HBM 访存（1-Pass）的。在数学上，方差可以等价展开为：

$$
\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$

在 GPU 寄存器 / SRAM 层面，我们在单一的一个 Block 遍历输入张量时，可以同时累计 $\sum x$ 和 $\sum x^2$。由于大模型的 Hidden Size（如 4096 或 8192）对应的字节数（约 8–16 KB）完全可以被塞进单个 SM 的 Shared Memory 中，因此无论 LayerNorm 还是 RMSNorm，现代算子在 HBM 层面都是只读一遍、写一遍。

虽然 $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$ 能实现 1-pass，但在 FP16 或 BF16 精度下，如果 $\mathbb{E}[X^2]$ 与 $(\mathbb{E}[X])^2$ 的值非常接近，相减容易引发灾难性抵消（Catastrophic Cancellation），导致方差精度丢失甚至计算出负数（最后开根号出 NaN）。因此在实际的 Kernel（如 Apex 或 Triton 内部）中，有时会采用 Welford 算法来兼顾 1-pass 和数值稳定性，或者在累加时强制转换到 FP32 进行计算。

---

### $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$ 的推导逻辑

假设数据的平均值为 $\mu$（即 $\mu = \mathbb{E}[X]$）。方差的原始定义是：每个数减去平均值的平方，再求平均。即：

$$
\text{Var}(X) = \mathbb{E}[(X - \mu)^2]
$$

根据 $(a - b)^2 = a^2 - 2ab + b^2$，展开得：

$$
\begin{aligned}
\text{Var}(X)
&= \mathbb{E}[X^2 - 2X\mu + \mu^2] \\
&= \mathbb{E}[X^2] - \mathbb{E}[2X\mu] + \mathbb{E}[\mu^2]
\end{aligned}
$$

- $\mathbb{E}[2X\mu] = 2\mu \cdot \mathbb{E}[X]$，而 $\mathbb{E}[X] = \mu$，所以 $\mathbb{E}[2X\mu] = 2\mu^2$
- $\mathbb{E}[\mu^2]$：因为 $\mu^2$ 是常数，常数的期望仍是它本身，即 $\mathbb{E}[\mu^2] = \mu^2$

代入得：

$$
\text{Var}(X) = \mathbb{E}[X^2] - 2\mu^2 + \mu^2 = \mathbb{E}[X^2] - \mu^2
$$

而 $\mu = \mathbb{E}[X]$，于是得到最终公式：

$$
\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$
