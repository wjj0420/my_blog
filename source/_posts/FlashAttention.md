---
title: FlashAttention
date: 2026-08-04 15:17:24
categories: [学习]
tags: [LLM, ai infra]
index_img: /img/covers/flashattention.svg
---

# FlashAttention

## 一、Introduction

为了加快LLM的训练和推理速度，针对transformer注意力机制的特点和GPU等硬件结构，通过**分块**和**重计算**来**减少HBM读写次数**,进而加快注意力计算的一种优化算法。

## 二、 Background

![GPU 内存层级结构：SRAM / HBM / DRAM](flashattention-memory.png)

### 2.1 硬件特性

GPU的内存结构如上图左侧所示：

**SRAM**是GPU的片上内存，GPU计算时**必须先把数据搬运到SRAM上**才能进行计算。内存最小，但是IO速度最快

**HBM**是高带宽内存，存放模型参数权重、KVcache以及其它计算中间变量。内存较大，IO速度较快。(是否正确)

**DRAM**是主存，cpu进程存放数据的地方。内存大，IO速度慢。

模型运行时，GPUs拥有海量线程来执行一个操作/函数（**kernel**），每个kernel将输入从HBM加载到寄存器和SRAM上，然后计算，最后将输出写回HBM。

### 2.2 操作特性

依据计算和内存访问的平衡，操作可以分为**计算密集型**操作和**内存密集型**操作

计算密集型：操作所需时间主要取决于**算术运算次数**，访问内存只占很少的时间。典型示例为维度很大的矩阵乘法、通道数很多的卷积。

内存密集型：操作所需时间主要取决于**内存访问次数**，计算花费的时间很少。典型示例为逐元素操作（激活函数，dropout）、reduce（求和、Softmax、Batch Normalization、layer Normalization）。

### 2.3 标准Attention实现（简化的核心部分）

输入$\mathbf{Q},\mathbf{K},\mathbf{V} \in \mathbb{R}^{N \times d}$

先计算注意力分数矩阵**S**（每个token的**Q**与所有token的**K**进行计算）
$$
\mathbf{S} = \mathbf{Q}\mathbf{K}^T \in \mathbb{R}^{N \times N}
$$
再计算注意力权重矩阵**P**(对**S**按行进行softmax)
$$
\quad \mathbf{P} = \mathrm{softmax}(\mathbf{S}) \in \mathbb{R}^{N \times N}
$$
最后计算输出矩阵**O**
$$
\quad \mathbf{O} = \mathbf{P}\mathbf{V} \in \mathbb{R}^{N \times d}
$$
标准的Attention实现需要将**S**和**P**存储到HBM中，占用$\mathbf{O(N^2)}$的内存

部分操作例如**softmax**操作是内存密集型操作，大量内存访问导致较慢的运行时间。

还有其它的逐元素操作如掩码（按行）和**P**的丢弃也加剧了这一现象。

## 三、 FlashAttention算法的设计与分析

### 3.1 设计目标

给定 HBM 中的输入$\mathbf{Q},\mathbf{K},\mathbf{V} \in \mathbb{R}^{N \times d}$，计算注意力输出$ \mathbf{O} \in \mathbb{R}^{N \times d}$ 并将其写入 HBM。目标是**减少 HBM 访问量**，**使访问量少于**$\mathbf{\Theta(N^2)}$。

### 3.2 分块

使用了**分块**（tiling）和**重计算**（recomputation）两种方法来实现。

分块计算的详细流程：

![FlashAttention 分块计算流程](flashattention-flow.png)

1. 如图1右侧所示，先从行维度将Q分成$T_r$个块，将K和V都分成$T_c$个块。

关于块大小的选取，原则是**使SRAM能同时放下**$\mathbf{Q_i}$、$\mathbf{K_j}$、$\mathbf{V_j}$、$\mathbf{O_i}$、$\ell_i$、$m_i$的前提下，让**分块的数量尽可能少**。按上述分块大小，所需的总空间为$(2B_c \times d + 2B_r \times d +2B_r)$。上述理论推导时忽略了$\ell_i$、$m_i$等低阶空间复杂度变量，实际工程时会采用保守缩小块大小的策略。

2. 遍历$\mathbf{Q}$的分块，从HBM取一个块$\mathbf{Q_i}$
3. 遍历$\mathbf{K}$、$\mathbf{V}$的分块，从HBM取[$\mathbf{K_j}$,$\mathbf{V_j}$]，注意二者分块的行序列要一致。
4. 在SRAM上计算注意力分数矩阵$\mathbf{S}_{ij} = \mathbf{Q}_i \mathbf{K}_j^T \in \mathbb{R}^{B_r \times B_c}$ 
5. 在SRAM上计算$\mathbf{Q_i}$中选中的行的每行最大值，注意力矩阵逐元素取e指数$\tilde{\mathbf{P}}_{ij} = \exp(\mathbf{S}_{ij} - \tilde{m}_{ij}) \in \mathbb{R}^{B_r \times B_c}$，进行softmax操作的分母$\tilde{\ell}_{ij} = \mathrm{rowsum}(\tilde{\mathbf{P}}_{ij}) \in \mathbb{R}^{B_r}$ 。
6. 更新$\mathbf{Q_i}$中选中行的每行行最大值$m_i$，每行各元素的e指数之和$\ell_i$。
7. 增量式计算更新$\mathbf{Q_i}$中选中的行的注意力输出$\mathbf{O}_i \leftarrow \mathrm{diag}(\ell_i^{\mathrm{new}})^{-1}\big(\mathrm{diag}(\ell_i)e^{m_i - m_i^{\mathrm{new}}}\mathbf{O}_i + e^{\tilde{m}_{ij} - m_i^{\mathrm{new}}}\tilde{\mathbf{P}}_{ij}\mathbf{V}_j\big)$，写入HBM。
8. 将更新后的$m_i$和$\ell_i$写回HBM。

与标准Attention访问次数对比分析：

二者$\mathbf{Q},\mathbf{K},\mathbf{V}$读取次数相同，区别在于FlashAttention不用存$\mathbf{S}$和$\mathbf{P}$，为什么传统attention需要存呢？——因为一般N很大，并且N>>d，SRAM存$\mathbf{Nd}$空间没问题，存不下S和P这种$\mathbf{N^2}$空间的。但是每次多了$\ell_i$、$m_i$的读写，总共需要$4 \times B_r \times T_r \times T_c = 4NT_c$次读写，同时多了$2 \times T_r \times (T_c-1) \times B_r \times d = 2Nd(T_c-1)$次读写，复杂度为$\mathbf{\Theta(N^2d^2/M)}$，而标准的Attention的HBM读写次数复杂度为$\mathbf{\Theta(N^2)}$。

对于典型的d（64-128）和M（约100KB）值，**$d^2$远小于M**，因此FlashAttention所需的HBM访问次数**比标准Attention实现的少很多倍**。

### 3.3 重计算

通过分块，无需存储S和P,但是训练时反向传播需要使用$\mathbf{S}$和$\mathbf{P}$来计算梯度。解决办法是**重新计算**。

通过存储输出$\mathbf{O}$和 softmax 归一化统计量$(m,\ell)$，我们可以在SRAM中轻松地重新计算注意力矩阵$\mathbf{S}$和$\mathbf{P}$。

相比之下，即使FlashAttention有更多的 FLOPs，重新计算通过减少 HBM 访问来加速反向传播。

### 3.4 扩展：Block-Sparse FlashAttention

用一个掩码矩阵$M \in \{0, 1\}^{N / B_r \times N / B_c}$来表示块与块之间是否需要计算，

