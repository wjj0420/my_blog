---
title: FlashAttention
date: 2026-08-04 15:17:24
categories: [学习]
tags: [LLM, ai infra]
---

# FlashAttention

## 一、Introduction

为了加快LLM的训练和推理速度，针对transformer注意力机制的特点和GPU等硬件结构，通过**分块**和**重计算**来**减少HBM读写次数**,进而加快注意力计算的一种优化算法。

I

## 二、 Background

![image-20260806093533372](https://picgocloud.com/m/a3ece8f9-4162-4529-a897-2282a1d7427d.png)

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

加速内存密集型操作的常用方法是**Kernel fusion**，但是模型训练的context下，一写中间值需要写回HBM进行梯度计算和反向传播。

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

## 三、 FlashAttention算法的设计

**设计目标：**给定 HBM 中的输入$\mathbf{Q},\mathbf{K},\mathbf{V} \in \mathbb{R}^{N \times d}$，计算注意力输出$ \mathbf{O} \in \mathbb{R}^{N \times d}$ 并将其写入 HBM。目标是**减少 HBM 访问量**，使访问量少于$\mathbf{O(N^2)}$。

**方案：**使用了**分块**（tiling）和**重计算**（recomputation）两种方法来实现。

分块计算的详细流程：

![image-20260806132829825](https://picgocloud.com/m/b4177fad-0e0d-4c42-a4dd-4a00312473b9.png)

