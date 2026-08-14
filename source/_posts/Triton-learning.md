---
title: Triton_learning
date: 2026-08-06 22:13:33
categories: [学习]
tags: [学习, GPU, ai infra]
index_img: /img/covers/triton.svg
---

## GPU简介

### 1. CPU vs GPU 的比喻

- **CPU**：像一个数学教授，非常聪明，能做复杂的计算，但只有十几双手（核心数少，比如8核、16核）。
- **GPU**：像1000个小学生，每个只会做简单的加法，但人数超多（几千个核心），可以同时处理大量简单任务。

**向量加法**就是典型的"简单任务重复无数遍"，特别适合GPU。

### 2. GPU 的结构

GPU 上有成千上万个 **线程（thread）**，每个线程做一点点工作。这些线程被组织成：

- **线程（Thread）**：最小的执行单元，就像一个小学生。
- **线程块（Block）**：一组线程的集合，就像一个小队。
- **网格（Grid）**：所有线程块组成的整体，就像整个学校。

![GPU 的线程、线程块与网格结构](/img/posts/triton-gpu-structure.png)



![线程组织方式示意图](/img/posts/triton-grid-block-thread.png)

![稀疏注意力示意图](/img/posts/triton-sparse-attention.jpg)

![向量加法并行化示意图](/img/posts/triton-vector-add.png)
