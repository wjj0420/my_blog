---
title: infra规划
date: 2026-08-12 21:15:46
categories: [学习]
tags: [AI Infra, 学习路线]
index_img: /img/covers/infra.svg
mermaid: true
---

# AI Infra 学习与研究路线

# 1. 总体目标

### 目标定位

未来 2～3 年主攻：

> **分布式训练与通信优化 + LLM 推理 Infra**

最终形成：

```
                    AI Infra
                       │
          ┌────────────┴────────────┐
          ↓                         ↓
   Distributed Training        LLM Inference
          │                         │
   ┌──────┼──────┐            ┌─────┼──────┐
   ↓      ↓      ↓            ↓     ↓      ↓
 CUDA   NCCL  Megatron       vLLM  KVCache  PD分离
          │      │                  │
       DeepSpeed │              Spec Decode
                 │                  │
                 └────────┬─────────┘
                          ↓
                    GPU / RDMA
                          ↓
                       Network
```

最终目标不是成为单纯的 CUDA 工程师、网络工程师或大模型应用开发者，而是：

> **懂 GPU、懂分布式系统、懂 LLM、懂网络的 AI Systems / AI Infra Engineer。**

------

# 2. 总体技术栈

## 2.1 GPU 底层

需要掌握：

- GPU Architecture
- SM
- Thread
- Warp
- Block
- Grid
- Register
- Shared Memory
- L1 Cache
- L2 Cache
- HBM
- Tensor Core
- CUDA Core
- CUDA Stream
- CUDA Event
- Memory Coalescing
- Occupancy
- Warp Divergence
- Register Pressure

重点理解：

```
HBM
 ↓
L2 Cache
 ↓
L1 / Shared Memory
 ↓
Register
 ↓
CUDA Core / Tensor Core
```

核心问题：

> 为什么一个 CUDA Kernel 会慢？

需要能够从以下几个方面分析：

```
Compute
Memory Bandwidth
Memory Access
Occupancy
Warp Divergence
Register Pressure
Shared Memory
```

------

# 3. Phase 0：CUDA 与 GPU 基础

## 目标

建立 GPU 编程和 GPU 执行模型基础。

## 学习内容

### GPU Architecture

掌握：

- CPU vs GPU
- SM
- Warp
- Thread
- Block
- Grid
- Register
- Shared Memory
- L1 / L2
- HBM
- Tensor Core

### CUDA

能够自己实现：

```
Vector Add
Reduction
Softmax
RMSNorm
LayerNorm
Matrix Multiplication
Attention
```

重点理解：

```
Thread
 ↓
Warp
 ↓
Block
 ↓
SM
 ↓
GPU
```

以及：

```
Global Memory
 ↓
L2
 ↓
L1 / Shared Memory
 ↓
Register
```

## 阶段产出

至少完成：

- CUDA Vector Add
- CUDA Reduce
- CUDA Softmax
- CUDA RMSNorm
- CUDA MatMul
- 简单 Attention Kernel

并使用 Benchmark 分析：

```
Latency
Throughput
Memory Bandwidth
FLOPS
Occupancy
```

------

# 4. Phase 1：集合通信

## 目标

理解 GPU 之间是如何进行数据交换的。

必须掌握：

```
AllReduce
AllGather
ReduceScatter
Broadcast
AllToAll
AllToAllv
```

重点理解：

### Ring AllReduce

```
GPU0 → GPU1 → GPU2 → GPU3
 ↑                       ↓
 └───────────────────────┘
```

理解：

```
Latency
Bandwidth
Message Size
GPU Number
Topology
```

之间的关系。

------

# 5. Phase 2：NCCL

这是整个训练通信方向的核心。

## 需要掌握

```
NCCL Communicator
NCCL Topology
NCCL Channel
NCCL Ring
NCCL Tree
NCCL Transport
CUDA IPC
NVLink
PCIe
RDMA
InfiniBand
RoCE
```

重点研究：

### GPU 拓扑

例如：

```
GPU0 ─ NVLink ─ GPU1
 │               │
PCIe            PCIe
 │               │
NIC0            NIC1
```

需要理解：

> GPU 到 GPU 的实际通信路径是什么？

以及：

```
GPU
 ↓
NVLink / PCIe
 ↓
NIC
 ↓
RDMA
 ↓
Switch
 ↓
NIC
 ↓
GPU
```

------

# 7. Phase 3：Megatron-LM

## 目标

理解大模型分布式训练。

------

## 7.1 Data Parallel

```
GPU0 → Model Copy
GPU1 → Model Copy
GPU2 → Model Copy
GPU3 → Model Copy

        ↓

Gradient AllReduce
```

重点：

- Data Parallel
- Gradient Synchronization
- AllReduce

------

## 7.2 Tensor Parallel

```
             Linear
               │
       ┌───────┼───────┐
       ↓       ↓       ↓
     GPU0    GPU1    GPU2
```

重点：

- Tensor Parallel
- AllGather
- ReduceScatter
- AllReduce

------

## 7.3 Pipeline Parallel

```
GPU0        GPU1        GPU2        GPU3

Layer1      Layer5      Layer9      Layer13
Layer2      Layer6      Layer10     Layer14
Layer3      Layer7      Layer11     Layer15
Layer4      Layer8      Layer12     Layer16
```

重点：

- Pipeline Parallel
- Pipeline Bubble
- 1F1B
- Interleaved Pipeline

------

## 7.4 3D Parallelism

最终理解：

```
DP × TP × PP
```

例如：

```
DP = 4
TP = 4
PP = 4

4 × 4 × 4 = 64 GPUs
```

核心问题：

> 为什么 GPU 数量增加后，训练效率不会线性增加？

------

# 8. Phase 4：DeepSpeed

重点掌握：

```
ZeRO-1
ZeRO-2
ZeRO-3
```

理解：

```
Parameters
Gradients
Optimizer States
```

如何在 GPU 之间切分。

进一步学习：

```
ZeRO
Offload
Activation Checkpointing
Communication Overlap
```

------

# 9. 第一阶段科研方向：训练通信优化

不建议一开始直接研究：

> 万卡集群通信优化

而应该从小规模实验开始。

------

## 9.1 拓扑感知通信优化

研究：

```
GPU Topology
      ↓
Communication Path
      ↓
Collective Algorithm
      ↓
Communication Performance
```

例如比较：

```
PCIe
NVLink
RDMA
```

不同拓扑下：

```
Latency
Bandwidth
Scaling Efficiency
```

------

## 9.2 计算与通信重叠

目标：

```
Compute
   │
   ├───────────────┐
   ↓               ↓
  GEMM          AllReduce
   │               │
   └──────┬────────┘
          ↓
        Next
```

通过：

- CUDA Stream
- NCCL
- Pipeline
- Asynchronous Communication

实现：

> **Communication Overlap**

重点指标：

```
Training Throughput
Communication Time
GPU Utilization
Scaling Efficiency
```

------

# 10. Phase 5：LLM Inference

训练方向达到一定程度之后进入推理。

推荐学习顺序：

```
vLLM
 ↓
PagedAttention
 ↓
KV Cache
 ↓
Continuous Batching
 ↓
Tensor Parallel
 ↓
Speculative Decoding
 ↓
PD Disaggregation
 ↓
KV Cache Transfer
 ↓
Inference Network
```

------

# 11. Phase 6：vLLM

首先把 vLLM 的整体执行链路搞懂：

```
Request
   ↓
Tokenizer
   ↓
Scheduler
   ↓
Continuous Batch
   ↓
Model Runner
   ↓
Attention
   ↓
KV Cache
   ↓
GPU
```

重点研究：

- Scheduler
- Continuous Batching
- KV Cache Manager
- PagedAttention
- GPU Worker
- Model Runner
- Memory Management

------

# 12. Phase 7：KV Cache

KV Cache 是推理 Infra 的核心。

Transformer Attention：

```
K = XWk
V = XWv
```

生成新 Token：

```
K_new
V_new
```

不断追加：

```
KV Cache

Token 1
Token 2
Token 3
...
Token N
```

Context 越长：

```
KV Cache
    ↓
GPU HBM
    ↓
Memory Pressure
    ↓
OOM
```

因此需要研究：

```
PagedAttention
KV Cache Paging
KV Cache Offloading
KV Cache Compression
KV Cache Sharing
KV Cache Eviction
KV Cache Placement
```

------

# 13. Phase 8：Continuous Batching

理解 LLM Decode 的动态 Batch。

例如：

```
Request A → Token
Request B → Token
Request C → Token
Request D → EOS
Request E → Token
```

下一轮：

```
A B C E
```

重新形成 Batch。

核心：

> 每一轮 Decode 都可能由不同请求组成动态 Batch。

重点指标：

```
Throughput
TTFT
TPOT
GPU Utilization
Batch Size
Scheduling Overhead
```

------

# 14. Phase 9：Speculative Decoding

基本结构：

```
              Draft Model
                   ↓
            t1 t2 t3 t4
                   ↓
             Target Model
                   ↓
               Verify
                   ↓
            Accept / Reject
```

传统生成：

```
Target
 ↓
1 Token
 ↓
Target
 ↓
1 Token
 ↓
Target
```

Speculative Decoding：

```
Draft
 ↓
多个 Token
 ↓
Target 一次验证
```

重点研究：

```
Acceptance Rate
Draft Latency
Target Latency
Memory Overhead
Throughput
TTFT
TPOT
```

------

# 15. Phase 10：Disaggregated Inference

这是重点方向。

传统：

```
Prefill + Decode
       ↓
    同一 GPU
```

分离式推理：

```
             Network
                │
       ┌────────┴────────┐
       ↓                 ↓
Prefill Cluster     Decode Cluster
       │                 │
    GPU GPU GPU        GPU GPU GPU
       │
       ↓
    KV Cache
       │
       └──────→ Network
                    ↓
                 Decode
```

即：

> **Prefill 和 Decode 分离部署。**

------

# 16. 为什么 PD Disaggregation 特别适合通信背景

Prefill：

```
输入 Prompt
    ↓
Prefill GPU
    ↓
产生 KV Cache
```

然后：

```
KV Cache
    ↓
Network
    ↓
Decode GPU
```

这会产生大量系统问题：

### 网络带宽

```
KV Cache Size
      ↓
Transfer Bandwidth
      ↓
Transfer Latency
```

### 网络拥塞

```
多个 Prefill
      ↓
KV Transfer
      ↓
Network Congestion
```

### GPU 调度

```
Prefill GPU
     ↓
哪个 Decode GPU？
```

### KV Cache Placement

```
KV Cache
   ↓
放在哪里？
   ↓
哪个 GPU？
   ↓
什么时候迁移？
```

因此：

> **PD Disaggregation = LLM + GPU + Network + RDMA + Distributed Systems**

非常适合网络背景。

------

# 17. 第二阶段科研方向：Inference + Network

重点考虑：

> **Network-aware LLM Inference**

整体结构：

```
                LLM Inference
                     │
          ┌──────────┴──────────┐
          ↓                     ↓
       Prefill                Decode
          │                     │
       KV Cache                 │
          │                     │
          └──────→ Network ─────┘
                      │
                     RDMA
                      │
                  Scheduler
```

可以研究：

- Network-aware Scheduling
- KV Cache Placement
- KV Cache Routing
- RDMA KV Transfer
- Congestion-aware Inference
- Topology-aware Inference
- GPU-aware Scheduling
- Prefill/Decode Load Balancing

------

# 18. 推荐的 4 个科研项目

## 项目 1：NCCL Topology Benchmark

### 目标

研究不同 GPU / 网络拓扑下集合通信性能。

测试：

```
2 GPU
4 GPU
8 GPU
```

以及：

```
PCIe
NVLink
RDMA
```

测量：

```
Latency
Bandwidth
Scaling Efficiency
```

### 最终产出

```
NCCL Benchmark
+
Topology Analysis
+
Performance Model
```

------

# 19. 项目 2：Megatron 通信与计算 Overlap

### Baseline

```
GEMM
 ↓
AllReduce
 ↓
GEMM
 ↓
AllReduce
```

### Optimization

```
GEMM ────────────────
      AllReduce ────────
            GEMM ─────────
```

利用：

- CUDA Stream
- NCCL
- Async Communication

优化：

```
Communication / Computation Overlap
```

最终测：

```
Training Throughput
GPU Utilization
Communication Ratio
Scaling Efficiency
```

------

# 20. 项目 3：KV Cache 优化

研究：

```
KV Cache
    ↓
Memory Pressure
    ↓
Eviction
    ↓
Offload / Migration
```

重点：

- KV Cache Placement
- KV Cache Eviction
- KV Cache Offloading
- KV Cache Compression

测试：

```
Short Context
Medium Context
Long Context
```

指标：

```
TTFT
TPOT
Throughput
GPU Memory Usage
Cache Hit Rate
```

------

# 21. 项目 4：PD Disaggregation + RDMA

最推荐作为核心科研项目。

架构：

```
            ┌──────────────┐
            │    Client    │
            └──────┬───────┘
                   ↓
          ┌─────────────────┐
          │ Prefill Cluster │
          └────────┬────────┘
                   ↓
               KV Cache
                   ↓
             RDMA Network
                   ↓
          ┌─────────────────┐
          │ Decode Cluster  │
          └────────┬────────┘
                   ↓
               Output
```

测试：

```
TCP
vs
RDMA
```

不同网络：

```
1 Gbps
10 Gbps
25 Gbps
100 Gbps
```

研究：

```
KV Transfer Latency
TTFT
TPOT
Throughput
GPU Utilization
Network Utilization
```

进一步做：

> **Network-aware KV Cache Scheduling**

这个方向非常适合作为论文/毕业课题。

------

# 22. 12 个月学习规划

| 时间      | 主攻内容                        | 目标              |
| --------- | ------------------------------- | ----------------- |
| 0～2 月   | CUDA + GPU Architecture         | 掌握 GPU 执行模型 |
| 2～4 月   | Collective Communication + NCCL | 掌握 GPU 通信     |
| 4～6 月   | Megatron + DeepSpeed            | 掌握分布式训练    |
| 6～8 月   | vLLM + KV Cache                 | 掌握 LLM 推理     |
| 8～10 月  | Speculative Decoding            | 掌握推理优化      |
| 10～12 月 | PD Disaggregation + RDMA        | 形成科研方向      |

------

# 23. 每个阶段的学习方法

不要：

```
看论文
 ↓
看源码
 ↓
继续看论文
 ↓
继续看源码
```

而应该：

```
理论
 ↓
最小 Demo
 ↓
源码
 ↓
Benchmark
 ↓
性能分析
 ↓
修改
 ↓
实验
```

例如学习 NCCL：

```
AllReduce 原理
      ↓
自己实现简单 AllReduce
      ↓
NCCL API
      ↓
NCCL 源码
      ↓
NCCL Benchmark
      ↓
改变 Message Size
      ↓
改变 GPU 数量
      ↓
分析 Scaling
```

------

# 24. 实验资源要求

不需要一开始就有万卡集群。

建议最低：

```
2 GPU
+
CUDA
+
NCCL
```

比较理想：

```
4～8 GPU
+
NVLink / PCIe
+
RDMA
+
25/100GbE
```

研究万卡问题时可以通过：

```
Small-scale Benchmark
+
Performance Model
+
Simulation
+
Extrapolation
```

研究大规模系统。

------

# 25. 未来需要掌握的技术栈

## GPU

```
CUDA
CUDA Graph
Triton
Tensor Core
FlashAttention
Kernel Optimization
```

## Distributed Training

```
PyTorch Distributed
NCCL
Megatron-LM
DeepSpeed
ZeRO
FSDP
DP
TP
PP
EP
```

## LLM Inference

```
vLLM
SGLang
TensorRT-LLM
PagedAttention
KV Cache
Continuous Batching
Speculative Decoding
PD Disaggregation
```

## Network

```
RDMA
RoCE
InfiniBand
NVLink
PCIe
NCCL Network
Topology
Congestion Control
```

## System

```
Linux
C/C++
Python
CUDA
Docker
Kubernetes
Distributed Systems
Performance Profiling
```

------

# 26. 就业方向

最终可以对应：

### AI Infra

```
LLM Inference Engineer
AI Infra Engineer
Distributed Training Engineer
GPU Cluster Engineer
```

### GPU / Systems

```
CUDA Engineer
GPU Optimization Engineer
Distributed Systems Engineer
```

### Network + AI

```
AI Network Engineer
Inference Network Engineer
RDMA Engineer
GPU Network Engineer
```

------

# 27. 技术能力优先级

建议优先级：

```
★★★★★
LLM Inference / AI Infra

★★★★★
Distributed Training / NCCL

★★★★☆
CUDA / GPU Optimization

★★★★☆
RDMA / GPU Network

★★★☆☆
DeepSpeed / Megatron 深度源码

★★★☆☆
Speculative Decoding

★★☆☆☆
纯 LLM Application / Agent
```

对于个人背景而言，**CUDA + NCCL + RDMA + LLM Inference** 的组合尤其值得建立。

------

# 28. 最终能力画像

最终希望形成：

```
                    AI Systems Engineer
                           │
              ┌────────────┴────────────┐
              ↓                         ↓
       Distributed Training        LLM Inference
              │                         │
          Megatron                  vLLM/SGLang
              │                         │
         DeepSpeed                  KV Cache
              │                         │
            NCCL                  Spec Decode
              │                         │
              └────────────┬────────────┘
                           ↓
                         CUDA
                           ↓
                          GPU
                           ↓
                    NVLink / PCIe
                           ↓
                       RDMA/NIC
                           ↓
                        Network
```

核心能力：

> **能够从 LLM 上层 workload 一直追到底层 GPU、通信库、RDMA 和网络，并定位系统瓶颈、设计优化方案、完成实验验证。**

------

# 29. 最终主线

整个研究生阶段可以始终围绕下面这条主线：

```
CUDA
 ↓
GPU Architecture
 ↓
Collective Communication
 ↓
NCCL
 ↓
Megatron / DeepSpeed
 ↓
Distributed Training
 ↓
vLLM
 ↓
KV Cache
 ↓
Speculative Decoding
 ↓
PD Disaggregation
 ↓
RDMA
 ↓
Inference Network
 ↓
Network-aware LLM Inference
```

最终形成一个明确的研究标签：

> **LLM Systems / AI Infra + GPU Communication + RDMA**
