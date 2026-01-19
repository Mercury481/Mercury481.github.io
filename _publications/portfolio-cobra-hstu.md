---
title: "基于 HSTU 与 COBRA 架构的下一代生成式推荐系统"
excerpt: "探索推荐系统的范式转移：从 TIGER 的纯语义 ID 生成，进化为 '稀疏遇见稠密' (Sparse Meets Dense) 的 COBRA 级联架构。本项目使用高性能 HSTU 作为骨干网络，在 ML-20M 数据集上实现了高效的端到端生成式检索。"
collection: portfolio
---

## 项目概览

本项目复现并优化了基于 **COBRA** 框架的生成式推荐系统。针对传统 ID 推荐和 TIGER 等早期生成式模型的局限性，本项目引入了 **SID2VID** 机制，并采用 Meta 最新的 **HSTU (Hierarchical Sequential Transduction Unit)** 替代传统 Transformer 作为序列建模的主干网络。

## 核心技术栈

* **模型架构**: HSTU (Hierarchical Sequential Transduction Unit) - *比 Transformer 快 5-10 倍的训练速度与更优的收敛性*。
* **生成策略**: COBRA / SID2VID (Semantic ID to Vector ID) - *级联生成稀疏语义 ID 与稠密向量 ID*。
* **数据索引**: Balanced RQ-KMeans - *生成层次化、平衡的语义码本 (Codebook)*。
* **数据集**: MovieLens-20M (27K+ Items, 20M+ Interactions)。

## 关键改进点 (Impact)

1.  **超越 TIGER 的表达能力**：
    TIGER (Generative Retrieval via Semantic IDs) 虽然开创了语义 ID 生成的先河，但仅依赖离散的 Codebook 会导致细粒度语义信息的丢失。本项目实施的 COBRA 策略，在预测 Semantic ID 后，继续预测 Item 的稠密向量 (Dense Vector)，有效找回了量化过程中丢失的信息，实现了 "Sparse Meets Dense" 的统一。

2.  **HSTU 高性能骨干**：
    使用 HSTU 替换原论文可能使用的标准 Transformer/SASRec。在 ML-20M 这种长序列数据集上，HSTU 利用其独特的注意力机制设计，在保证精度的同时大幅降低了显存占用并提升了推理速度。

3.  **高质量语义聚类**：
    实现了 Balanced K-Means 算法生成 Semantic IDs，解决了传统 RQ-VAE 在长尾物品上 ID 分配不均的问题，将 Codebook 利用率提升至接近 100%。


[阅读详细技术博客](/posts/2026-01-18-cobra-hstu-deep-dive)

