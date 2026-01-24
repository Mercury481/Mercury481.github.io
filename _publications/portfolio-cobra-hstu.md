---
title: '超越语义ID：在 HSTU 中复现 COBRA 与交错式生成的探索'
tags:
  - 生成式推荐
  - HSTU
  - COBRA
  - 语义ID
  - 推荐系统
excerpt: "本项目是对 Shao 的 HSTU-Semantic-ID 的深度扩展，旨在将百度 COBRA 架构的序列交错生成机制引入 HSTU 推荐模型。项目的核心创新在于实现了语义 ID (SID) 与向量 ID (VID) 的混合训练，通过交替预测 SID（分类任务）和 VID（度量学习）来增强模型对物品语义与协同信号的双重捕捉。"
collection: portfolio
---

## 项目概览

本项目是对 Shao 的 HSTU-Semantic-ID 的深度扩展，旨在将**百度 COBRA 架构的“序列交错生成**”机制引入 Meta 的 HSTU 推荐模型。项目核心在于打破传统 ID 的限制，实现了语义 ID (SID) 与向量 ID (VID) 的混合训练。通过交替预测 SID（离散分类）和 VID（连续度量），本项目探索了如何在生成式推荐中融合语义理解与协同信号。

## 核心技术栈

基础架构：HSTU, Transformer

生成机制：COBRA 风格的 SID-VID 交错序列生成

损失函数：Sampled Softmax (分类) + In-Batch InfoNCE (对比学习)

SID 生成算法：Balanced K-Means

数据集: MovieLens-20M

## 关键改进点

1. 用于 SID 生成的**平衡 K-Means 算法**： 针对原版 K-Means 容易产生的簇大小不均（马太效应）问题，实现了带有容忍度约束的 Balanced K-Means。通过**迭代优化与强制数据搬运**，确保每一层 Codebook 的分布均匀，显著提升了 SID 的语义分辨率。

2. SID 质量评估体系： 构建了多维度的评估脚本 (new_eval_SID.py)，从**重建损失**(MSE/Cosine)、**码本利用率**、**Token 分布熵**（归一化熵）以及**前缀冲突率**四个维度，量化评估 SID 的生成质量，验证了平衡聚类对特征塌陷的抑制效果。

3. COBRA 的复现和强化： 在 HSTU 中重构了输入层以支持 [SID, VID] 的交错输入，设计了适配交错序列的 **Relative Time Bias（相对时间偏执）**与 Attention Mask。实现了**双流损失函数** (`sidvid_loss.py`)，在联合训练中平衡稀疏 ID 的分类任务与密集向量的 InfoNCE 度量学习任务。

4. 召回和融合策略扩展： 实现了从 Direct (SID→VID)、Neighbor Substitute (邻居替代) 到 Soft Decode (软解码) 的全模式召回。特别是引入了 Beam Fusion (波束融合) 策略，通过**两阶段搜索（Top-K SID → Top-N VID）**并融合双路概率得分，有效解决了贪心解码的路径错误问题，提升了推荐的鲁棒性。


[阅读详细技术博客](https://jerryayu.github.io/posts/2026/01/cobra-hstu-deep-dive/)









