---
title: '超越语义ID：在 HSTU 中复现 COBRA 与交错式生成的探索'
date: 2026-01-23
permalink: /posts/2026/01/cobra-hstu-deep-dive/
tags:
  - 生成式推荐
  - HSTU
  - COBRA
  - 语义ID
  - 推荐系统
toc: true  # 开启右侧文章目录
toc_sticky: true # 目录随页面滚动
header:
  og_image: /images/posts/placeholder_cobra.png
excerpt: "在 HSTU 中探索 COBRA 架构，实现语义 ID (SID) 和向量 ID (VID) 的交错式生成。"
---

> **简而言之**  
 本博客记录了我基于 Shao 的 HSTU-Semantic-ID 项目进行的深度扩展。我尝试将百度的 COBRA 架构（序列交错生成）引入 HSTU，实现了 SID (语义ID) 和 VID (向量ID/视觉ID) 的混合训练。虽然最终在 MovieLens-20M 上的直接效果未达 SOTA，但我们在平衡聚类、混合解码策略以及双流损失函数上的探索，揭示了生成式推荐中“分类”与“度量学习”之间的博弈。
{: .notice--info}

> **相关代码**: [github.com/gabriel-reina/sid2vid-recommender](https://github.com/gabriel-reina/sid2vid-recommender)
{: .notice--info}

---

## 1 从语义 ID 开始：站在巨人的肩膀上

我的项目深受 Shao (shaox192) 的开源工作、Meta 的 HSTU 和百度的 COBRA 的启发。

传统的推荐系统（如原始 HSTU）使用任意 ID 作为输入，存在严重的冷启动和长尾问题。Shao 的核心洞察在于：传统的推荐系统使用任意 ID，存在严重的**冷启动**和**长尾问题**。Shao 的目标是在生成式推荐HSTU中，将 SID 集成到序列推荐器中，以改进表示和预测。通过引入 SID，我们希望模型能“理解”物品（比如知道这是一部科幻片），而不仅仅是记住一个 ID 号码。即是在生成式推荐 HSTU 中，将 SID 集成到序列推荐器中，以改进表示和预测。

在本项目中，我沿用了 Shao 的基础设置，这构成了我们工作的基石：

1.  **世界知识嵌入**：利用 `qwen3-0.6B` 对 MovieLens 中的电影进行文本嵌入，捕捉深层语义。
2.  **SID 的生成**：使用 **RQ-KMeans** 聚类将高维向量离散化为分层的 SID 元组。正如 Shao 指出的，SID 中的原始整数本身并不一定具有语义距离。
    * 我们收集池中所有项目，使用编码器生成表示。
    * 使用 K-means 聚类（例如分成 256 组），分配聚类 ID，这填充了 SID 元组的一位。
    * 计算残差（嵌入向量减去聚类中心），重复上述步骤直到达到所需层数。
    * 这通常由 3 层组成，每层 128 个代码，并根据谷歌的 TIGER 算法增加了一个额外的差异化层。

> **需要注意的是**：SID 中的原始整数本身并不一定具有语义距离。
{: .notice--warning} 

3.  **SID 元组的嵌入 (PrefixN)**：这是将元组转为整数索引的关键。业界（实际上也包括学术界）对“生成式万物”的热情已经为物品生成了丰富的语义嵌入。找到在传统系统中重用这些嵌入的方法，是一条稳步前进的实用途径。

Shao 提供的 **PrefixN 实现本质上是“路径求和”**。对于一个 SID $(c_1, c_2, c_3)$，模型实际计算的向量是：

$$\mathbf{E} = \text{Emb}(\text{Hash}(c_1)) + \text{Emb}(\text{Hash}(c_1, c_2)) + \text{Emb}(\text{Hash}(c_1, c_2, c_3))$$

在 `LocalSIDEmbeddingModule.get_item_embeddings` 方法中，代码逻辑如下：

```python
# 遍历每一层前缀，例如 range(1, 4) -> 1, 2, 3
for prefix_i in range(1, self._num_layers + 1):
    # [关键步骤1]：切片获取前缀，例如 (12, 5)
    # self.SID2embID 将这个元组转换成一个唯一的整数 ID
    emb_id = self.SID2embID(sid_tup[:, :, :prefix_i])

    # [关键步骤2]：取模，防止 ID 超过嵌入表大小
    emb_id_mod = torch.remainder(emb_id, self._emb_tbl_size)

    # [关键步骤3]：查表并 [关键步骤4]：累加
    embs = self._get_emb_from_idx(emb_id_mod, self._item_emb_SID)
    sum_embs += embs
```
这样做的好处是实现了**层级共享**：所有以 $c_1$ 开头的物品都会共享第一项嵌入，从而让长尾物品也能获得很好的基础表示。

4. **SID 的融合**：Shao 的研究指出，保留原始的任意 ID 嵌入表，并将其与 prefixN SID 嵌入融合（prefixN-indEmb）效果最好。他尝试了 sum、FC、MLP 甚至 Gate 机制。

然而，Shao 得出的结论有些残酷但真实：在行为数据极其丰富（如 ML-20M）的场景下，“协同信号”往往碾压“语义信号”。 用户的选择往往更受社交认同（“喜欢这个的人也喜欢那个”）驱动，而非纯粹的内容。此时，粗糙的 SID 共享可能会给强协同信号引入噪声。

这让我开始思考百度的 COBRA 给出的另一种答案：Why not both? 如果我们不仅生成 SID，还生成能够精确定位的 VID (Vector ID) 呢？

---

## 2 迈向完美的聚类：平衡 K-Means
### 2.1 生成 SID 
 在生成 SID 时，生成的 SID 的质量至关重要。原版的 K-Means 很容易陷入“**马太效应**”：热门簇吸纳了绝大多数电影，导致语义分辨率极低。

受 OneRec 启发，我实现了 Balanced K-Means（见代码 save_balenced.py）。通过在聚类过程中添加平衡项 (--balance-tol)，强制让每个簇的电影数量尽可能均匀。

> **这个代码的作用是离线生成 SID**。它读取预训练好的商品向量，使用一种改进的平衡 K-Means 算法进行层级聚类，生成 SID 码本和查找表，并保存为 .pkl 文件供训练使用。针对普通 K-Means 容易导致簇大小不均匀的问题，实现了 BalancedKMeans，强制每个簇的大小在一定范围内。
{: .notice--info}

我的 fit 方法包含了一套强制平衡逻辑：

```python
def fit(self, X):
        n_samples = X.shape[0]
        # 计算每个簇的平均样本数 (理想状态)
        avg_size = n_samples / self.n_clusters
        # 设定最小和最大允许的簇大小 (基于容忍度 tol)
        min_size = int(np.ceil(avg_size * (1 - self.tol)))
        max_size = int(np.floor(avg_size * (1 + self.tol)))
        min_size = max(min_size, 1)

        # 1. 先跑一次标准的 K-Means 作为初始化
        init_kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=1)
        init_kmeans.fit(X)
        self.cluster_centers_ = init_kmeans.cluster_centers_
        self.labels_ = init_kmeans.labels_

        # 2. 迭代优化平衡性
        for _ in range(self.max_iter):
            # 计算所有点到所有中心的距离
            distances = euclidean_distances(X, self.cluster_centers_)
            # 按距离分配最近的簇
            new_labels = np.argmin(distances, axis=1)
            # 统计每个簇的实际大小
            cluster_sizes = np.bincount(new_labels, minlength=self.n_clusters)
            
            # 如果所有簇的大小都在 [min_size, max_size] 范围内，说明平衡了，退出循环
            if np.all((cluster_sizes >= min_size) & (cluster_sizes <= max_size)):
                # ... (更新 labels 和 centers 并 break)
                break

            # 如果不平衡，开始搬运数据点
            for c in range(self.n_clusters):
                current_size = cluster_sizes[c]
                # 如果当前簇没有超员，跳过
                if current_size <= max_size:
                    continue

                # 找到该簇内的所有点
                c_global_indices = np.where(new_labels == c)[0]
                # 获取这些点到中心的距离
                c_distances = distances[c_global_indices, c]
                # 按距离从大到小排序 (优先搬运离中心最远的点)
                sorted_idx = np.argsort(c_distances)[::-1]
                
                # 计算需要搬运多少个点
                transfer_count = current_size - max_size
                for i in range(transfer_count):
                    # ... (边界检查)
                    transfer_global_idx = c_global_indices[sorted_idx[i]]
                    # 找到当前最小的簇
                    min_cluster = np.argmin(cluster_sizes)
                    # 强制把这个点分配给最小的簇
                    new_labels[transfer_global_idx] = min_cluster
                    # 更新计数器
                    cluster_sizes[c] -= 1
                    cluster_sizes[min_cluster] += 1

            # 根据调整后的标签重新计算簇中心
            self.cluster_centers_ = np.array([X[new_labels == c].mean(axis=0) for c in range(self.n_clusters)])
            self.labels_ = new_labels

        return self
```

1. **初始化**：先跑一次标准 K-Means。

2. **迭代优化**：计算每个簇的平均样本数（理想状态），设定基于容忍度 tol 的最小/最大簇大小。

2. **搬运数据**：如果簇大小不平衡，计算点到中心的距离。找到超员簇中离中心最远的点，强制将其分配给当前最小的簇。

这确保了每一层生成的 Code 分布是均匀的。

### 2.2 SID 质量评估

为了量化这一点，我在 `new_eval_SID.py` 中引入了四个评估维度：

1.  **重建损失**：MSE 和 Cosine Loss，衡量 SID 重组出的向量与原始向量的差异。
2.  **码本利用率**：每一层有多少个码被实际使用了（去重后的数量）。
3.  **Token 分布熵**：衡量 Token 使用的均匀程度。
    * 对于第 $l$ 层，首先计算每个 Token $k$ 的出现概率 $p_k^{(l)}$：$p_k^{(l)} = \frac{\text{Count}(k \text{ in layer } l)}{N}$
    * 原始熵：$H^{(l)} = - \sum_{k=1}^{K} p_k^{(l)} \log_2(p_k^{(l)})$(仅对 $$p_k > 0$$ 的项求和)
    * **归一化熵**：$\bar{H}^{(l)} = \frac{H^{(l)}}{\log_2 K}$， $\log_2 K$ 是理论上的最大熵（即均匀分布时的熵）。
    * 结果越接近 1.0，说明分布越均匀（平衡 K-Means 的效果好）。
    
4.  **码本冲突率**：衡量有多少商品被分配了完全相同的 SID 前缀。

> **实证效果 (ML-20M)**：使用 Balanced K-Means 后，第 3 层的归一化标记熵从 ~0.8+ 提高到 ~0.99+；前缀冲突率从 ~10% 下降到 ~6%。这是一个巨大的进步。
{: .notice--warning} 

---

## 3 COBRA 复现：序列交错与相对时间偏执

为了验证 COBRA 的想法，我基于 HSTU 架构进行了深度的改造，实现了 **序列交错**并进行**评估**。
### 3.1 交错输入 

输入序列不再是单纯的 Item ID，而是变成了 `SID1, VID1, SID2, VID2, ...` 的形式，模型需要交替预测下一个标记。

**输入处理**：输入 SID 和 VID，分别加上**类型编码**（Type Embedding，SID=0, VID=1），然后交错展平，使得序列长度翻倍 `[B, N, 2, D] -> [B, 2N, D]`，最后加上位置向量`[B, 2N, 1]`。

### 3.2 HSTU 的相对时间偏执 (`Relative_Time_Bias`)

首先，在 generate_user_embeddings 函数中，代码检测到输入序列长度翻倍（因为 SID 和 VID 交错输入）后，会对原始的时间戳进行**“克隆”**：构造出的 interleaved_ts 序列中，相邻的 (SID, VID) 对拥有相同的值。

然后是 **Attention Mask** 的处理：标准的因果掩码。HSTU 使用的是标准的**上三角因果掩码**。在交错序列 `[SID, VID]` 下的运作逻辑：由于序列在物理位置上被展平为 SID 在前，VID 在后：

1. 生成 SID 时：位置为 $i$。Causal Mask 允许它看到 $0$ 到 $i$ 的位置。它只能看到过去的信息。
2. 生成 VID 时：位置为 $i+1$。Causal Mask 允许它看到 $0$ 到 $i+1$ 的位置。这意味着 VID 可以看到同一个时间步的 SID。

```python
self.register_buffer(
    "_attn_mask",
    torch.triu(  # 上三角矩阵
        torch.ones(
            (
                self._max_sequence_length + max_output_len, 
                self._max_sequence_length + max_output_len,
            ),
            dtype=torch.bool,
        ),
        diagonal=1,
    ),
)
```

最后是**相对时间偏执**。既然 Attention Mask 是标准上三角的，模型如何知道 `SID(t1)` 和 `VID(t1)` 的关系比 `VID(t0)` 和 `SID(t1)` 的关系更紧密？

对于交错序列 `[SID_1(t1), VID_1(t1), SID_2(t2), VID_2(t2)]`，在 Attention Bias 中区分了两种情况：

1.  **同一个时间步内部 (SID_1 -> VID_1)**：时间差 = 0。模型查找“时间差为0”的 Bias 桶。这告诉模型：“注意了，这个 SID 是你当前正在生成的 VID 的孪生兄弟。”
2.  **跨时间步 (VID_1 -> SID_2)**：时间差 > 0。这告诉模型：“这是历史交互。”  

### 3.3 损失函数

这是本项目最核心的冲突点：生成式推荐中“分类”与“度量学习”的结合。我们在 `sidvid_loss.py` 中实现了联合监督 SID 和 VID。VID 分支使用批处理 `InfoNCE sampling_strategy="in-batch"`。严格的 `SID -> VID -> SID -> VID`更新频率。在 `sidvid_loss.py` 和 `train.py` 的架构下，通常是一个 Batch 同时包含 SID 和 VID 的预测任务，通过 `loss = loss_sid + lambda * loss_vid` 一次性反向传播。

  
**SID 部分**：当模型需要预测 SID 时，使用 Sampled Softmax Loss（因为 SID 空间很大，但没那么大，采样几个负例就行）。SID：沿用原始“采样 softmax”负采样损失，仅对 SID 位置。
  1. 采样 R 个负例；
  2. 计算正例与负例 logits（带温度）；
  3. 如果负例不小心采样到了正例（撞车），把它的分数设为极小值 (-5e4)，屏蔽掉；
  4. 对拼接后的向量做 log_softmax，取正例项的 -log 概率作为位置级 NLL。
 
**VID 部分**：当模型需要预测 VID 时，使用 In-batch InfoNCE Loss（对比学习，把 Batch 里其他所有商品都当负例，学习区分度极高的稠密向量）。用 is_sid_mask 排除掉 SID 的干扰，只在 VID 之间做对比学习的。VID：严格的 batch 内 InfoNCE。
   1. 对每个位置，将其预测向量与当前 batch 的所有 item 向量做 softmax，并以对应的正类 id（存在于缓存集合中）作为目标。 
   2. 注意：仅在 VID 查询位置上，将正例限定为“同 id 且缓存类型为 VID”，排除缓存中的 SID 空间向量，避免空间混杂造成训练退化。
   3. InfoNCE 强迫模型把 VID Embedding 拉近到它对应的商品向量，同时推开 Batch 里的其他所有商品。Mask 的作用至关重要，防止模型错误地把 VID 拉向一个 SID 向量。

### 3.4 双流损失函数与采样器的博弈

**稀疏 ID 分类 (SID)**：对于 SID 预测，我们沿用原始 HSTU 的做法。在每个时间步对 SID 进行 samplesoftmax 交叉熵（与 COBRA 公式匹配）。

  SID 的任务本质是分类，于是采用 **Sampled Softmax**方法，从全局采样 R 个负例，计算 Logits。如果负例不小心采样到了正例（撞车），将其分数设为极小值 (-5e4) 屏蔽掉。这不依赖 Batch Size。

**密集向量对比 (VID)**：对于 VID 预测，我们采用了 **In-Batch InfoNCE**。

  而 VID 是度量学习，我们希望预测向量与真实物品向量在空间上尽可能接近。所以**将 Batch 内的其他所有 Item 向量作为负例**。
  
  一个关键细节 (`is_sid_mask`)是：在计算 VID 的对比损失时，必须严格**屏蔽**掉 batch 内的 SID 向量。否则，模型会错误地将 VID 拉向某个 SID 的中心，导致空间混杂和训练退化。

**采样器的特殊处理**：为了适配 InfoNCE 在交错场景下的需求，我在 `vid_negatives_sampler` 中**关闭了去重 (`dedup`)**。

  在交错模式下，Item 100 会出现在序列的两个位置（作为 SID Target 和 VID Target）。关闭去重后，`process_batch` 会忠实地记录 Batch 中每一个位置的 `(ID, Embedding)` 对，并通过 `flat_vid_mask` 区分哪些位置需要计算 InfoNCE。


### 3.5 如何解码？召回与评估策略

在 `embedding_modules.py` 中，我设计了一个能够处理混合模式的“翻译官”。如果是 SID+VID 混合模式（`prefixN-indEmb`），它不仅会将 ID 翻译成 SID 码并像搭积木一样拼装向量，还会把“积木搭出来的向量”和“原本的 ID 向量”融合在一起，提供最丰富的信息给 Transformer。

1. SID→VID 直接（`eval_sidvid_mode="direct"`）：
   流程：预测前 1 个 SID，将其添加到序列中以获得vid_query，然后执行 ANN Top-K。严格再现 COBRA 的两步评估（SID 然后 VID）。
> 1. Direct：sidvid直接召回：评估路径（SID→VID→ANN）
> 2. 先基于 t-1 的用户状态预测 Top-1 SID。
> 3.  将该 SID 追加到序列末尾，生成下一步的 VID 查询向量。
> 4. 使用该 VID 做 ANN Top-K 检索，并以原序列最后一个 target_id 计算召回指标。
> 5. 注意：为了不屏蔽 ground-truth 的 target_id，最终 ANN 检索阶段**仅过滤历史 (t-1) 的已看ID，不过滤预测出的 SID**。
2. 邻居替代（`eval_sidvid_mode="neighbor"`）：
   步骤：取预测的前 1 个 SID，将其替换为其最近的邻居（不包括自身），构建vid_query，并运行 ANN。当存在码本冲突或 top-1 不稳定时，提高鲁棒性。
> 1. **Neighbor Substitute 策略**：硬最近邻替代：用于解决 SID 预测不准或 SID 聚类中心本身有偏差的问题。
> 2.  t-1 预测 Top-1 SID。
> 3.  在全库排除自身做最近邻检索得到 neighbor SID。
> 4.  用 neighbor_ids 替换掉原来的预测值 SID，填入序列最后一位。
> 5.  生成 VID，ANN Top-K 召回评估。（同上）
3.  软解码（`eval_sidvid_mode="soft"`）：
  过程：取前 K 个 SID（eval_soft_top_k），应用 softmax temperature，形成soft_embedding = Σ w_k · sid_k，导出vid_query，并运行 ANN。参数：eval_soft_top_k控制temperature候选数量和平滑度。
> 1. **Soft Decode 策略**：软解码策略。不把 SID 当作硬的离散 Token，而是当作概率分布。
> 2.  t-1 状态下取 Top-K SID 及分数。
> 3.  softmax(scores/τ) 加权聚合 Top-K 的 item embedding 得到 soft_embedding。
> 4.  用该 soft_embedding 作为最后一步的 embedding SID：ID 还 是填 Top-1 的 ID (为了 mask 正常工作)，但是 Embedding 被强行替换成了 soft_emb。
> 5.  模型拿着这个“混合语义向量”去生成 VID 查询，ANN 召回评估。
> 6.  默认 soft_top_k=5, τ=0.5，可按需调整。
4.  beam召回和融合（`eval_sidvid_mode="beam"`）：
  两个阶段：（1）宽度为 SID 的梁beam_top_k；（2）对于每个梁，构建一个vid_query并取出ann_top_n物品。
> 1. **Beam Fusion 策略**：波束搜索与融合：这是提升指标的关键，也是计算量最大的。
> 2.  用历史状态 t-1 预测 SID，基于 SID 候选索引检索 Top-K 真实 SID。
> 3.  对每个真实 SID（逐个替换，不并行混合），生成下一步 VID 的查询向量并做 ANN 召回 Top-N 真实 VID。
> 4.  将 K（SID）×N（VID） 的候选以融合打分排序（beam_sid_score 与 ann_vid_score 直接相乘），与 true VID target 比对计算指标。
> > 1.  **prod（概率乘积）**：score(k,i) = (p_sid(k))^α × (p_vid(k,i))^(1-α)强调α = eval_beam_alpha在这两方面都很优秀的候选人。
> > 2.  **sum（加权和）**：score(k,i) = α · p_sid(k) + (1-α) · p_vid(k,i)跨尺度的线性权衡。
> 5.  beamfusion分数：softmax(alpha * beamscore) * softmax(beta * annscore), alpha和beta是温度系数，需要按需调整

  5.  Beam Fusion 的意义

**容错性**：如果只用 Top-1 SID（贪心搜索），一旦第一步猜错了类别（比如用户其实想看动作片，但模型预测科幻片略高一点点），后面就全错了，永远召回不了动作片。Beam Fusion 允许模型“三心二意”，保留多种可能性，最后靠具体的商品质量来反向修正路径。

**解决 SID 粒度过粗的问题**： SID 是聚类得到的，有时两个截然不同的电影可能被分到同一个 SID，或者同一个系列的电影被分到不同的 SID。Beam Fusion 通过同时探索多个邻近的 SID，打破了硬聚类的边界。

---

## 4. 结果与反思：分类 vs 度量学习

最终结果是令人深思的。

  **HSTU Direct (Sampled Softmax)** 在 MovieLens-20M 上基本达到了 SOTA 水准。
  **COBRA (交替 SID/VID)** 仅仅达到了 SOTA 的 1/3。

**为什么？** 除了代码实现的复杂性，我认为根本原因在于 **显存限制与 Loss 本质的冲突**。

1.  **HSTU Direct (分类任务)**：
    Sampled Softmax 并不依赖 Batch Size。它计算的是 Query 与 Item Embedding 的点积，是一个 20000 类的分类问题。哪怕 Batch=1，它也可以从库里采 2000 个负样本，训练十分高效。

2.  **SID2VID 的 VID (度量学习)**：
    InfoNCE Loss 需要大量的负样本来“撑开”向量空间。在双塔/生成架构中，为了计算效率，负样本来源于 **同一个 Batch**。
    由于我们将序列长度翻倍（交错输入），加上复杂的 Attention Mask 和梯度图，显存占用激增，导致我们不得不缩小 Batch Size。**小 Batch 限制了 InfoNCE 的负样本数量，直接扼杀了度量学习的效果。**

---

### 5 最后的想法

虽然在 ML-20M 上没有击败 Baseline，但这次探索并非徒劳。我们验证了平衡 K-Means 对 SID 质量的提升，实现了复杂的交错注意力机制，并深入理解了混合 Loss 的训练难点。

对于拥有海量算力（大 Batch）和富媒体内容（强内容信号）的场景，COBRA 架构依然极具潜力。而对于传统的协同过滤主导的数据集，Shao 的结论依然成立：有时候，简单的 ID 就是最好的。

感谢阅读！如果您对代码细节感兴趣，欢迎查看仓库中的 `sidvid_loss.py` 和 `embedding_modules.py`。
