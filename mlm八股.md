### BLIP

[LLM大模型: blip2/blip3多模态大模型原理 - 第七子007 - 博客园](https://www.cnblogs.com/theseventhson/p/18488142)

![image-20251221143325769](./image/image-20251221143325769.png)

三个损失：

**ITC:图文对比学习**

**ITM:图文匹配**

**LM:基于图片的文本生成**

在 BLIP（Bootstrapping Language-Image Pre-training）架构中，**ITC (Image-Text Contrastive)** 和 **ITM (Image-Text Matching)** 是两个核心的视觉语言预训练任务。虽然它们都旨在对齐视觉和文本信息，但在实现机制、计算粒度以及模型结构上的分工有显著区别。

------

### 1. ITC (Image-Text Contrastive) 图像-文本对比损失

ITC 的目标是**在一个共享的特征空间中，通过对比学习让互为正样本的图像和文本特征尽可能接近，而让负样本尽可能远离。**

- **结构：** 采用**双编码器（Dual-Encoder）**架构。图像经过视觉编码器（ViT），文本经过文本编码器，分别得到全局表征（如 `[CLS]` token 的向量）。
- **计算：** 计算图像特征 $v$ 和文本特征 $w$ 之间的余弦相似度。损失函数通常使用 **InfoNCE Loss**，通过对比一个 batch 内的所有样本对来实现。
- **作用：** * 学习**全局对齐**。
  - 提高检索效率（可以预先计算特征，通过点积快速检索）。
  - 在 BLIP 中，ITC 还会利用**动量队列（Momentum Queue）来增加负样本的数量**。

------

### 2. ITM (Image-Text Matching) 图像-文本匹配损失

ITM 的目标是判断给定的一对图像和文本是否真正匹配，这是一个**二分类任务**。

- **结构：** 采用**融合编码器（Fusion-Encoder）**架构。文本特征会通过**交叉注意力层（Cross-Attention）与图像特征进行深层交互**。
- **计算：** 将融合后的多模态特征输入到一个线性分类器（Head），输出两个类别的概率：`Match` 或 `Not Match`。损失函数通常是**交叉熵损失（Binary Cross-Entropy）**。
- **负采样策略：** 为了让模型学到更细致的区分能力，BLIP 在计算 ITM 时会采用 **Hard Negative Mining**。**它会利用 ITC 计算出的相似度分数，挑选出那些“长得很像正样本但实际不是”的负样本（即相似度高但标签为负的对）来进行训练。**
- **作用：** * 学习**局部/细粒度对齐**。
  - 捕捉图像和文本之间复杂的语义关联。

------

### 3. ITC 与 ITM 的核心区别总结

| **特性**     | **ITC (对比损失)**               | **ITM (匹配损失)**                             |
| ------------ | -------------------------------- | ---------------------------------------------- |
| **任务类型** | 对比学习（Ranking/Contrastive）  | 二分类（Binary Classification）                |
| **模型架构** | 双编码器（特征不直接交互）       | 融合编码器（通过 Cross-Attention 深度交互）    |
| **计算开销** | 较低，特征可缓存，适合大规模检索 | 较高，每一对都需要经过 Transformer 层          |
| **对齐粒度** | 粗粒度、全局特征对齐             | 细粒度、多模态语义融合                         |
| **输入处理** | 独立编码图像和文本               | 图像和文本特征在中间层进行 Cross-Modality 交互 |
| **典型用途** | 快速图文检索（Retrieval）        | 重新排序（Re-ranking）、验证匹配度             |

------

### 为什么 BLIP 需要两者同时存在？

这两个损失函数是**互补**的：

1. **效率与精度的平衡：** **ITC 能够快速筛选出候选集，而 ITM 则对候选集进行精确的“重排”和验证**。
2. **相互促进：** ITC 为 ITM 提供了高质量的“难负样本（Hard Negatives）”，而 ITM 深入的交互能力帮助模型理解更复杂的逻辑（比如空间关系、数量等），这反过来也有助于编码器学到更好的单模态特征。

在 BLIP 的预训练过程中，除了这两个损失，通常还会配合 **LM (Language Modeling)** 损失（即解码器部分），利用 Captioning 任务进一步增强模型生成文本描述的能力。

你想了解 BLIP 之后如何利用这些损失进行“人工数据清洗（CapFilt）”的过程吗？

![image-20251221143345800](./image/image-20251221143345800.png)

数据集自举训练

1、先用网络数据和人类标注的数据去训练，得到一个预训练模型

2、然后使用人类标注的数据集进行微调，得到filter和captioner

3、然后将网络收集的数据给到3中的filter模型进行过滤

4、然后将网络收集的数据给到3中的captioner模型生成文本，再将图文给filter进行过滤

5、最终，人工标注的，以及3和4中基于网络数据进行清洗过的数据，又得到了一个新的预训练数据，以此往复

### BLIP2

[庖丁解牛BLIP2-CSDN博客](https://blog.csdn.net/m0_53157173/article/details/148367477?ops_request_misc=elastic_search_misc&request_id=1608095b13cb4bb495d4587553fa176f&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-4-148367477-null-null.142^v102^pc_search_result_base4&utm_term=blip2源码&spm=1018.2226.3001.4187)

第一阶段预训练(**Q-former学习视觉语言表征之间的关系**)，和BLIP的训练方法差不多

![image-20251221155932128](./image/image-20251221155932128.png)

```
import torch
import torch.nn.functional as F

# ====================== 1. 初始化所有核心组件（联合训练共用） ======================
# 1.1 图像编码器：CLIP-ViT-L/14，冻结权重
image_encoder = ViTEncoder(pretrained="clip-vit-large-14")
for param in image_encoder.parameters():
    param.requires_grad = False

# 1.2 Q-Former：核心桥梁（Self-Attention + Cross-Attention + FFN），可训练
q_former = QFormer(
    num_queries=32,        # Learned Queries数量（固定长度）
    hidden_size=768,       # 隐藏层维度（和ViT/文本一致）
    num_layers=12,         # Q-Former的block数量
    vocab_size=10000       # 文本词汇表大小（适配ITG生成）
)

# 1.3 任务专属Head（均为可训练）
itm_head = torch.nn.Linear(768, 2)          # ITM：二分类（匹配/不匹配）
itg_lm_head = torch.nn.Linear(768, 10000)   # ITG：文本生成（预测下一个token）

# 1.4 优化器：仅更新Q-Former + 各任务Head
optimizer = torch.optim.Adam([
    {"params": q_former.parameters()},
    {"params": itm_head.parameters()},
    {"params": itg_lm_head.parameters()}
], lr=1e-4)

# ====================== 2. 输入数据（单批次覆盖三个任务的需求） ======================
batch_size = 8
# 2.1 基础输入
images = torch.randn(batch_size, 3, 224, 224)          # 图像：[8,3,224,224]
text_tokens = torch.randint(0, 10000, (batch_size, 16))# 文本token：[8,16]（16为序列长度）
text_embeddings = TokenEmbedding(10000, 768)(text_tokens)  # 文本嵌入：[8,16,768]

# 2.2 任务专属标签
itm_labels = torch.tensor([1,1,0,1,0,0,1,0])           # ITM标签：1=匹配，0=不匹配 [8]
itg_labels = text_tokens[:, 1:].contiguous()           # ITG生成目标：去掉第一个token [8,15]
itg_attention_mask = torch.ones(batch_size, 15)        # ITG掩码：[8,15]（1表示有效token）

# ====================== 3. 图像侧：ViT提取特征（所有任务共用K/V） ======================
image_features = image_encoder(images)  # 图像特征序列：[8,256,768]（256=ViT patch数）

# ====================== 4. 共享组件：提取Learned Queries ======================
learned_queries = q_former.get_learned_queries(batch_size=batch_size)  # [8,32,768]

# ====================== 5. 任务1：ITC（图文对比学习） ======================
# 5.1 图像嵌入I：仅Queries + 图像特征（无文本）
image_queries_itc = q_former.cross_attention(
    query=learned_queries, key=image_features, value=image_features
)  # [8,32,768]
image_emb = F.normalize(torch.mean(image_queries_itc, dim=1), p=2, dim=-1)  # [8,768]

# 5.2 文本嵌入T：仅文本 + Self-Attention（无图像）
text_self_attn_mask_bi = torch.ones(batch_size, 16, 16)  # 双向掩码
text_self_attn_itc = q_former.self_attention(text_embeddings, attention_mask=text_self_attn_mask_bi)  # [8,16,768]
text_emb = F.normalize(torch.mean(text_self_attn_itc, dim=1), p=2, dim=-1)  # [8,768]

# 5.3 ITC损失（InfoNCE）
sim_matrix = torch.matmul(image_emb, text_emb.t()) / 0.07  # 相似度矩阵：[8,8]，温度系数0.07
labels_itc = torch.arange(batch_size).to(sim_matrix.device)
itc_loss_i2t = F.cross_entropy(sim_matrix, labels_itc)          # 图像→文本
itc_loss_t2i = F.cross_entropy(sim_matrix.t(), labels_itc)      # 文本→图像
itc_loss = (itc_loss_i2t + itc_loss_t2i) / 2                    # 总ITC损失

# ====================== 6. 任务2：ITM（图文匹配） ======================
# 6.1 拼接Queries+文本，走完整Q-Former流程
query_text_seq_itm = torch.cat([learned_queries, text_embeddings], dim=1)  # [8,48,768]
self_attn_mask_itm = torch.ones(batch_size, 48, 48)                        # 双向掩码：[8,48,48]

# 6.2 Self-Attention + Cross-Attention + FFN
self_attn_itm = q_former.self_attention(query_text_seq_itm, attention_mask=self_attn_mask_itm)  # [8,48,768]
cross_attn_itm = q_former.cross_attention(self_attn_itm, key=image_features, value=image_features)  # [8,48,768]
fusion_repr_itm = q_former.feed_forward(cross_attn_itm)                                            # [8,48,768]

# 6.3 ITM损失
pooled_repr_itm = fusion_repr_itm[:, 0, :]  # 取<cls>位置：[8,768]
itm_logits = itm_head(pooled_repr_itm)      # 预测匹配概率：[8,2]
itm_loss = F.cross_entropy(itm_logits, itm_labels)  # 交叉熵损失

# ====================== 7. 任务3：ITG（图像引导文本生成） ======================
# 7.1 拼接Queries+文本前缀（仅前15个token，适配生成目标）
text_embeddings_itg = text_embeddings[:, :-1, :]  # 文本前缀：[8,15,768]（去掉最后一个token）
query_text_seq_itg = torch.cat([learned_queries, text_embeddings_itg], dim=1)  # [8,47,768]

# 7.2 因果掩码（生成任务核心：只能看前面的token）
self_attn_mask_itg = torch.tril(torch.ones(batch_size, 47, 47))  # 下三角掩码：[8,47,47]

# 7.3 Self-Attention + Cross-Attention + FFN
self_attn_itg = q_former.self_attention(query_text_seq_itg, attention_mask=self_attn_mask_itg)  # [8,47,768]
cross_attn_itg = q_former.cross_attention(self_attn_itg, key=image_features, value=image_features)  # [8,47,768]
fusion_repr_itg = q_former.feed_forward(cross_attn_itg)                                            # [8,47,768]

# 7.4 提取文本部分的表示（去掉Queries，仅保留文本前缀的输出）
text_repr_itg = fusion_repr_itg[:, 32:, :]  # 去掉前32个Queries：[8,15,768]

# 7.5 ITG损失（语言模型交叉熵）
itg_logits = itg_lm_head(text_repr_itg)  # 预测token：[8,15,10000]
itg_loss = F.cross_entropy(
    itg_logits.reshape(-1, 10000),  # [120,10000]（8*15=120）
    itg_labels.reshape(-1),         # [120]
    ignore_index=0                  # 忽略padding token（假设0是PAD）
)

# ====================== 8. 联合训练：总损失 + 反向传播 ======================
# 8.1 损失加权（超参可调整，通常设为1:1:1）
total_loss = itc_loss + itm_loss + itg_loss

# 8.2 反向传播 + 优化
optimizer.zero_grad()  # 清空梯度
total_loss.backward()  # 反向传播
optimizer.step()       # 更新参数

# 8.3 打印训练日志
print(f"Total Loss: {total_loss.item():.4f}")
print(f"  - ITC Loss: {itc_loss.item():.4f}")
print(f"  - ITM Loss: {itm_loss.item():.4f}")
print(f"  - ITG Loss: {itg_loss.item():.4f}")
```



第二阶段预训练(**视觉到语言生成学习，并训练Q-Forform使其输出的视觉表示能够被LLM解读**)

![image-20251221160125873](./image/image-20251221160125873.png)

![image-20260307222530913](./image/image-20260307222530913.png)

# CogVLM：NeurIPS 2024 视觉专家模型解读

![image-20251221180916216](./image/image-20251221180916216.png)

这个图对应的是**智谱AI发表在NeurIPS 2024的论文《CogVLM: Visual Expert for Pretrained Language Models》**（arXiv链接：https://arxiv.org/abs/2311.03079），这是CogVLM模型的核心架构图。结合论文内容，图中模块的论文级解读如下：

### 一、论文核心背景（图的设计动机）

论文针对**“浅层对齐多模态模型（如BLIP-2）视觉能力弱、深度融合模型（如LLaVA）易遗忘NLP能力”**的痛点，提出了**“视觉专家（Visual Expert）”**机制：**在冻结预训练语言模型（LLM）主体参数的前提下，在LLM每一层插入可训练的视觉专家模块**，实现“图文特征深度融合+保留LLM原文本能力”——你提供的图正是这一核心机制的可视化。

### 二、图(a)：对应论文的“输入处理模块”（论文2.1节）

图(a)是CogVLM的**输入特征对齐流程**，对应论文中“ViT编码器+MLP适配器”的设计：

1. **Patchified images → ViT encoder**：

1. 论文中使用**EVA2-CLIP-E作为ViT编码器**，将图像分割为patch后编码为视觉特征序列；

1. **ViT encoder → MLP Adapter**：

1. 论文中用**两层SwiGLU结构的MLP适配器**，将ViT输出的视觉特征映射到与LLM词嵌入相同的维度空间；

1. **Paired Text → Word embedding**：

1. 文本经词嵌入层转化为文本特征序列；

1. **Concat + Position ids for RoPE**：

1. 论文中规定：**所有图像特征共享相同的位置ID（全设为0），文本特征的位置ID从1开始递增**（图中`[0 0 0 ... 1 2 3 ...]`），通过旋转位置编码（RoPE）区分图文特征，避免位置混淆。

### 三、图(b)：对应论文的“视觉专家模块”（论文2.2节）

图(b)是CogVLM的核心创新——**视觉专家（Visual Expert）**，对应论文中“LLM每一层的视觉专家结构”：

1. **特征拆分与预处理**：

1. 输入特征被拆分为`Image features`和`Text features`，文本特征先经`LayerNorm`（与论文中“保持LLM层归一化逻辑一致”对应）；

1. **QKV matrix的双分支设计**：

1. 论文中明确：视觉专家为图像特征单独设计了**可训练的QKV矩阵**（图中紫色`QKV matrix`），而文本特征复用LLM原有的QKV矩阵（图中白色`QKV matrix`）——这是“视觉专家”的核心：仅训练图像对应的QKV参数，冻结LLM原有参数；

   **在计算完qkv之后，图像和文本的qkv是进行拼接还是直接相加呢？**

   **把「图像的 Q 序列」和「文本的 Q 序列」在序列长度维度拼接，得到一个 “混合模态的 Q 序列”**

   **同理，K 序列、V 序列也做同样的拼接，得到 “混合模态的 K 序列”“混合模态的 V 序列”**

1. **Multi-head Attention**：

1. 图像与文本的Q/K/V在多头注意力层交互，实现“图文特征深度融合”（论文中称为“跨模态注意力交互”）；

1. **FFN的双分支设计**：

1. 图像特征对应**可训练的FFN**（图中紫色`FFN`），文本特征复用LLM原FFN（图中白色`FFN`）——与QKV设计一致，仅训练图像对应的FFN参数。

### 四、图与论文的对应关系总结

这张图是论文中的**Figure 3**（CogVLM架构图），直接对应论文的核心架构：

- 图(a)是“输入特征对齐流程”，解决“图文特征维度/位置不兼容”问题；
- 图(b)是“视觉专家模块”，解决“图文深度融合+避免LLM遗忘”问题。

这篇论文的核心贡献正是通过图中所示的结构，在冻结LLM参数的前提下，实现了图文特征的深度融合，让CogVLM在15+跨模态基准上达到SOTA，同时保留了LLM原有的文本能力。

## CogVLM2

![image-20251221212913579](./image/image-20251221212913579.png)

# CogVLM2：2024 年 8 月 29 日发布的视觉语言模型

《CogVLM2: Visual Language Models for Image and Video Understanding》是**智谱AI与清华大学知识工程实验室于2024年8月29日发表在arXiv（编号2408.16500）的论文**，是CogVLM系列的第二代多模态模型，核心目标是解决“前代模型视觉分辨率低、文本处理能力有限、视频理解缺失”的痛点，在保持轻量化的同时实现“媲美闭源模型的多模态能力”。

### 一、论文核心动机

前代CogVLM存在3个局限：

1. 图像输入分辨率仅224×224，无法捕捉文档、工业影像的细节；
2. 文本处理长度有限，不支持长文档交互；
3. 仅支持静态图像，缺乏视频时序理解能力。

论文的核心思路是：**继承“视觉专家（Visual Expert）”架构，通过“高分辨率视觉编码+动态模态融合+多阶段训练”，在冻结LLM主体参数的前提下，同时提升图像/视频理解能力、支持长文本交互**。

### 二、模型架构（论文核心创新）

CogVLM2采用**“50亿参数视觉编码器 + 70亿参数视觉专家模块 + 8B参数Llama3-8B-Instruct语言基座”**的异构架构（总参数量19B，推理时仅激活约120亿参数，兼顾性能与效率），核心设计包括：

1. **高分辨率视觉编码**：

   用EVA2-CLIP-E作为视觉编码器，支持1344×1344像素输入；**在编码器后加入“2×2卷积降采样模块”，压缩高分辨率特征序列长度，平衡性能与推理速度**。

1. **优化的视觉专家模块**：

   继承CogVLM的“视觉专家注入LLM层”设计，但新增**动态模态融合机制**——**根据任务类型（如VQA/图像描述）调整视觉-语言注意力权重，实现“任务适配的跨模态交互**”。

1. **图文位置编码区分**：

   图像特征的位置ID全设为0，文本特征位置ID从1开始递增（通过RoPE编码区分模态），避免图文特征混淆。

### 三、预训练与微调策略（论文关键训练方法）

1. **预训练阶段**：
   1. 数据层面：采用“迭代精炼+合成数据”——先用初始模型标注数据并人工校正，再合成中文OCR、GUI图像等稀缺数据，解决视觉-语言数据噪声大、分布有限的问题；
   2. 训练策略：分三阶段逐步启用参数（先训交叉注意力、再训视觉专家、最后训视觉编码器），同时混合“语言预训练数据+视觉-语言数据”，避免LLM文本能力退化；并**逐步提升图像分辨率**（从224×224到1344×1344），让模型渐进适应高分辨率输入。
2. **微调阶段**：
   1. 图像SFT：分两阶段——先在30万对齐语料+VQA数据集上增强基础能力，再用5万偏好数据优化输出风格；
   2. 视频SFT：从图像模型迁移，输入24帧视频，新增“时序定位调优”，通过自动化生成视频问答数据（用GPT-4o过滤），实现视频时序理解。

### 四、核心改进与性能表现

1. **核心能力升级**：
   1. 支持1344×1344图像分辨率、8K文本长度；
   2. 新增CogVLM2-Video分支，支持视频时序理解；
   3. 提供中英双语版本，针对中文OCR、古汉字等场景专项优化。
2. **基准测试结果**：

3. 在MMBench、MM-Vet、TextVQA、DocVQA等基准上取得SOTA：
   - TextVQA（中文）：85.0分（超越GPT-4V的78.0分）；
   - DocVQA：92.3分（开源模型第一）；
   - OCRbench（中文）：780分（较前代提升32%）。

### 五、开源与应用

论文对应的模型已开源（GitHub：THUDM/CogVLM2），基于Llama3-8B-Instruct提供英文和中英双语版本，且支持Int4量化（16GB显存即可推理），可落地于工业质检（PCB焊点缺陷识别）、智能文档处理（合同审核）、医疗影像解析等场景。

这篇论文的核心贡献是：**以19B轻量化参数量，实现了“高分辨率视觉理解+长文本交互+视频时序能力”的统一，为开源多模态模型提供了“性能媲美闭源、部署成本低”的新范式**。

### Qwen-VL

![image-20251221215518576](./image/image-20251221215518576.png)

### QwenVL系列

[多模态技术梳理：Qwen-VL系列 - 知乎](https://zhuanlan.zhihu.com/p/25267823390)

#### qwen-vl

![image-20260326164419182](./image/image-20260326164419182.png)

![727ed6ba-a268-46c4-bdc2-7b34d0336c5c](./image/727ed6ba-a268-46c4-bdc2-7b34d0336c5c.png)

### qwen2-vl

![image-20251226195615434](./image/image-20251226195615434.png)

#### **还有就是将模态映射层从Cross-Attention改为了MLP**

三阶段训练方法。

第一阶段，我们专注于**训练视觉转换器（ViT）**组件，利用大量图像-文本对来增强大型语言模型（LLM）中的语义理解。

第二阶段，我们**解冻所有参数**，使用更广泛的数据进行训练，实现更全面的学习。

在最后阶段，我们**锁定ViT参数**，并利用教学数据集对LLM进行独占**微调**。



## qwen2.5-vl

1. **引入窗口注意力（Window Attention）机制**，优化视觉编码器的推理效率；
2. **提出动态帧率采样（Dynamic FPS Sampling）**，将动态分辨率概念扩展到时间维度，显著提升长视频理解能力；
3. **在时间域引入与绝对时间对齐的多模态位置编码（[MRoPE](https://zhida.zhihu.com/search?content_id=260266360&content_type=Article&match_order=1&q=MRoPE&zhida_source=entity)）**，改善模型对事件节奏的把握；     
4. **将预训练语料从 1.2 万亿扩展到 4.1 万亿 tokens**，结合高质量监督数据，增强多任务泛化能力。

![image-20260107001700899](./image/image-20260107001700899.png)

[(6 条消息) 【Qwen】Qwen2.5-VL技术报告 - 知乎](https://zhuanlan.zhihu.com/p/1927463592279671080)

![image-20251226204401149](./image/image-20251226204401149.png)

![image-20251226203923542](./image/image-20251226203923542.png)



![image-20251226200649230](./image/image-20251226200649230.png)

## 预训练：

[(1 封私信 / 14 条消息) 【多模态大模型】Qwen2.5-VL解剖 - 知乎](https://zhuanlan.zhihu.com/p/24986805514)

![image-20260330212722823](./image/image-20260330212722823.png)

1. **视觉预训练:** **仅训练 ViT**，使用**图像标题、视觉知识和 OCR 数据**。视觉转换器（ViT）被训练以提升其与语言模型的对齐度，为多模态理解奠定坚实基础

2. **多模态预训练:** **解冻所有模型参数，使用交错数据、VQA、视频、智能体等多种数据**。以增强其处理复杂视觉信息的能力

3. **长上下文预训练:** **引入视频、智能体数据，并增加序列长度**。进一步提升模型在较长序列中的推理能力

   

## 后训练：

**1> 监督微调 (SFT)**

SFT阶段用到的instruction data包含约 200 万条数据，50% 为纯文本数据，50% 为多模态数据（图文和视频文本）。在数据过滤流程中，先使用 Qwen2-VL-Instag （一个基于Qwen2-VL的分类模型）将 QA 对分层分类为 **8 个主要领域和 30 个细粒度子类别**，然后对于这些细分类别，使用**领域定制过滤，**结合基于规则和基于模型的过滤方法。

- **基于规则的过滤:** 删除重复模式、不完整或格式错误的条目，以及不相关或可能导致有害输出的查询和答案。
- **基于模型的过滤:** 使用 Qwen2.5-VL 系列训练的奖励模型评估多模态 QA 对。

此外，在训练中还使用**拒绝采样 (Rejection Sampling)**技术，增强模型的推理能力。使用一个中间版本的 Qwen2.5-VL 模型，对带有标注（ground truth）的数据集生成响应，将模型生成的响应与标注的正确答案进行比较，只保留模型输出与正确答案匹配的样本，丢弃不匹配的样本。此外还进一步过滤掉不理想的输出，例如：代码切换 (code-switching)、过长 (excessive length)、重复模式 (repetitive patterns)等。通过这种方式，确保数据集中只包含高质量、准确的示例。

这里会不会因此丢弃掉一些好的困难样本？报告中并没有提及，似乎对于SFT阶段，正确性的要求压倒难度，并不指望通过这一阶段获得更强的能力。

**2> 直接偏好优化 (DPO)**

报告中基本一笔带过。仅使用图文和纯文本数据，不使用视频数据，利用偏好数据将模型与人类偏好对齐。

没有使用GRPO和基于规则的强化学习。对于数学、代码以外的任务，似乎没有特别好的规则定义方法，还是要回到基于奖励模型或者偏好数据的方法。



## Qwen3-vl

- **改进的交错式 MRoPE，显著提升对图像与视频中时空信息的建模能力；**

- **DeepStack 引入，有效利用多层级 [ViT](https://zhida.zhihu.com/search?content_id=266934707&content_type=Article&match_order=1&q=ViT&zhida_source=entity) 特征，强化视觉-语言对齐；**

- **基于文本的时间对齐机制，从 T-RoPE 演进为显式的文本时间戳对齐，实现更精确的时序定位。**

  

![image-20251226214005304](./image/image-20251226214005304.png)

**维度越低，频率变化越快**

**嵌入维度被划分为时间（t）、水平（h）和垂直（w）子空间，每个子空间分配不同的旋转频率。这导致频谱不平衡，后续研究显示这会降低长视频理解基准的性能**。为解决这个问题，我们重新设计了频率分配，通过在嵌入维度中交错t、h和w分量（Huang等，2025）。这确保每个时空轴在低频和高频频带上均有统一的表示

![image-20251227000042665](./image/image-20251227000042665.png)



![image-20251227000026007](./image/image-20251227000026007.png)



![image-20251226235543239](./image/image-20251226235543239.png)



![image-20251226235727315](./image/image-20251226235727315.png)

![image-20251226235937260](./image/image-20251226235937260.png)

这个感觉有问题，如果文本段的t都一样，感觉文本会失去位置关系

![0133c01b-c3b7-402e-9457-fd17f8ced52c](./image/0133c01b-c3b7-402e-9457-fd17f8ced52c.png)

## 训练流程

### 预训练

[(1 封私信 / 12 条消息) Qwen3-VL技术报告：模型结构、训练方法浅尝 - 知乎](https://zhuanlan.zhihu.com/p/1977471328991876909)

![image-20260330211103954](./image/image-20260330211103954.png)

1、**视觉语言对齐**：**仅训练MLP**，以高质量图像-标题对、视觉知识集合和 [OCR](https://zhida.zhihu.com/search?content_id=266934707&content_type=Article&match_order=1&q=OCR&zhida_source=entity) 数据为主

2、**多模态预训练**：**全参、解冻全部模型**，混合视觉-语言（VL）数据和纯文本数据。VL 数据：包含交错图文文档、视觉接地任务、VQA、STEM 数据和少量视频数据（引入时间维度理解）。纯文本数据：保障语言能力不退化。

3、**长上下文预训练**：**全参、解冻全部模型**，扩展模型处理长序列的能力，适配长文档、长视频等场景。

- 增加纯文本数据占比，强化长文本理解。
- VL 数据中扩充视频数据和智能体指令跟随数据，支撑长流程任务

4、**超长上下文预训练**：**将上下文窗口推至极限**，适配 256K tokens 级别的超长输入。使用**全参数**训练，聚焦长视频和长文档理解任务

### 后训练

1、**sft:赋予模型指令跟随能力，激活潜在推理技能**

- 训练策略：分两阶段推进

1. 初始阶段：**以 32K 上下文长度训练，覆盖基础指令场景**。

2. 扩展阶段：**将上下文长度提升至 256K，重点训练长文档、长视频相关任务。**

   

   模型变体拆分：

   - 非思考模型（Non-thinking）：使用标准格式数据，适配快速响应场景。
   - 思考模型（Thinking）：采用链式思维（CoT）格式数据，显式建模推理过程，强化复杂任务处理能力。

2、**强到弱蒸馏**

- 两阶段蒸馏：**仅使用纯文本数据蒸馏，专门优化 LLM backbone**

1. 离线蒸馏（Off-policy Distillation）：融合教师模型的输出结果进行响应蒸馏，为学生模型奠定推理基础。
2. 在线蒸馏（On-policy Distillation）：让学生模型基于提示生成响应，通过最小化 KL 散度对齐师生模型的预测分布。

3、**强化学习**(**SAPO**)

1. 推理强化学习（Reasoning RL）：针对数学、编码、视觉接地等可确定性验证的任务，通过规则或代码执行器判断结果正确性。
2. 通用强化学习（General RL）：聚焦指令跟随精度和偏好对齐，优化模型在开放场景下的响应质量。

## llava

![image-20260112225830696](./image/image-20260112225830696.png)

一阶段预训练

主要是为了特征对齐，使用单轮对话数据预测图片的caption,只训练这个线性层W

二阶段 端到端微调

训练W和LLM



# LLaVA：两阶段训练及优化详解

LLaVA（Large Language and Vision Assistant）采用**两阶段训练范式**，核心是先对齐视觉-语言特征空间，再通过指令微调让模型学会遵循视觉指令并生成自然语言响应。以下是各阶段的详细流程与关键细节。

### 一、整体架构概述

LLaVA的基础架构由三部分组成：

| 组件           | 作用                            | 初始状态                       |
| -------------- | ------------------------------- | ------------------------------ |
| **视觉编码器** | 提取图像特征（如CLIP ViT-L/14） | 冻结（所有阶段）               |
| **MLP投影层**  | 将视觉特征映射到LLM词嵌入空间   | 第一阶段训练，第二阶段继续训练 |
| **语言模型**   | 生成文本响应（如Vicuna-7B/13B） | 第一阶段冻结，第二阶段训练     |

### 二、第一阶段：视觉-语言特征对齐预训练（Stage 1: Feature Alignment）

#### 1. 核心目标

- 仅训练**MLP投影层**，冻结视觉编码器与语言模型权重
- 将视觉特征空间与语言模型的词嵌入空间对齐，使LLM能理解图像内容
- 建立图像到文本的基础映射，类似训练视觉“分词器”

#### 2. 数据集

- 原始：**LAION-CC-SBU 558K子集**（LLaVA-Pretrain）或Filtered CC3M（595K图文对，CC-595K）
- 处理方式：将图文对转换为“朴素问答”格式（如：“描述这张图片”→图像→原始标题）
- 样本格式：`[图像] → "描述这张图片" → [图片标题]`

#### 3. 输出

- 训练好的**MLP投影层权重**，可与冻结的视觉编码器、语言模型组合成基础多模态模型

### 三、第二阶段：视觉指令微调（Stage 2: Visual Instruction Tuning）

#### 1. 核心目标

- 联合训练**MLP投影层+语言模型**权重，保持视觉编码器冻结
- 让模型学会遵循多模态指令，生成符合人类偏好的自然语言响应
- 支持复杂视觉任务：VQA、图像描述、多轮对话、视觉推理等

#### 2. 数据集

- 核心：**GPT-4生成的158K高质量多模态指令数据**（Instruct-158K），含多样化任务类型
- 扩展（LLaVA-1.5+）：学术任务数据（515K VQA、图像描述等），总计约650K+样本
- 任务覆盖：单轮/多轮问答、图像描述、视觉推理、细粒度定位等
- 样本格式：`[图像] → "详细描述这张图片中的动物及其行为" → [GPT-4生成的详细回答]`

#### 3. 输出

- 完整的**LLaVA模型权重**（视觉编码器+训练的MLP+训练的语言模型）
- 可直接用于多模态对话与视觉任务推理

### 五、训练流程总结表

| 阶段           | 训练组件    | 数据集          | 核心目标           | 训练时长 | 内存优化           |
| -------------- | ----------- | --------------- | ------------------ | -------- | ------------------ |
| **特征对齐**   | 仅MLP投影层 | 558K/595K图文对 | 视觉-语言空间对齐  | 1–4小时  | BF16，梯度检查点   |
| **指令微调**   | MLP+LLM     | 650K+指令数据   | 指令跟随与响应生成 | 20小时   | LoRA/FSDP/ZeRO     |
| **进阶预训练** | 全参数      | 4M+高质量数据   | 知识注入           | 额外天数 | 混合精度，模型并行 |

# Flamingo

[Flamingo：打破模态壁垒的少样本视觉语言模型_flamingo模型哪一年提出的-CSDN博客](https://blog.csdn.net/m0_65010824/article/details/155012939?ops_request_misc=&request_id=&biz_id=102&utm_term=flamingo&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-155012939.142^v102^pc_search_result_base4&spm=1018.2226.3001.4187)

## Vit



![image-20251226152206455](./image/image-20251226152206455.png)

1、首先将图片切片成patch,每个patch是16 * 16 * 3

2、加一个cls的token在最前面，然后将每个patch拉平，拉成一个长为768的向量，然后做一个线性映射，先升维，再降维

3、加上位置编码，位置编码用的是**可训练的绝对位置编码**。最后得到input_embedding

![image-20251226183326722](./image/image-20251226183326722.png)

## Clip

![image-20251226190447542](./image/image-20251226190447542.png)

np.dot():计算点积

![image-20260308184552755](./image/image-20260308184552755.png)

np.dot(I_f,W_i) : 这里指将特征线性映射

np.exp(t) : 这里t指温度系数

![image-20260308190058146](./image/image-20260308190058146.png)

##  Swin Transformer

[狗都能看懂的Swin Transformer的讲解和代码实现-CSDN博客](https://blog.csdn.net/weixin_42392454/article/details/141395092?ops_request_misc=elastic_search_misc&request_id=a44266cb920fc3ee8affb3587ed913e6&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-141395092-null-null.142^v102^pc_search_result_base4&utm_term=swin transformer&spm=1018.2226.3001.4187)

是的，在 Swin Transformer 中，**W-MSA（Window Multi-head Self-Attention）和 SW-MSA（Shifted Window Multi-head Self-Attention）是成对出现、交替使用的。**

这是 Swin Transformer 核心设计的精髓。如果只用 W-MSA，模型就失去了跨窗口通信的能力；如果只用普通的全局 Attention，计算量又太大。

------

### 1. 为什么要两个都用？（交替机制）

Swin Transformer 的基本单元是由**两个连续的 Block** 组成的：

- **第 $L$ 层（W-MSA）**：将**特征图划分为不重叠的窗口（Window）**，只在每个窗口内部做注意力计算。这极大地降低了计算量，但缺点是“窗口与窗口之间是孤立的”，没有任何信息交换。
- **第 $L+1$ 层（SW-MSA）**：将**窗口的划分位置进行偏移（Shift）**。通过这种偏移，**原本属于不同窗口的边缘像素现在被划分到了同一个新窗口中**。这样，在这一层做 Attention 时，信息就能实现**跨窗口的交互**。

### 2. 它们的关系：效率与性能的平衡

| **模块**   | **划分方式**         | **主要作用**                       | **缺点**                                             |
| ---------- | -------------------- | ---------------------------------- | ---------------------------------------------------- |
| **W-MSA**  | 规则的、不重叠的窗口 | 极大地降低计算复杂度（线性复杂度） | 缺乏跨窗口的信息交换（感受野受限）                   |
| **SW-MSA** | 偏移后的窗口         | 引入跨窗口连接，扩大感受野         | 窗口数量增加，计算变得复杂（通过 Cyclic Shift 解决） |

### 3. SW-MSA 的巧妙之处：循环移位 (Cyclic Shift)

你可能会问：窗口一偏移，边缘剩下的那些不规则的小块怎么办？

为了保持计算效率，Swin Transformer 引入了 **Cyclic Shift（循环移位）**：

1. 把偏移后多出来的边角部分移到对角线上，拼成一个规则的矩形。
2. 通过 **Masked MSA（掩码注意力）** 来确保计算时，只有在原图中相邻的像素才会互相计算权重，避免了把本不相关的边角信息算在一起。

### 总结

在代码实现或官方架构图中，你会看到：

- **Layer 1**: W-MSA
- **Layer 2**: SW-MSA
- **Layer 3**: W-MSA
- **Layer 4**: SW-MSA
- ...

这种**交替堆叠**的设计，使得 Swin Transformer 既拥有了类似 CNN 的局部感应野（Local Receptive Field）和层级结构，又保留了 Transformer 捕获长距离依赖的能力。

既然你之前在研究相对位置偏置，你会发现 **W-MSA 和 SW-MSA 都会共用同一套相对位置偏置的逻辑**。你想看看这两者在代码层面上是如何通过 `mask` 来区分计算的吗？
