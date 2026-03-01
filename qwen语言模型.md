![image-20260113200944782](./image/image-20260113200944782.png)

## Qwen

<img src="./image/5efb410b-b91e-4c4f-8a2f-e485a0e9f064.png" alt="5efb410b-b91e-4c4f-8a2f-e485a0e9f064"  />

### Qwen3

多语言

MOE架构没有用共享专家

三阶段预训练：

1、general state：提升通用的理解能力

2、reasoning state: 提升推理能力，在代码、推理数据上进行

3、长上下文数据，提高rope的频率从10000到1000000，使用yarn和DCA(Dual Chunk Attention)

旗舰模型后训练阶段：

1、上思维链冷启动 掌握初步推理能力

2、RL GRPO 

3、为了让模型在快慢模式上的思考能力  SFT 推理数据和非推理数据

4、通用RL 

通用模型后训练：

1、off-policy ：教师模型产生think和非think数据，进行sft

1、on-policy:  学生模型生成输出→对齐教师模型的概率分布（通过 KL 散度约束）

主要为了解决think control 和 蒸馏小模型



## Qwen3-Next

【闭关两周半，完全从零开始实现Qwen3-Next模型（参数60M，线性注意力GatedDeltaNet），从架构原理到代码实现，绝对是你能找到的最详细的讲解】https://www.bilibili.com/video/BV1f8sdziERA?vd_source=c5c396652c0c83be15efe54e0c348c90

【【技术前沿】Qwen3-Next-80B架构详解，线性注意力革命,下一代Attention模型已来！采用稀疏的MOE架构！通义大模型 AI大模型  qwen3】https://www.bilibili.com/video/BV1k9pqznEEz?vd_source=c5c396652c0c83be15efe54e0c348c90

![image-20260217171657256](./image/image-20260217171657256.png)

![image-20260217172005179](./image/image-20260217172005179.png)

![image-20260217172952314](./image/image-20260217172952314.png)

![image-20260217173253723](./image/image-20260217173253723.png)

![image-20260217173432542](./image/image-20260217173432542.png)

![image-20260217174220132](./image/image-20260217174220132.png)

![image-20260217174349864](./image/image-20260217174349864.png)



![image-20260217174644789](./image/image-20260217174644789.png)

![image-20260217180642875](./image/image-20260217180642875.png)

![image-20260217180928974](./image/image-20260217180928974.png)

![image-20260217181909167](./image/image-20260217181909167.png)

![image-20260217182246477](./image/image-20260217182246477.png)

QWEN3.5

[(15 条消息) 【LLM】Qwen3.5解剖 - 知乎](https://zhuanlan.zhihu.com/p/2005306558997882654)