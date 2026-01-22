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