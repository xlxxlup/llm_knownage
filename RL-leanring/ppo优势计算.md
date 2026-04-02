### PPO

这份报告将从 **模型角色**、**数据流转**、**核心数学算式**以及**工程实现细节**四个维度，为你彻底复盘 PPO（Proximal Policy Optimization）在大型语言模型（LLM）中的训练全链路。

------

## 1. 核心角色分配 (The Four Pillars)

在 `trl` 或类似的 RLHF 框架中，PPO 训练通常涉及四个模型（或四个逻辑角色）：

| **角色**                 | **物理实体**                               | **是否更新** | **输入**           | **输出**                     |
| ------------------------ | ------------------------------------------ | ------------ | ------------------ | ---------------------------- |
| **Actor (策略模型)**     | 待优化的 LLM                               | **是**       | `query + response` | 每一个 Token 的 Logits       |
| **Critic (价值模型)**    | 回归模型 (通常是 Actor 加一个 Scalar Head) | **是**       | `query + response` | 每一个 Token 的预测价值 $V$  |
| **Reference (参考模型)** | SFT 后的冻结 LLM                           | 否           | `query + response` | 每一个 Token 的参考 Logprobs |
| **Reward (奖励模型)**    | 冻结的分类/评分模型                        | 否           | `query + response` | 整个序列的一个标量分值 $S$   |

------

## 2. 训练阶段一：Rollout (采样与离线评估)

这是数据的“生产”阶段。

1. **采样 (Sampling)**：Actor 模型根据 `query` (Prompt) 生成 `response`。
2. **Logprobs 计算**：
   - Actor 计算当前生成的 $\log \pi_{\theta}(a_t|s_t)$。
   - Reference 计算基准的 $\log \pi_{ref}(a_t|s_t)$。
3. **价值评估 (Value Inference)**：Critic 模型对整个序列跑一遍前向，得到每个位置的价值估计 $V_t$。
4. **打分 (Rewarding)**：Reward 模型对完整的序列 `query + response` 给出一个全局分数 $Score$。

------

## 3. 训练阶段二：数据预处理 (Token-level Reward Construction)

这是 PPO 最精细的地方。虽然 $Score$ 只有一个，但 PPO 需要**每一个 Token** 都有奖励。

### 3.1 奖励序列构建 ($r_t$)

对于 Response 中的每一个 Token $t$：

- **KL 惩罚**：$kl_t = \log \pi_{\theta}(a_t|s_t) - \log \pi_{ref}(a_t|s_t)$

- **即时奖励**：

  $$r_t = \begin{cases} -\beta \cdot kl_t, & t < T \\ Score - \beta \cdot kl_t, & t = T \end{cases}$$

  *注：只有最后一个 Token 会加上 RM 的分数，前面全是 KL 惩罚。*

### 3.2 优势与回报计算 (GAE)

使用 **GAE (Generalized Advantage Estimation)** 将即时奖励转化为优势函数 $A_t$：

1. **TD Error**：$\delta_t = r_t + \gamma V_{t+1} - V_t$
2. **Advantage ($A_t$)**：$A_t = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \dots$
3. **Return ($R_t$)**：$R_t = A_t + V_t$（这是 Critic 的学习目标）。

> **核心细节**：因为 $A_t$ 的计算是向后回溯的，所以即便前面的 Token 即时奖励 $r_t$ 只有 KL 惩罚，但通过 $\delta_{t+1}$ 等项，最后一个 Token 的大奖 $Score$ 会被“折现”回每一个中间 Token。

------

## 4. 训练阶段三：Optimization (梯度更新)

拿到 $A_t$ 和 $R_t$ 后，开启 `ppo_epochs` 次循环更新。

### 4.1 Actor Loss (策略裁剪损失)

$$L^{clip} = - \mathbb{E} \left[ \min \left( \frac{\pi_{new}}{\pi_{old}} A_t, \text{clip}\left(\frac{\pi_{new}}{\pi_{old}}, 1-\epsilon, 1+\epsilon\right) A_t \right) \right]$$

- **输入**：当前时刻的 Logprobs、采样时的 Logprobs、计算好的优势 $A_t$。
- **逻辑**：如果某个动作 $A_t > 0$（表现好），就调大它的概率，但最大不能超过 $1+\epsilon$ 倍。

### 4.2 Critic Loss (价值均方误差)

$$L^{vf} = \mathbb{E} \left[ (V_{new}(s_t) - R_t)^2 \right]$$

- **输入**：Critic 预测的 $V_t$、计算好的回报 $R_t$。
- **逻辑**：让 Critic 学会准确预估“在这个节点往后走，总分能拿多少”。**此处覆盖 Response 的所有 Token。**

### 4.3 Entropy Loss (熵损失)

$$L^{ent} = - \mathbb{E} [ H(\pi_{\theta}(s_t)) ]$$

- **逻辑**：防止模型输出变得太单一，维持探索能力。

------

## 5. 关键工程细节 (Deep Dive)

1. **Masking (掩码)**：
   - 在计算所有 Loss 时，必须通过 Mask 矩阵**忽略掉 Prompt 部分**。梯度只在 Response 部分产生。
2. **Advantage Normalization**：
   - 在一个 Batch 内，通常会将 $A_t$ 进行标准化（减均值除方差），这能显著稳定训练，防止某些长难句产生的巨大 Reward 破坏梯度。
3. **Precision (精度)**：
   - 通常 Actor 用 BF16/FP16，但 **Critic 模型建议用 FP32 或较高的精度**，因为 $V$ 值是一个回归值，对数值波动非常敏感。
4. **The $\beta$ Scheduler**：
   - KL 惩罚系数 $\beta$ 有时是动态的。如果 KL 漂移太大，就调大 $\beta$ 强制模型“回家”；如果 KL 太小，就调小 $\beta$ 放任模型去拿高分。

------

## 6. 总结：为什么 PPO 这么设计？

- **为什么要 Actor-Critic？** Actor 负责跑，Critic 负责评价。没有 Critic，Actor 很难在复杂的长文本生成中搞清楚到底是哪个词写得好（信誉分配问题）。
- **为什么要 Clip？** 因为采样太贵了，我们想用同一批数据多练几次。Clip 保证了即便练了好几遍，模型也不会跑得太远导致数学失效。
- **为什么要全序列 Critic Loss？** 虽然 RM 只给最后打分，但 RL 的本质是序列决策。我们需要 Critic 具备“见微知著”的能力，在句子刚说了一半时就能预判结局。

这份流程体现了 PPO 如何在“获取高分”与“不跑偏”之间寻找微细的平衡。你现在的工程任务，是打算在 **DeepSpeed-Chat** 还是 **Verl** 框架下跑这一套逻辑？这两者的并行化实现（尤其是 Actor 和 Critic 的切分）会有细微差异。



![image-20251223140136461](./image/image-20251223140136461.png)

![image-20251223141008887](./image/image-20251223141008887.png)

![image-20251223142306577](./image/image-20251223142306577.png)

![image-20251223143328979](./image/image-20251223143328979.png)

![image-20251223143851054](./image/image-20251223143851054.png)

![image-20251223144109107](./image/image-20251223144109107.png)

![image-20251223145320636](./image/image-20251223145320636.png)

![image-20251223150031180](./image/image-20251223150031180.png)

![image-20260101145326225](./image/image-20260101145326225.png)

![image-20260101145612484](./image/image-20260101145612484.png)

![image-20260101145857926](../llm八股.assets/image-20260101145857926.png)







# 固定参数下的自然推导计算

我现在**完全不调整任何参数**，基于 “固定的基础参数” 从头完整、正确推导所有步骤 —— 包括 delta（单步优势）、GAE 优势（advantages）、returns 的计算，全程不修改 rewards/values 等参数，确保每一步数值都是自然推导的结果。

### 第一步：明确所有固定基础参数（无任何微调）

设定 3 步序列（t=0、t=1、t=2，t=2 是最后一步），所有参数均为 “真实场景下的固定值”（奖励、状态价值由数据 / 模型给出，不调整）：



| 变量类别     | 变量名               | 取值                                        | 核心说明                                               |
| -------- | ----------------- | ----------------------------------------- | -------------------------------------------------- |
| 超参数      | $\gamma$（折现因子）    | 0.9                                       | 未来收益的时间折扣（t+1 步的收益折算到 t 步的权重）                      |
| 超参数      | $\lambda$（GAE 衰减） | 0.95                                      | 控制远期优势对当前步的影响权重                                    |
| 状态价值（预期） | valuesₜ           | values₀=10、values₁=8、values₂=5            | 模型对各步状态的 “预期总收益”（固定，模型输出）                          |
| 即时奖励     | rewardsₜ          | r₀=2、r₁=4、r₂=6                            | 各步执行动作后拿到的即时奖励（固定，数据给出）                            |
| 下一步价值    | nextvaluesₜ       | nextvalues₀=8、nextvalues₁=5、nextvalues₂=0 | t=0 的下一步是 t=1（故 nextvalues₀=values₁）；t=2 无下一步，故为 0 |

### 第二步：计算每一步的「单步优势 delta」（无调整，纯公式计算）

delta 是 GAE 的基础，公式为：

$\delta_t = \text{rewards}_t + \gamma \times \text{nextvalues}_t - \text{values}_t$

逐步计算，完全基于固定参数，不修改任何值：

#### 1. t=2（最后一步）的 delta

$\delta_2 = r_2 + \gamma \times \text{nextvalues}_2 - \text{values}_2$

代入固定值：

$\delta_2 = 6 + 0.9 \times 0 - 5 = 6 - 5 = 1$

#### 2. t=1 的 delta

$\delta_1 = r_1 + \gamma \times \text{nextvalues}_1 - \text{values}_1$

代入固定值：

$\delta_1 = 4 + 0.9 \times 5 - 8 = 4 + 4.5 - 8 = 0.5$

#### 3. t=0 的 delta

$\delta_0 = r_0 + \gamma \times \text{nextvalues}_0 - \text{values}_0$

代入固定值：

$\delta_0 = 2 + 0.9 \times 8 - 10 = 2 + 7.2 - 10 = -0.8$

✅ 最终 delta 结果（无调整）：



* δ₂=1、δ₁=0.5、δ₀=-0.8

### 第三步：从后往前递归计算「GAE 优势（advantages）」（无调整）

核心递归公式（从后往前算，因为下一步的 GAE 优势先算好）：

$\text{advantages}_t = \delta_t + \gamma \times \lambda \times \text{advantages}_{t+1}$

（最后一步 t=2 的下一步无值，故$\text{advantages}_{3}=0$）

#### 步骤 1：计算 t=2 的 GAE 优势（advantages₂）

t=2 是最后一步，无后续步骤，因此：

$\text{advantages}_2 = \delta_2 + \gamma \times \lambda \times \text{advantages}_3$

代入值：

$\text{advantages}_2 = 1 + 0.9 \times 0.95 \times 0 = 1$

#### 步骤 2：计算 t=1 的 GAE 优势（advantages₁）

t=1 的下一步是 t=2，已算出$\text{advantages}_2=1$，因此：

$\text{advantages}_1 = \delta_1 + \gamma \times \lambda \times \text{advantages}_2$

先算$\gamma \times \lambda = 0.9 \times 0.95 = 0.855$，再代入：

$\text{advantages}_1 = 0.5 + 0.855 \times 1 = 1.355$

#### 步骤 3：计算 t=0 的 GAE 优势（advantages₀）

t=0 的下一步是 t=1，已算出$\text{advantages}_1=1.355$，因此：

$\text{advantages}_0 = \delta_0 + \gamma \times \lambda \times \text{advantages}_1$

代入值：

$\text{advantages}_0 = -0.8 + 0.855 \times 1.355$

先算乘法：$0.855 \times 1.355 ≈ 1.1585$，再求和：

$\text{advantages}_0 ≈ -0.8 + 1.1585 = 0.3585$

✅ 最终 GAE 优势结果（纯自然推导，无调整）：



* t=0：advantages₀≈0.3585

* t=1：advantages₁=1.355

* t=2：advantages₂=1

### 第四步：计算「returns（实际折现总收益）」（无调整）

核心公式（优势定义的直接变形）：

$\text{returns}_t = \text{advantages}_t + \text{values}_t$

逐步计算：

#### 1. t=2 的 returns

$\text{returns}_2 = \text{advantages}_2 + \text{values}_2 = 1 + 5 = 6$

#### 2. t=1 的 returns

$\text{returns}_1 = \text{advantages}_1 + \text{values}_1 = 1.355 + 8 = 9.355$

#### 3. t=0 的 returns

$\text{returns}_0 = \text{advantages}_0 + \text{values}_0 ≈ 0.3585 + 10 = 10.3585$

✅ 最终 returns 结果（无调整）：



* t=0：returns₀≈10.3585

* t=1：returns₁=9.355

* t=2：returns₂=6

### 关键补充：解释之前的 “参数微调”

之前我调整 rewards 是为了**匹配示例中给出的 advantages 数值（5、7.275、7.22）**，方便你理解逻辑；但真实训练中，**所有参数（rewards、values）都是固定的**——rewards 由环境 / 数据给出，values 由模型输出，delta 和 GAE 优势都是基于这些固定值自然计算的，不会人为调整。

核心逻辑永远不变：



1. delta 是 “单步实际收益 - 单步预期收益”（带折现）；

2. GAE 优势是 “从后往前递归，把多步 delta 加权累积”；

3. returns = advantages + values 是 “实际总收益 = 超出预期的部分 + 预期收益” 的变形。

### 下一步迭代建议

需要我把这套 “固定参数 + 自然推导” 的计算过程写成可运行的 Python 代码，你可以直接运行并修改参数（比如改 γ/λ/rewards），实时看到 delta、advantages、returns 的变化吗？

> （注：文档部分内容可能由 AI 生成）





这是一个非常经典且深刻的问题。简单来说：**Reward Model 是“裁判”，而 Value Model 是“军师”。**

虽然它们都和“得分”有关，但在 PPO（以及所有 Actor-Critic 架构）中，Value Model 的存在是为了解决强化学习中最头疼的问题：**高方差（Variance）\**和\**归因难题（Credit Assignment）**。

以下是为什么要“多此一举”引入 Value Model 的核心逻辑：

------

### 1. 裁判 vs. 预期 (Reward vs. Value)

- **Reward Model ($R$):** 它像一个裁判，只负责对模型生成的**最终结果**打分（比如：这篇文章写得好不好？代码跑得通吗？）。它告诉你的是“现状”。
- **Value Model ($V$):** 它像一个军师，在模型生成每一个词、做每一个动作时，都在**预判**：“基于现在的状态，我最终能拿多少分？”

### 2. 核心原因：降低方差 (Variance Reduction)

如果我们只用 Reward，训练会非常不稳定。

假设模型写了两段话，一段得了 0.9 分，另一段得了 0.8 分。

- **只看 Reward：** 策略梯度会觉得两段都不错，都要加强。
- **有了 Value：** 如果 Value Model 预判这段话原本能得 0.95 分，结果只得了 0.9 分。虽然 0.9 是高分，但**低于预期**，Value Model 会告诉模型：“这其实退步了，得改。”

通过对比“实际得分”和“预期得分”，我们得到了 **优势函数 (Advantage Function)**：

$$A(s, a) = Q(s, a) - V(s)$$

PPO 优化的是这个 **$A$ (优势)**，而不是原始的 $R$。这样可以过滤掉环境自带的背景噪音，让模型专注于那些“真正让结果变好”的动作。

### 3. 解决“跨时空”的归因问题

在生成长文本时，Reward 通常在最后才给（Sparse Reward）。

- **没有 Value：** 模型只知道最后结果很好，但不知道是第 3 行写得好，还是第 50 行写得好。
- **有了 Value：** Value Model 会给每一个 token 估计一个价值。如果写到第 5 行时，Value 突然从 0.5 飙升到 0.8，算法就知道：“刚才那几步走得真妙！”

这就是所谓的**信度分配 (Credit Assignment)**，它把遥远的奖励拆解到了每一个细小的步骤中。

------

### 总结一下

| **维度**     | **Reward Model**         | **Value Model (Critic)**       |
| ------------ | ------------------------ | ------------------------------ |
| **本质**     | 外部反馈，定义什么是“好” | 内部预测，估计未来的“总回报”   |
| **存在意义** | 提供学习目标             | **稳定**训练过程，减少盲目尝试 |
| **实时性**   | 通常在序列结束时给出     | 在生成过程中的每一步都在计算   |

如果没有 Value Model，PPO 就会退化成原始的 Policy Gradient（如 REINFORCE），那训练效率和稳定性会让你想砸电脑。

**你想深入了解一下 Value Model 是如何通过 GAE（广义优势估计）来进一步平衡偏差和方差的吗？**





既然你要准备面试，我们把这个问题拆解为“面试金句”和“深度原理解析”两部分。

------

## 1. 面试回答总结：为什么有了 Reward 还要 Value？

如果在面试中被问到，你可以按照这个逻辑链条来回答，显得既专业又有系统性思维：

- **核心定义（定性）：** **Reward Model 是客观的“裁判”**，给出最终得分（Ground Truth）；而 **Value Model 是主观的“预言家”**，预测当前状态下的期望回报。
- **三大核心价值（定量）：**
  1. **降低方差（Variance Reduction）：** 只用 Reward 会让梯度更新受随机性影响巨大（比如运气好抽到一个高分）。通过引入 Value 作为基准（Baseline），我们只学习“超出预期”的部分（即 Advantage），从而过滤掉环境背景噪音，让训练更稳。
  2. **解决信度分配（Credit Assignment）：** Reward 通常是稀疏且滞后的（只有最后才给分）。Value Model 将长期的奖励拆解到每一个中间步骤，让模型知道具体是哪一个动作导致了最终的好或坏。
  3. **计算优势函数（Advantage）：** PPO 优化的目标是 $A(s, a) = Q - V$。没有 Value，我们就无法判断一个动作是“真的好”还是“仅仅因为当前状态本来就容易拿高分”。

------

## 2. 深度进阶：Value Model 如何通过 GAE 平衡偏差与方差？

在强化学习中，我们要估计的是**优势函数 $A_t$**。GAE（Generalized Advantage Estimation）的出现，本质上是为了解决 **“看多远”** 的问题。

### 偏差（Bias）与方差（Variance）的博弈

在估计回报时，我们有两种极端：

- **TD(0) / 1-step Return ($\lambda=0$):** 只看一步奖励，后面全靠 Value Model 猜。
  - *方差低：* 因为只用了一步真实奖励，随机性小。
  - *偏差高：* 如果 Value Model 预测不准，你的估计就全错了。
- **Monte Carlo / $\infty$-step Return ($\lambda=1$):** 一直跑完整个序列，把所有 Reward 加起来。
  - *偏差为 0：* 这是真实的实验结果。
  - *方差极高：* 序列越长，路径中的随机性累积越多，训练极不稳定。

### GAE 的魔力：加权平均

GAE 不再纠结于选“1步”还是“全步”，它通过一个超参数 $\lambda$ 对所有 $n$ 步的估计做了**指数加权平均**。

首先定义 TD 残差（一步之内的误差）：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

GAE 的公式如下：

$$\hat{A}_t^{GAE}(\gamma, \lambda) = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

### 如何实现平衡？

- **当 $\lambda \to 0$：** 公式退化为 $\delta_t$。这对应 **TD(0)**，完全相信 Value Model。此时**方差最小，但偏差最大**（依赖模型准确度）。
- **当 $\lambda \to 1$：** 公式累积了序列中所有的 $r_t$。这对应 **Monte Carlo**。此时**偏差为 0，但方差最大**（受路径随机性影响）。

**结论：**

通过调节 $\lambda$（通常取 0.95），GAE 在“信任模型预测（低方差）”和“信任真实奖励（低偏差）”之间找到了黄金分割点。**Value Model 越准，我们就可以把 $\lambda$ 调小一点，让训练更快收敛。**

------

这种“用偏置换稳定”的思想是 PPO 成为目前最稳健算法的核心原因。

**你想看看在具体的 Python 代码（如 Stable Baselines3 或 Transformer 库）中，GAE 是如何被计算出来的吗？**