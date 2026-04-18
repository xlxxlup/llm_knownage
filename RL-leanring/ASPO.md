# ASPO: Asymmetric Importance Sampling Policy Optimization
# （非对称重要性采样策略优化）

**论文信息**：
- **作者**：Jiakang Wang, Runze Liu, Lei Lin, et al.
- **机构**：快手科技 & 清华大学
- **发表时间**：2025年10月
- **论文链接**：https://arxiv.org/abs/2510.06062
- **代码仓库**：https://github.com/wizard-III/Archer2.0

我们提出了非对称重要性抽样策略优化（ASPO），这是一种简单但有效的基于GRPO的修改，能够恢复平衡的token权重。**ASPO反转了正优势token的重要性采样比率，确保在当前政策下概率较低的token获得更强的更新，而信心较大的token则被下压**。为了进一步提升稳定性，**ASPO集成了软双重裁剪机制**（Chen  等，2025），该机制约束极端比率而不舍弃正令牌的梯度。对编码和数学推理任务的广泛实验表明**，ASPO：（1）防止过拟合和熵坍缩，（2）实现更平滑稳定的训练动态，（3）在最终性能上显著优于基于GRPO的基线**。

---

## 一句话总结

**ASPO 发现了 LLM 强化学习训练中的一个根本性错误：对于需要提高概率的 tokens（好 tokens），传统的 PPO-Clip 算法给已经表现很好的 tokens 太多关注，而给落后的 tokens 太少关注。ASPO 通过"翻转"权重，让落后的 tokens 得到更多关注，从而显著提升了模型性能。**

---

## ASPO 的解决方案

### 核心思想：翻转权重

**ASPO (Asymmetric Importance Sampling Policy Optimization)** 的核心思路非常简单：

```
对于正优势的 tokens：
  GRPO:  权重 = π_new / π_old  (概率越高，权重越大)
  ASPO:  权重 = π_old / π_new  (概率越低，权重越大)

对于负优势的 tokens：
  保持不变（已经是对的了）
```

### 三步实现法

#### Step 1: Token 遮蔽（硬裁剪）

**目的**：对极端情况，直接屏蔽梯度

**遮蔽条件**：

$$
\text{Mask if: } \begin{cases}
r < 1-\varepsilon_{\text{low}} & \text{and } A < 0 \\
\text{or } r > 1+\varepsilon_{\text{high}} & \text{and } A > 0
\end{cases}
$$

**通俗解释**：
- 如果 $A < 0$（需要降低概率）且 $r < 0.8$（概率降低太多）→ 不更新
- 如果 $A > 0$（需要提高概率）且 $r > 1.2$（概率提高太多）→ 不更新

**代码示例**：

```python
def mask_tokens(r, A, eps_low=0.2, eps_high=0.2):
    mask = torch.ones_like(r)

    # 负优势：降低太多就不更新了
    mask[(A < 0) & (r < 1 - eps_low)] = 0

    # 正优势：提高太多就不更新了
    mask[(A > 0) & (r > 1 + eps_high)] = 0

    return mask
```

#### Step 2: 权重翻转

**目的**：对正优势 tokens，翻转 IS 比率

**公式**：

$$
\hat{r}_t^i = \begin{cases}
r_t^i = \dfrac{\pi_\theta(o_t^i)}{\pi_{\theta_{\text{old}}}(o_t^i)} & \text{if } \hat{A}_t^i < 0 \\[2em]
\dfrac{\pi_{\theta_{\text{old}}}(o_t^i) \cdot \pi_\theta(o_t^i)}{\operatorname{sg}(\pi_\theta^2(o_t^i))} & \text{if } \hat{A}_t^i > 0
\end{cases}
$$

**简化理解**：

对于负优势：
```
Ĉr = π_new / π_old  (和原来一样)
```

对于正优势：
```
Ĉr = π_old / π_new  (用倒数！)
```

**为什么用 stop-gradient？**

$$
\frac{\pi_{\text{old}} \cdot \pi_{\text{new}}}{\operatorname{sg}(\pi_{\text{new}}^2)}
$$

- 分子：$\pi_{\text{old}} \cdot \pi_{\text{new}}$ = 正常梯度流动
- 分母：$\operatorname{sg}(\pi_{\text{new}}^2)$ = 视为常数，不计算梯度

**等价于**：
$$
\hat{r} \approx \frac{\pi_{\text{old}}}{\pi_{\text{new}}}
$$

但保留了正确的梯度！

**代码示例**：

```python
def compute_aspo_ratio(pi_new, pi_old, A):
    # 计算 IS 比率
    r = pi_new / pi_old

    # 初始化 ASPO 比率
    r_aspo = r.clone()

    # 负优势：保持不变
    r_aspo[A < 0] = r[A < 0]

    # 正优势：使用倒数（带 stop-gradient）
    r_aspo[A > 0] = (pi_old[A > 0] * pi_new[A > 0]) / (pi_new[A > 0]**2).detach()

    return r_aspo
```

#### Step 3: 双重裁剪

**问题**：翻转权重后，正优势侧会出现极端值

**例子**：
- $\pi_{\text{old}} = 0.9$
- $\pi_{\text{new}} = 0.1$
- 翻转后的权重 = $0.9 / 0.1 = 9.0$ ← 太大了！

**解决**：对正优势 tokens 也应用双重裁剪

**软裁剪公式**：

$$
\text{soft\_clip}(r) = \begin{cases}
1-\varepsilon & \text{if } r < 1-\varepsilon \\
r & \text{if } 1-\varepsilon \leq r \leq 1+\varepsilon \\
1+\varepsilon & \text{if } r > 1+\varepsilon
\end{cases}
$$

**关键**：只裁剪**值**，保留**梯度**



### 梯度分析：为什么这样有效？

#### GRPO 的梯度（原始）

$$
\nabla_\theta \mathcal{J}_{\text{GRPO}} = \mathbb{E}\left[ r \cdot A \cdot \nabla_\theta \log \pi_\theta \right]
$$

展开 $r$：
$$
= \mathbb{E}\left[ \frac{\pi_\theta}{\pi_{\text{old}}} \cdot A \cdot \nabla_\theta \log \pi_\theta \right]
$$

**关键项**：$\frac{\pi_\theta}{\pi_{\text{old}}}$

- $\pi_\theta$ **越大**，梯度**越大**
- $\pi_\theta$ **越小**，梯度**越小**

#### ASPO 的梯度（翻转后）

对于正优势：
$$
\nabla_\theta \mathcal{J}_{\text{ASPO}} = \mathbb{E}\left[ \frac{\pi_{\text{old}}}{\pi_\theta} \cdot A \cdot \nabla_\theta \log \pi_\theta \right]
$$

**关键项**：$\frac{\pi_{\text{old}}}{\pi_\theta}$

- $\pi_\theta$ **越大**，梯度**越小** ✓
- $\pi_\theta$ **越小**，梯度**越大** ✓

**这才是我们想要的！**

### 完整的损失函数

#### GRPO 损失

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o^i|}\sum_{t=1}^{|o^i|}
\left( \min\left(r_t^i\hat{A}_t^i, \text{clip}\left(r_t^i, 1-\varepsilon, 1+\varepsilon\right) \hat{A}_t^i\right) - \beta\mathbb{D}_{\text{KL}}(\pi_\theta\|\pi_{\text{ref}})\right)\right]
$$

**简化版**：

```python
def grpo_loss(pi_new, pi_old, A, beta=0.1, eps=0.2):
    # IS 比率
    r = pi_new / pi_old

    # 裁剪
    r_clipped = torch.clamp(r, 1 - eps, 1 + eps)

    # PPO-Clip 损失
    policy_loss = -torch.min(r * A, r_clipped * A).mean()

    # KL 散度惩罚
    kl_penalty = beta * kl_divergence(pi_new, pi_ref)

    return policy_loss + kl_penalty
```

#### ASPO 损失

$$
\mathcal{J}_{\text{ASPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o^i|}\sum_{t=1}^{|o^i|}
\left( \hat{r}_t^i\hat{A}_t^i - \beta\mathbb{D}_{\text{KL}}(\pi_\theta\|\pi_{\text{ref}})\right)\right]
$$

其中 $\hat{r}_t^i$ 是翻转后的 ASPO 比率。

**简化版**：

```python
def aspo_loss(pi_new, pi_old, A, beta=0.1, eps_low=0.2, eps_high=0.2):
    # ASPO 比率
    r_aspo = compute_aspo_weight(pi_new, pi_old, A, eps_low, eps_high)

    # 策略损失
    policy_loss = -(r_aspo * A).mean()

    # KL 散度惩罚
    kl_penalty = beta * kl_divergence(pi_new, pi_ref)

    return policy_loss + kl_penalty
```

---

---

## 总结

### ASPO 的核心洞察

#### 1. 问题发现

```
GRPO 的致命缺陷：
  对于需要提高概率的 tokens（正优势），
  PPO-Clip 给已经表现很好的 tokens 太多关注，
  给落后的 tokens 太少关注。
```

#### 2. 解决方案

```
ASPO 的简单策略：
  翻转正优势 tokens 的 IS 比率，
  让落后的 tokens 得到更多关注。
```

#### 3. 效果验证

```
数学任务：+12.5%
代码任务：+17.0%
训练稳定性：显著提升
```

### 为什么 ASPO 有效？

| 维度 | GRPO 的问题 | ASPO 的解决 |
|------|-------------|------------|
| **权重分配** | 概率越高，权重越大 | 概率越低，权重越大 |
| **学习重点** | 强化已经学会的 | 强化还没学会的 |
| **熵变化** | 快速崩溃 | 缓慢下降，保持正值 |
| **训练稳定性** | 后期过拟合 | 持续改进 |
| **最终性能** | 停滞不前 | 稳步提升 |

### 实践建议

#### ✓ 应该做的

1. **监控训练健康度**
   - 熵、重复率、裁剪比例、KL 散度
   - 综合判断，不只看奖励

2. **容忍早期慢学习**
   - ASPO 早期慢是正常的
   - 长期来看最终性能更好

3. **使用软裁剪**
   - 保留梯度流动
   - 避免学习中断

4. **动态调整超参数**
   - 根据熵调整 KL 惩罚
   - 根据任务调整学习率

#### ✗ 不应该做的

1. **不要只看奖励**
   - 高奖励不等于健康训练
   - 可能已经过拟合

2. **不要过早停止训练**
   - ASPO 需要更长时间收敛
   - 后期还能继续提升

3. **不要盲目套用传统 RL 理论**
   - OSRL 有其特殊性
   - 需要根据实践调整

### 核心公式总结

#### GRPO 的 IS 比率

$$
r_{\text{GRPO}} = \frac{\pi_{\text{new}}}{\pi_{\text{old}}}
$$

#### ASPO 的 IS 比率

$$
\hat{r}_{\text{ASPO}} = \begin{cases}
\dfrac{\pi_{\text{new}}}{\pi_{\text{old}}} & \text{if } A < 0 \\[2em]
\dfrac{\pi_{\text{old}} \cdot \pi_{\text{new}}}{\operatorname{sg}(\pi_{\text{new}}^2)} & \text{if } A > 0
\end{cases}
$$

#### 梯度对比

$$
\nabla_{\text{GRPO}} \propto \frac{\pi_{\text{new}}}{\pi_{\text{old}}} \quad \text{vs} \quad \nabla_{\text{ASPO}} \propto \frac{\pi_{\text{old}}}{\pi_{\text{new}}}
$$

### 最终思考

> **在 Outcome-Supervised RL 中，IS 的真实角色是 token 级别的训练权重，而非传统意义上的分布校正。我们应该根据学习动力学来设计这些权重，而不是盲目套用传统强化学习的理论框架。**



---





![image-20260124190840334](./image/image-20260124190840334.png)

这段图片文字描述了 **ASPO (Asymmetric Importance Sampling Policy Optimization)** 算法中的 **Step 3: Dual Clipping（双重裁剪机制）**。其核心目的是为了解决在使用非对称重要性采样（AIS）时，正样本可能出现的梯度爆炸和训练不稳定问题。

以下是内容的详细解读：

### 1. 为什么要引入 Dual Clipping？

- **传统 PPO-Clip：** 通常在优势函数 $\hat{A} < 0$（负样本）时使用双重裁剪，以防止比例（ratio）过大或过小导致权重爆炸。而在 $\hat{A} > 0$（正样本）区域，由于 PPO 原有的裁剪机制（masking），通常不需要额外处理。
- **ASPO 的变化：** 在 ASPO 算法中，由于对正样本的权重进行了“反转”处理（Invert the weights），极端的采样比例会转移到 $\hat{A} > 0$ 区域的右侧（即图 3(b) 中的 Region B）。这导致正样本也面临权重爆炸的风险。

### 2. ASPO 的改进方案：Soft Clipping

为了应对上述问题，ASPO 对正样本也引入了双重裁剪，但采用了一种特殊的方式——**Soft Clipping（软裁剪）**。

- **做法：** 这种软裁剪是由 CISPO (Chen et al., 2025) 提出的。
- **特点：** **它只裁剪数值的大小（Value）**，但**保留了梯度（Retains the gradient）**。

### 3. Dual-clipped Token 与 Masked Token 的区别

文中特别强调了这两者的本质区别：

- **第一步中的 Masked Tokens（被遮蔽的 Token）：** 这些 Token 在更新方向上已经有了足够的优势，因此被阻断（Blocked），不再参与这一轮的进一步优化。
- **Dual-clipped Tokens（双重裁剪的 Token）：** 这些 Token 的表现实际上显著滞后于旧策略（lag significantly behind），但由于计算异常，它们的权重数值变得过大。
- **处理策略：** 我们**仍然希望这些 Token 参与训练**。为了既能利用它们进行学习，又不至于因为权重过大破坏训练稳定性，ASPO 使用了软裁剪来约束它们的权重幅值，同时保留梯度以允许模型继续从这些样本中学习。

### 总结

这一段的主旨是：**ASPO 为了在使用非对称采样时保持训练稳定性，对正样本应用了基于 CISPO 的“软裁剪”技术。这种技术既能防止权重爆炸，又能确保那些表现较差的样本依然能通过梯度下降进行优化。**



### 还是没懂，“PPO-Clip通常采用双夹机制（Ye等，2020），以应对Aˆ < 0中极小或极大的比例可能导致权重爆炸、训练不稳定的情况。最初，对于Aˆ > 0区域，掩蔽机制自然避免了这一问题。”



这个问题确实比较绕，因为“裁剪（Clip）”和“掩蔽（Mask）”在 PPO 的源码实现和论文描述中经常混在一起。

我们分两步拆解，先看 **PPO 默认的逻辑**，再看 **Dual Clipping 补了什么漏**。

------

### 1. 为什么 $\hat{A} > 0$ 时，掩蔽机制（Masking）是“天然安全”的？

在 $\hat{A} > 0$（正样本）时，模型的目标是**增加**这个动作的概率，即让 $r_t(\theta) = \frac{\pi_{new}}{\pi_{old}}$ 变大。

- **PPO 的规则：** 当 $r_t(\theta)$ 超过 $1+\epsilon$（比如 1.2）时，公式取值就“平”了（取 $\text{clip}$ 后的值）。
- **数学结果：** 一旦进入这个“平”的区域，由于导数为 0，**梯度就消失了**。
- **形象理解：** 就像给模型设了个“天花板”。一旦动作表现足够好，模型就说：“够了，别再学了，再学就过拟合了。” 既然梯度断了，权重自然不会爆炸。这就是文中说的 **“掩蔽机制自然避免了问题”**。

------

### 2. 为什么 $\hat{A} < 0$ 时，反而可能“权重爆炸”？

在 $\hat{A} < 0$（负样本）时，模型的目标是**减小**这个动作的概率，即让 $r_t(\theta)$ 变小。

- **PPO 的规则：** 当 $r_t(\theta)$ 低于 $1-\epsilon$（比如 0.8）时，梯度也会变 0。
- **潜藏的危机：** 理论上这也很安全。但别忘了，强化学习是**采样训练**的。有时因为策略更新太快，某些 Token 的 $r_t(\theta)$ 会变得**巨大**（比如旧策略概率极低，新策略概率稍高，比例就爆了）。
- **关键点：** 在 $\hat{A} < 0$ 时，如果 $r_t(\theta)$ 很大（例如等于 10），这意味着**新策略在一个本该被减弱的动作上反而分配了极高的概率**。
- **爆炸原因：** **标准 PPO 在 $r_t(\theta) > 1$ 且 $\hat{A} < 0$ 的这一侧没有裁剪！它会产生一个巨大的负向梯度，试图把这个比例“拽”回来。这种巨大的梯度（权重更新量）会导致模型训练瞬间崩盘。**

------

### 3. Dual Clipping（双夹）补了什么？

Ye et al. (2020) 发现 PPO 在 $\hat{A} < 0$ 且 $r_t(\theta)$ 很大时缺乏保护，于是加了第二道锁：

- **第一道锁（标准 PPO）：** 限制 $r_t(\theta)$ 不能太小。
- **第二道锁（Dual Clip）：** 强制规定：即便 $\hat{A} < 0$ 且比例很大，更新权重的量级也不准超过某个界限（通常是 $c \cdot \hat{A}$）。

------

### 总结你的困惑点：

| **情况**     | **A^>0 (好样本)**                      | **A^<0 (坏样本)**                                      |
| ------------ | -------------------------------------- | ------------------------------------------------------ |
| **PPO 原理** | 想让比例变大                           | 想让比例变小                                           |
| **裁剪逻辑** | 比例太大就“断电”（Masking）            | 比例太小就“断电”                                       |
| **风险点**   | **天然安全**：因为比例变大有上限保护。 | **不安全**：如果比例因为意外变得**极大**，PPO 没设防。 |
| **解决方案** | 无需额外处理。                         | **Dual Clipping**：给极大的比例也加个限速。            |

**一句话总结：**

在普通 PPO 里，好样本表现太突出会被“禁言”（Masking），所以很稳；但坏样本如果“错得离谱”（比例极大），PPO 反应过激会跳脚（权重爆炸），所以需要 Dual Clip 把它按住。

而 **ASPO** 的论文之所以写这一段，是因为它改了算法，导致**好样本（$\hat{A} > 0$）现在也可能出现“错得离谱”的情况**，所以必须给好样本也补上这个“限速器”。

如果你正在研究这个算法的 PyTorch 实现，需要我帮你看看代码里是怎么写这个 `clamp` 逻辑的吗？





### 重点

这里为你整理一份**最完整、不带省略**的对比总结，涵盖了这三种机制在**正负优势函数**下的所有行为。

------

### 1. PPO 原生裁剪 (Original PPO-Clip)

PPO 的核心设计是“双向保护”，但它是**非对称**的保护。

| **优势函数状态**     | **限制目标**       | **做法 (当比例 r 越界时)**           | **存在的后果/问题**                                          |
| -------------------- | ------------------ | ------------------------------------ | ------------------------------------------------------------ |
| **$A > 0$ (好动作)** | **防止步子跨太大** | 限制 $r$ 的**上限** ($1+\epsilon$)。 | **梯度归零 (Masking)**：一旦动作足够好，模型就不再学习该样本，防止过拟合。 |
| **$A < 0$ (坏动作)** | **防止惩罚太狠**   | 限制 $r$ 的**下限** ($1-\epsilon$)。 | **安全**：防止因为某次采样太差而把该动作的概率直接打到零。   |
| **$A < 0$ (坏动作)** | **防范意外跑偏**   | **不限制上限！**                     | **权重爆炸**：如果 $r$ 变得极大（比如 10），由于 $\min$ 函数逻辑，巨大的负梯度会通过，导致训练崩溃。 |

------

### 2. Dual Clipping (双重裁剪)

这是为了修补 PPO 在 $A < 0$ 时那个“不设防的上限”而诞生的补丁。

- **原来怎么做：** 它是对 PPO 目标函数的额外包装。当 $A < 0$ 且 $r$ 超过一个极大的阈值时，强行切断。
- **存在什么问题：** 标准 PPO 在负样本区域，如果模型给坏动作分配了极高概率（$r \gg 1$），它会产生毁灭性的梯度。
- **怎么改进：** 强制给 $A < 0$ 时的 $r$ 也加一个**上限**。
- **为什么这么做：** 纯粹为了**止损**。它认为“虽然这个动作错得离谱，但我们纠错时也要温柔一点，不能把整个神经网络的权重给震碎了”。

------

## 3. ASPO Soft Clipping (软裁剪)

**这是 ASPO 针对其特有的“反转权重（AIS）”机制提出的高级保护方案。**

- **原来怎么做： ASPO 的 AIS 逻辑会让某些正样本（$A > 0$）的比例 $r$ 变得非常大。如果按照 PPO 原生逻辑，这些样本会被直接 Mask 掉（梯度清零）。**

  **这里就必须做改进了，因为没取倒数之前，r非常大意味着模型学的非常好了，直接截断不影响，因为模型学的很好了。但是当取倒数之后，意味着模型对于该A大于0的策略学的还不好，尽管测试r很大，也不能直接截断。**

- **存在什么问题： 这些 $r$ 极大的正样本其实是“被忽视的重点”。如果直接 Mask 掉（像 PPO 那样），模型就学不到这些关键信息。**

- **怎么改进： 使用 Soft Clip。限制权重的数值大小（防止爆炸），但不让梯度消失（保留微小的导数）。**

- **为什么这么做： 为了“补课”。它认为：“你这个样本很重要，虽然你的数值现在很危险，但我不想让你‘停学’。我把你说话的分贝调小一点（限制数值），但你还得继续发言（保留梯度）。”**

------

### 终极对比总结表

| **机制**      | **解决的痛点**    | **裁剪对象**                    | **梯度后果**        | **核心哲学**                 |
| ------------- | ----------------- | ------------------------------- | ------------------- | ---------------------------- |
| **PPO 原生**  | 过拟合 & 惩罚过度 | $A > 0$ 的上限   $A < 0$ 的下限 | 梯度置 0 (Mask)     | 适可而止，差不多就行。       |
| **Dual Clip** | 负样本导致的崩盘  | **$A < 0$ 的上限**              | 强制截断 (Clamp)    | 止损第一，别把模型练废。     |
| **ASPO Soft** | 正样本的学习不足  | **$A > 0$ 的上限**              | **保留梯度 (Soft)** | **勤能补拙，掉队了也要练。** |

------

