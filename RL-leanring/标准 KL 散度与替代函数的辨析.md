# 标准 KL 散度与替代函数的辨析

### 一、先明确：标准KL散度的定义

KL散度（相对熵）用于衡量两个概率分布  $P$  和  $Q$  的差异，**标准定义**是：

 $D_{KL}(P \parallel Q) = \mathbb{E}_{x \sim P} \left[ \log \frac{P(x)}{Q(x)} \right]$ 

- 离散形式： $D_{KL}(P \parallel Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$ 

- 连续形式： $D_{KL}(P \parallel Q) = \int P(x) \log \frac{P(x)}{Q(x)} dx$ 

核心性质：非负（ $D_{KL} \geq 0$ ），当且仅当  $P=Q$  时取0；**不满足对称性**（ $D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P)$ ）。

---

### 二、你图里的KL散度公式为什么“长这样”？

你图里的公式是：

 $D_{KL}\left(\pi_\theta \parallel \pi_{ref}\right) = \frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - \log \frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - 1$ 

令  $r = \frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)}$ ，公式简化为：

 $f(r) = r - \log r - 1$ 

#### 1. 它和标准KL散度的关系

- **形式完全不同**：标准KL是**期望形式**（求和/积分），而你图里是**逐点函数形式**，没有期望/求和符号。

- **性质巧合相似**：

    - 定义域： $r > 0$ （概率值恒正）

    - 最小值：当  $r=1$ （即  $\pi_{ref} = \pi_\theta$ ）时， $f(1)=0$ ，和KL散度“分布相同时为0”的性质一致。

    - 凸性：二阶导数  $f''(r) = \frac{1}{r^2} > 0$ ，是凸函数，和KL散度的凸性一致。

#### 2. 出现这种写法的可能原因

##### （1）**笔误/排版错误**

这是最可能的原因：

- 标准KL散度  $D_{KL}(\pi_\theta \parallel \pi_{ref})$  应该是：

 $D_{KL}(\pi_\theta \parallel \pi_{ref}) = \mathbb{E}_{o_i \sim \pi_\theta(\cdot|q)} \left[ \log \frac{\pi_\theta(o_i|q)}{\pi_{ref}(o_i|q)} \right]$ 

- 你图里的公式遗漏了**期望/求和符号**，并且把  $\log \frac{\pi_\theta}{\pi_{ref}}$  错写成了  $\frac{\pi_{ref}}{\pi_\theta} - \log \frac{\pi_{ref}}{\pi_\theta} - 1$ ，属于形式上的书写错误。

##### （2）**特定文献的替代函数/近似**

在一些优化场景（比如PPO自适应KL惩罚、TRPO约束），研究者会用**凸替代函数**来近似KL散度：

- 函数  $f(r) = r - \log r - 1$  是凸函数，且在  $r=1$  处取最小值0，可作为KL散度的**近似惩罚项**，用来约束  $\pi_\theta$  和  $\pi_{ref}$  的距离。

- 但它**不是严格意义上的KL散度**，只是优化时的替代目标。

##### （3）**符号颠倒**

如果把  $D_{KL}(\pi_{ref} \parallel \pi_\theta)$  写成  $D_{KL}(\pi_\theta \parallel \pi_{ref})$ ，也会导致形式混淆，但即使颠倒，标准KL也不会变成图里的样子。

---

### 三、结合你公式(1)的上下文（PPO类目标）

公式(1)是典型的**PPO（近端策略优化）目标函数**：

 $\frac{1}{G}\sum_{i=1}^G \left( \min\left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \ \text{clip}\left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon \right) A_i \right) - \beta D_{KL}(\pi_\theta \parallel \pi_{ref}) \right)$ 

- 这里的  $D_{KL}$  本意是**约束新策略 ** $\pi_\theta$  ** 和参考策略 ** $\pi_{ref}$  ** 的距离**，防止策略更新过大。

- 原始PPO论文里，KL散度用的是**标准定义**，你图里的公式(2)大概率是**笔误**或**自定义近似**，不是通用的KL散度写法。

---

### 四、结论

1. 你图里的KL散度公式**不符合标准定义**，是一个**逐点凸函数形式**（ $r - \log r - 1$ ），而非期望形式。

2. 它可能是：

    - 笔误/排版错误（遗漏了期望符号，写错了表达式）

    - 特定文献中用来近似KL散度的**替代惩罚函数**

3. 严格来说，**真正的KL散度必须是期望形式**，你图里的写法是不标准的。

---

要不要我帮你推导一下「标准KL散度」和「你图里这个替代函数」之间的数学关系，看看它是不是KL散度的上界/下界？这样能更清楚它的近似意义。
> （注：文档部分内容可能由 AI 生成）