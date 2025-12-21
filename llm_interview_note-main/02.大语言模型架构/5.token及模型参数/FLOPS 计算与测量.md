# FLOPS 计算与测量

要理解**FLOPS（每秒浮点运算次数，Floating-point Operations Per Second）** 的计算，首先需要明确两个核心概念：



1. **FLOP（大写，无后缀）**：指**单次浮点运算**（如 1 次加法 `a+b` 或 1 次乘法 `a×b`，1 次 “乘加操作” `a×b+c` 算 2 次 FLOP：1 次乘法 + 1 次加法）。

2. **FLOPS（小写 s，带后缀）**：指**每秒能执行的 FLOP 总数**，是衡量硬件计算能力（如 GPU/CPU）或模型计算量的核心指标（注：有时也用 “FLOPs” 泛指模型总计算量，需结合语境判断，本文统一区分：FLOP = 单次运算，FLOPS = 每秒运算数，模型总计算量 = 总 FLOP）。

计算 FLOPS 的核心逻辑是：**先估算任务的 “总 FLOP（即完成任务需多少浮点运算）”，再除以 “完成任务的时间（秒）”**，即：

`FLOPS = 总FLOP / 耗时（s）`

实际场景中，我们更常先计算**模型的 “总 FLOP”**（即训练 / 推理一次需要多少浮点运算），再结合硬件实际耗时反推 FLOPS，或用工具直接测量。下面分「基础案例」和「深度学习模型（全连接 / 卷积 / Transformer）」详细讲解计算方法。

## 一、基础概念：FLOP 的计数规则

先明确 “哪些操作算浮点运算”，这是计算的基础：



* 1 次**加法**（如 `3.14 + 2.71`）→ 1 FLOP

* 1 次**乘法**（如 `3.14 × 2.71`）→ 1 FLOP

* 1 次**乘加（MAC，Multiply-Accumulate）**（如 `y = y + a×b`）→ 2 FLOP（1 次乘法 + 1 次加法）

* 复杂运算（如 `sin(x)`、`softmax`）：拆解为基础浮点运算（如 `softmax` 含指数、求和、除法，约 `n` 次 FLOP，`n` 为特征维度），但工程上常简化估算。

## 二、简单案例：手动计算总 FLOP

先从数学运算入手，理解 “总 FLOP” 的计算逻辑，再过渡到深度学习模型。

### 案例 1：向量点积（如 `y = x·w = x₁w₁ + x₂w₂ + ... + x_d w_d`）

假设输入向量 `x` 维度为 `d`（长度 `d`），权重向量 `w` 维度也为 `d`：



* 计算过程：`d` 次乘法（`x₁w₁, x₂w₂, ..., x_d w_d`） + `(d-1)` 次加法（将 `d` 个乘积相加）

* 总 FLOP = `d（乘法） + (d-1)（加法） = 2d - 1`

* 简化估算：当 `d` 较大时，`-1` 可忽略，总 FLOP ≈ `2d`（或直接近似为 `d`，取决于是否简化乘加操作）。

### 案例 2：矩阵乘法（如 `C = A × B`，`A` 为 `m×k`，`B` 为 `k×n`，`C` 为 `m×n`）

矩阵乘法是深度学习的核心组件（如全连接层、注意力层），计算逻辑如下：



* 每个输出元素 `C_ij = A_i₁B₁j + A_i₂B₂j + ... + A_ik B_kj`（即 `A` 的第 `i` 行与 `B` 的第 `j` 列的点积）；

* 单个 `C_ij` 的 FLOP：`k` 次乘法 + `(k-1)` 次加法 = `2k - 1`；

* 总输出元素数：`m×n`（`C` 的维度）；

* 总 FLOP = `m×n×(2k - 1)` ≈ `2mkn`（`k` 较大时，`-1` 可忽略）。

**例子**：`A` 是 `2×3` 矩阵，`B` 是 `3×4` 矩阵，总 FLOP ≈ `2×2×3×4 = 48`（验证：`m=2, k=3, n=4`，`2mkn=2×2×3×4=48`）。

## 三、深度学习模型：核心层的总 FLOP 计算

深度学习模型的总 FLOP 是各层 FLOP 之和，下面讲解最常见的「全连接层」「卷积层」「Transformer 层」的计算方法（均为**推理阶段**的估算，训练阶段需乘以 2，因反向传播的计算量约等于正向传播）。

### 1. 全连接层（Fully Connected Layer）

全连接层是 “输入向量→权重矩阵→输出向量” 的映射，公式为 `y = Wx + b`，其中：



* `x`：输入向量，维度 `d_in`（特征数）；

* `W`：权重矩阵，维度 `d_out × d_in`（`d_out` 为输出特征数）；

* `b`：偏置向量，维度 `d_out`。

#### 计算逻辑：



1. **矩阵乘法&#x20;**`Wx`：`W` 是 `d_out×d_in`，`x` 是 `d_in×1`，输出 `d_out×1`，总 FLOP ≈ `2×d_in×d_out`（参考矩阵乘法公式：`m=d_out, k=d_in, n=1`，`2mkn=2×d_out×d_in×1`）；

2. **加偏置&#x20;**`+b`：`d_out` 次加法，总 FLOP = `d_out`；

3. **总 FLOP（全连接层）**：`2×d_in×d_out + d_out ≈ 2×d_in×d_out`（`d_in` 较大时，`d_out` 可忽略）。

**例子**：输入 `d_in=512`，输出 `d_out=1024`，全连接层总 FLOP ≈ `2×512×1024 ≈ 1.048e6`（约 100 万 FLOP）。

### 2. 卷积层（Convolutional Layer，CNN 核心）

卷积层的计算更复杂，需结合输入特征图、卷积核、步幅、 padding 等参数，公式为：

`总FLOP ≈ 2 × C_in × C_out × K × K × H_out × W_out`

其中各参数定义：



* `C_in`：输入特征图的通道数（如 RGB 图像 `C_in=3`）；

* `C_out`：输出特征图的通道数（即卷积核数量）；

* `K`：卷积核的尺寸（如 `3×3` 卷积核，`K=3`）；

* `H_out / W_out`：输出特征图的高度 / 宽度（需根据输入尺寸 `H_in/W_in`、步幅 `s`、padding `p` 计算：`H_out = (H_in - K + 2p)/s + 1`）；

* 系数 `2`：源于 “每个卷积窗口的乘加操作”（`K×K` 次乘法 + `K×K-1` 次加法 ≈ `2K²` 次 FLOP）。

#### 推导逻辑：



1. 单个卷积核（`K×K×C_in`）与输入特征图卷积：每个输出通道的每个像素，需计算 `C_in×K×K` 次乘加 → `2×C_in×K×K` 次 FLOP；

2. 输出特征图有 `C_out` 个通道，每个通道尺寸 `H_out×W_out` → 总 FLOP = `C_out × H_out × W_out × 2×C_in×K×K`。

**例子**：输入特征图 `3×224×224`（`C_in=3, H_in=224, W_in=224`），卷积核 `3×3×3×64`（`K=3, C_out=64`），步幅 `s=1`，padding `p=1`：



* 输出尺寸：`H_out = (224 - 3 + 2×1)/1 + 1 = 224`，`W_out=224`；

* 总 FLOP ≈ `2×3×64×3×3×224×224 ≈ 1.76e9`（约 17.6 亿 FLOP）。

### 3. Transformer 层（大模型核心）

Transformer 层的计算量主要来自「多头自注意力（Multi-Head Attention）」和「前馈网络（Feed-Forward Network, FFN）」，总 FLOP 是两者之和。

#### （1）多头自注意力（MHA）

假设：序列长度 `n`，模型维度 `d_model`，头数 `h`，每个头的维度 `d_k = d_v = d_model/h`（通常 `d_model` 能被 `h` 整除）。

MHA 的总 FLOP ≈ `4×n²×d_model`（简化估算，详细拆解如下）：



1. **Q/K/V 线性投影**：3 个全连接层（输入 `d_model`，输出 `d_model`），总 FLOP ≈ `3×2×d_model×d_model = 6d_model²`（可忽略，因 `n²d_model` 远大于 `d_model²`）；

2. **注意力分数计算（Q×K^T）**：`Q`（`n×d_model`）× `K^T`（`d_model×n`）→ 输出 `n×n`，总 FLOP ≈ `2×n²×d_model`；

3. **注意力权重 ×V**：`注意力权重（n×n）` × `V（n×d_model）` → 输出 `n×d_model`，总 FLOP ≈ `2×n²×d_model`；

4. **多头合并（线性层）**：1 个全连接层，总 FLOP ≈ `2×d_model×d_model`（可忽略）。

#### （2）前馈网络（FFN）

FFN 通常是 “升维→激活→降维” 结构，公式为 `FFN(x) = W₂·ReLU(W₁x + b₁) + b₂`，其中 `W₁` 维度 `d_ff×d_model`（`d_ff` 为中间层维度，通常 `d_ff=4d_model`），`W₂` 维度 `d_model×d_ff`。

FFN 的总 FLOP ≈ `2×d_model×d_ff×n`（因序列长度 `n`，每个 token 需独立计算）：



* 升维（`W₁x`）：`2×d_model×d_ff` 次 FLOP per token；

* 降维（`W₂·ReLU(...)`）：`2×d_ff×d_model` 次 FLOP per token；

* 总 FLOP = `n × (2d_model d_ff + 2d_ff d_model) = 4n d_model d_ff` → 若 `d_ff=4d_model`，则 ≈ `16n d_model²`。

#### （3）Transformer 层总 FLOP

总 FLOP = MHA FLOP + FFN FLOP ≈ `4n²d_model + 16n d_model²`（当 `n` 较大时，`4n²d_model` 占主导，如 `n=1024, d_model=512`，`n²d_model` 远大于 `n d_model²`）。

**例子**：`n=1024`，`d_model=512`，`h=8`，`d_ff=2048`：



* MHA FLOP ≈ `4×1024²×512 ≈ 2.1e9`；

* FFN FLOP ≈ `4×1024×512×2048 ≈ 4.2e9`；

* Transformer 层总 FLOP ≈ `6.3e9`（约 63 亿 FLOP）。

## 四、如何实际测量 FLOPS？

手动计算仅用于 “估算”，实际场景中需用工具直接测量硬件的 FLOPS（或模型的总 FLOP），常用工具如下：

### 1. 测量模型总 FLOP（无需运行，静态估算）



* **PyTorch**：用 `thop`（Pytorch-OpCounter）库，一行代码估算：



```
from thop import profile

import torch

from torchvision.models import resnet50

model = resnet50()

input = torch.randn(1, 3, 224, 224)  # (batch\_size, C\_in, H, W)

flops, params = profile(model, inputs=(input,))

print(f"模型总FLOP：{flops/1e9:.2f} G FLOP")  # 输出约4.1亿FLOP（resnet50推理）
```



* **TensorFlow**：用 `tf.profiler` 或 `keras-flops` 库。

### 2. 测量硬件实际 FLOPS（需运行任务，动态测量）



* **PyTorch**：结合 `time` 库，计算 “总 FLOP / 耗时”：



```
import time

import torch

from thop import profile

model = resnet50().cuda()

input = torch.randn(32, 3, 224, 224).cuda()  # batch\_size=32

flops, \_ = profile(model, inputs=(input,))  # 单batch总FLOP

\# 多次运行取平均，减少误差

model.eval()

with torch.no\_grad():

&#x20;   start = time.time()

&#x20;   for \_ in range(100):

&#x20;       output = model(input)

&#x20;   torch.cuda.synchronize()  # 等待GPU完成计算

&#x20;   end = time.time()

avg\_time = (end - start) / 100  # 单batch平均耗时（s）

flops\_per\_batch = flops  # 单batch总FLOP

flops = flops\_per\_batch / avg\_time  # 硬件实际FLOPS

print(f"GPU实际FLOPS：{flops/1e12:.2f} T FLOPS")  # 如RTX 3090约200-300 TFLOPS（FP32）
```



* **NVIDIA 工具**：用 `nvidia-smi` 查看 GPU 理论 FLOPS（如 RTX 4090 的 FP32 理论 FLOPS 约 83 TFLOPS），实际 FLOPS 通常为理论值的 50%-80%（因内存带宽、并行效率等限制）。

## 五、关键注意事项



1. **精度区分**：FLOPS 分精度（FP32/FP16/FP8），如 “FP16 FLOPS” 指每秒执行的半精度浮点运算次数，数值通常是 FP32 的 2 倍（同硬件下，半精度运算更快）；

2. **训练 vs 推理**：训练阶段的总 FLOP 约为推理阶段的 2 倍（需反向传播计算梯度，梯度计算量≈正向传播）；

3. **简化估算**：手动计算时通常忽略偏置、激活函数（如 ReLU、softmax）的 FLOP（占比 < 5%），工具测量会包含这些细节；

4. **Batch Size 影响**：总 FLOP 与 batch size 成正比（如 batch size=32 的总 FLOP 是 batch size=1 的 32 倍），但 FLOPS（每秒运算数）会随 batch size 增大而提升（硬件并行效率更高）。

总结：FLOPS 的计算核心是 “先算总 FLOP（任务需多少浮点运算），再除以耗时”。深度学习场景中，需根据层类型（全连接 / 卷积 / 注意力）按公式估算总 FLOP，实际应用中优先用工具（如 thop、nvidia-smi）测量，避免手动计算误差。

> （注：文档部分内容可能由 AI 生成）