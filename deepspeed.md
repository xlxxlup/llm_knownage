## 分布式训练

1、加快训练速度

2、模型太大，单张卡无法训练

### DP

<img src="./image/image-20260112160618074.png" alt="image-20260112160618074" style="zoom:50%;" />

<img src="./image/image-20260112160637787.png" alt="image-20260112160637787" style="zoom:50%;" />

### DDP

<img src="./image/image-20260112160827546.png" alt="image-20260112160827546" style="zoom:50%;" />

<img src="./image/image-20260112160837464.png" alt="image-20260112160837464" style="zoom:50%;" />

<img src="./image/image-20260112160853594.png" alt="image-20260112160853594" style="zoom:50%;" />

<img src="./image/image-20260112160914401.png" alt="image-20260112160914401" style="zoom:50%;" />

<img src="./image/image-20260112161124294.png" alt="image-20260112161124294" style="zoom:50%;" />

<img src="./image/image-20260112161209386.png" alt="image-20260112161209386" style="zoom:50%;" />

<img src="./image/image-20260112161237251.png" alt="image-20260112161237251" style="zoom:50%;" />

<img src="./image/image-20260112161321579.png" alt="image-20260112161321579" style="zoom:50%;" />

**不同卡数据不一样，所以得到的梯度也不一样，使用Ring-AllReduce进行同步，同步梯度这些信息，保持网络参数一致**

**当一个桶里的参数都计算好之后，就进行同步**

<img src="./image/image-20260112162215125.png" alt="image-20260112162215125" style="zoom:50%;" />

**每个卡上的网络参数梯度就一样了**

<img src="./image/image-20260112162251698.png" alt="image-20260112162251698" style="zoom:50%;" />

<img src="./image/image-20260112162346884.png" alt="image-20260112162346884" style="zoom:50%;" />

### DeepSpeed 

【动画理解Pytorch 大模型分布式训练技术 DP，DDP，DeepSpeed ZeRO技术】https://www.bilibili.com/video/BV1mm42137X8?vd_source=c5c396652c0c83be15efe54e0c348c90

# Deepspeed Zero 各阶段策略核心差异对比

### 第一张图：常规分布式数据并行训练

<img src="./image/image-20260112223128416.png" alt="image-20260112223128416" style="zoom:50%;" />

这是**标准数据并行**的内存分布：

- 每个GPU（GPU:0/1/2）都保存**完整的模型参数（FP16）、完整的梯度（FP16）、完整的优化器状态（FP32梯度、一阶动量、二阶动量、FP32参数）**。
- 每个GPU的内存占用较高（蓝色大块区域是重复存储的优化器状态/参数），模型规模受单GPU内存限制较大。

### 第二张图：Deepspeed-Zero1（优化器状态分片）

<img src="./image/image-20260112223155572.png" alt="image-20260112223155572" style="zoom:50%;" />

这是**Zero1策略**的内存分布：

- 核心是**优化器状态（FP32梯度、动量、FP32参数）分片存储**：不再每个GPU保存完整的优化器状态，而是将其拆分后分配到不同GPU（比如GPU:0的蓝色大块减少，GPU:1/2各自承担一部分优化器状态）。
- FP16参数、FP16梯度仍在每个GPU保存完整副本，内存占用比常规并行降低（优化器状态不再重复存储）。

### 第三张图：Deepspeed-Zero2（优化器状态+梯度分片）

<img src="./image/image-20260112223223048.png" alt="image-20260112223223048" style="zoom:50%;" />

这是**Zero2策略**的内存分布：

- 在Zero1基础上，新增**FP16梯度分片存储**：不仅优化器状态分片，梯度也拆分到不同GPU（GPU:0的FP16梯度区域缩小，GPU:1/2各自承担部分梯度）。
- 仅FP16参数仍在每个GPU保存完整副本，内存占用比Zero1进一步降低（梯度也不再重复存储）。

### 第四张图：Deepspeed-Zero3（参数+梯度+优化器状态全分片）

<img src="./image/image-20260112223242862.png" alt="image-20260112223242862" style="zoom:50%;" />

这是**Zero3策略**的内存分布：

- 实现**参数（FP16）、梯度（FP16）、优化器状态全部分片存储**：每个GPU仅保存模型的**一部分参数、一部分梯度、一部分优化器状态**（GPU:0的FP16参数区域大幅缩小，GPU:1/2仅保留对应分片）。
- 内存占用是Zero系列中最低的，可支持训练远超单GPU内存容量的超大模型（每个GPU仅承担模型的“碎片”数据）。

要不要我帮你整理一份**Deepspeed Zero各阶段策略的核心差异对比表**？

![image-20260112222457583](./image/image-20260112222457583.png)