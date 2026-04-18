## peft

![image-20260107001533962](./image/image-20260107001533962.png)

#### Prompt-Tuning

#### 核心位置：原始输入 embedding 后拼接可训练 prompt

**标准 Prompt-Tuning 是在原始输入经过 embedding 层得到词嵌入后，在前面（或其他位置）拼接一段可训练的软提示向量 (prompt embedding)，而非在 embedding 模块前加参数**。

![6241067a-d2ea-4360-b31d-3e95b995a41b](./image/6241067a-d2ea-4360-b31d-3e95b995a41b.png)

**Hard prompt: 人写的**

**Soft prompt: 随机初始化**

#### P-Tuning

#### 和普通 Prompt-Tuning 的关键差别

- **Prompt-Tuning**：

  直接初始化一组 `[prompt_len, hidden_dim]` 的可训练向量，**直接训练这组裸向量**。

  

- **P-Tuning (v1)**：

  不直接训向量，而是加一个**超轻量的小编码器**（常用 LSTM，也可以是 MLP），

  由这个小网络输出 prompt 向量，**只训练这个小 encoder + 少量参数**。

![image-20260107002507358](./image/image-20260107002507358.png)

#### Prefix-Tuning

![image-20260107002729822](./image/image-20260107002729822.png)

![image-20260107003303086](./image/image-20260107003303086.png)

![image-20260313232009120](./image/image-20260313232009120.png)