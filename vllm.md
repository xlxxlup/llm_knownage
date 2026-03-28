# VLLM

[(1 封私信 / 24 条消息) 大模型推理框架vLLM原理详解！ - 知乎](https://zhuanlan.zhihu.com/p/1953390686150844488)

vLLM 引入了 **PagedAttention、Prefix Sharing、Continuous Batching** 和 **KV Cache 淘汰策略**，形成高性能推理方案：

- **PagedAttention**：将 KV Cache 分页管理，按需分配逻辑页，解决内部与外部显存碎片问题。
- **Prefix Sharing**：相同前缀请求共享 KV Cache，节省显存并减少重复计算，分叉时通过 Copy-on-Write 保证正确性。
- **Continuous Batching**：动态批处理机制，让新请求随时加入正在执行的 batch，提高 GPU 利用率 并 降低请求延迟。
- **KV Cache 淘汰策略**：显存不足时先换到 CPU 内存，内存不足时再淘汰，必要时重新计算。