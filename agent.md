# MCP

MCP Server:MCP服务器，里面就是包含一些执行特殊任务的工具

![image-20251220134638824](./image/image-20251220134638824.png)

一般python写的MCP Server 使用uvx启动，node写的MCP Server 使用npx启动

![image-20251220141501950](./image/image-20251220141501950.png)

**cline 和 weather的交互过程**

**输入：cline给weather服务器的消息**

**输出：weather给cline的消息**

**以下内容是cline连接mcp服务器时交互过程**

![image-20251220160712985](./image/image-20251220160712985.png)

**cline 进行工具调用与MCP Server的交互内容**

![image-20251220160917697](./image/image-20251220160917697.png)

#### **cline 与 模型的交互过程**

**cline向大模型发起请求时，prompt会携带很多信息**

![image-20251220170651324](./image/image-20251220170651324.png)

![image-20251220170726661](./image/image-20251220170726661.png)

**模型每一步都会先进行思考<thinking>，再做决策<use_mcp_tool>，cline再通过MCP Server去调用工具，cline再将工具结果返回给模型(observation)**

![image-20251220170225458](./image/image-20251220170225458.png)

**<attempt_completion>:模型认为任务已完成，最终返回给cline的信息。**

## A2A

![image-20251220180626594](./image/image-20251220180626594.png)

A2A作用在agent之间

![image-20251220172433302](./image/image-20251220172433302.png)

**MCP作用在大模型和MCP之间**

![image-20251220172634420](./image/image-20251220172634420.png)



A2A主要流程 

agent注册阶段

![image-20251220175913476](./image/image-20251220175913476.png)

agent card

![image-20251220180210107](./image/image-20251220180210107.png)



![image-20251220180230931](./image/image-20251220180230931.png)

![image-20251220180246869](./image/image-20251220180246869.png)

![image-20251220180448394](./image/image-20251220180448394.png)

agent问答阶段

![image-20251220175834021](./image/image-20251220175834021.png)





## skills

![image-20260401144037759](./image/image-20260401144037759.png)

![image-20260401144152127](./image/image-20260401144152127.png)

![image-20260401140816125](./image/image-20260401140816125.png)

![image-20260401144652554](./image/image-20260401144652554.png)

reference：一些其他的文档啥的

script: 一些代码脚本啥的

![image-20260401143830675](./image/image-20260401143830675.png)

## Harness

[Harness Engineering（驾驭工程） | 菜鸟教程](https://www.runoob.com/ai-agent/harness-engineering.html)

![image-20260401145602804](./image/image-20260401145602804.png)

## Mem0框架

![image-20260401214905589](./image/image-20260401214905589.png)