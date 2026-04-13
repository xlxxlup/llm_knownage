# OpenClaw

Gateway 是 OpenClaw 的心脏，负责所有消息的接入、调度和状态管理。

![image-20260410191429480](./image/image-20260410191429480.png)

![image-20260410191626810](./image/image-20260410191626810.png)

![image-20260410191701373](./image/image-20260410191701373.png)

![image-20260410191712334](./image/image-20260410191712334.png)

![image-20260410191755178](./image/image-20260410191755178.png)

```text
<!-- SKILL.md -->
---
name: web_search
description: 在互联网上搜索信息
version: 1.0.0
author: community
---

# 网页搜索技能

## 使用说明

当用户要求搜索信息时，使用此技能。

## 工具

### search

搜索互联网获取最新信息。

参数：
- query: 搜索关键词
- limit: 返回结果数量（默认 5）

## 示例

用户：帮我搜索一下 React 19 的新特性
Agent：使用 search 工具，query = "React 19 new features"
```

![image-20260410191908875](./image/image-20260410191908875.png)

![image-20260410192031397](./image/image-20260410192031397.png)

## 一、会话管理

![image-20260410182241113](./image/image-20260410182241113.png)

![image-20260410182331038](./image/image-20260410182331038.png)

## 二、记忆系统

```text
 短期记忆（Session Memory）
 长期记忆（Long-term Memory）
```

![image-20260410182837024](./image/image-20260410182837024.png)

![image-20260410182948175](./image/image-20260410182948175.png)

**较早期的摘要压缩，特别早期的直接截断**

![image-20260410190345665](./image/image-20260410190345665.png)

## 三、记忆系统

![image-20260410190544196](./image/image-20260410190544196.png)

![image-20260410190636564](./image/image-20260410190636564.png)

![image-20260410190938304](./image/image-20260410190938304.png)