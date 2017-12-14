---
layout: post
title: 梯度量化优化多机同步SGD
date: 2017-12-13 17:57:24.000000000 +09:00
---

# 梯度量化更新与paddle pserver
多机训练的优化要求pserver支持更灵活的功能，梯度量化更新就是一个典型的例子。本文档内容包括：

- 多机训练面临的问题
- 什么是梯度量化
- 为什么梯度量化可以几乎无精度损失
- 梯度量化带来的收益
- paddle pserver存在的问题及建议

## 1. 问题
对于数据并行的分布式同步SGD训练，我们会把每一个mini-batch分成多份，分别交给trainer做训练。
假设\\(z^{(i)}\\)是第i个trainer的输入，根据\\(z^{(i)}\\)我们计算出梯度\\(g^{(i)}\\). 
pserver收集所有trainer的梯度，并取平均值，如下：

$$\overline g = \frac 1{N} \sum_{i}^N g^{(i)}$$

然后，各个trainer从pserver上拉取\\(\overline g\\)，并更新当前trainer的parameter.
在多机训练过程中，机器之前的通信是非常大的瓶颈，主要体现在pserver和trainer之间float类型梯度的传递。如果我们可以对float类型的梯度信息进行转换压缩，可以一定程度上优化整体训练性能。

## 2. 梯度量化
trainer在把梯度交给pserver之间，为了减少传输代价，先将梯度其从float类型量化为三元组{-1， 0， 1}，具体量化公式如下：

$$\widetilde {g_t^k} = quantize(g_t^k) = s_t^k * sign(g_t^k) \circ b_t^k  \tag{1}$$

其中：

$$s_t^k \triangleq \max (abs(g_t^k)) \tag{2}$$

\\(g_t^k\\) 为第t个batch的第k个layer(所有trainer)算出来的梯度;
\\(sign(g_t^k)\\)和\\(abs(g_t^k)\\)分别取\\(g_t^k\\)的符号和绝对值。
\\(\circ\\)符号是Hadamard product运算。
\\(b_t^k\\)是一个随机的二进制向量，它的每一个元素服从伯努利分布：

$$P(b_t^k[j] =1 | b_t^k) = \frac {|(g_t^k[j] |} {s_t^k} \tag{3}$$

$$P(b_t^k[j] =0 | b_t^k) = 1 - \frac {|(g_t^k[j] |} {s_t^k} \tag{4}$$


其中，\\(b_t^k[j]\\)和\\(g_t^k[j]\\)是分别是\\(b_t^k\\)和\\(g_t^k\\)的第j个元素.

通过等式(1)得到量化后的梯度\\(\widetilde {g_t^k}\\)，是一个元素取值范围为 {−1, 0, +1}的向量;

## 3. 分析

假设\\(z\\)是一个batch data, \\(w\\)是模型的parameter, loss function为\\(Q(z, w)\\)，那么我们的优化目标就是最小化：

$$C(w)  \triangleq E(Q(z, w)) \tag{5}$$

我们一般按以下方式更新梯度：

$$w_{t+1} = w_t - \eta_t X \tag{6}$$

对于普通方法：

$$X = g_t = \bigtriangledown _wQ(z_t, w_t) \tag{7}$$

对于量化方法：

$$X = \widetilde g_t = s_t * sign(g_t) \circ b_t \tag{8}$$

根据等式(3)(4)有：

$$E(\widetilde g_t) = E(s_t * sign(g_t) \circ b_t) \\ 
= E(s_t * sign(g_t) \circ E(b_t|z_t)) \\
= E(g_t)\\
= E(\bigtriangledown _wQ(z_t, w_t))\\
= \bigtriangledown _wE(Q(z_t, w_t))\\
=  \bigtriangledown _w C(w)$$

所以我们量化后梯度的期望就等于我们要优化目标函数的对w的微分。

## 4. 优化效果

通过等式(1)得到量化后的梯度\\(\widetilde {g_t^k}\\)，是一个元素取值范围为 {−1, 0, +1}的向量，其中每个元素对应原来的一个float类型的梯度。
也就是我们可以用两个bit来编码表示一个梯度了，比如`00`表示-1， `01`表示0， `10`表示1，从4个字节缩减到2个bit，**网络传输数据量减少为原来的1/16**。

## 5. paddle pserver的限制
如上节分析，我们可以将4个梯度的量化结果编码到一个uint8_t中，然后pserver收集并计算以uint8_t类型存储的梯度信息。
而且，在量化各个trainer的梯度前，我们还要根据等式(2)收集所有trainer上的绝对值最大梯度。
但是，当前parameter server只能收集trainer的weight和gradients。
在设计实现新版pserver时，我们应该考虑到上述需求，比如实现如下功能：

```
pclient.init_parameter(key="max_abs_grad", shape=[1], type="float32") // 初始化参数
pclient.push_parameter(key="max_abs_grad", value=0.5) // 传本地parameter到pserver
pclient.pull_parameter(key="max_abs_grad", reduce_method="max") // 拉取reduce之后的parameters

pclient.init_parameter(key="op_id_grads", shape=[2,2], type="uint8")
pclient.push_parameter(key="op_id_grad", value=[[1,5],[8,2]])
pclient.pull_parameter(key="op_id_grad", reduce_method="decode_average")
```

> 转载请注明出处
