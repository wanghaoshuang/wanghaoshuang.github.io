---
layout: post
title: Log uniform distribution sampler
date: 2017-11-27 15:32:24.000000000 +09:00
---


> 转载请注明出处：http://wanghaoshuang.github.io/2017/11/Log-uniform-distribution-sampler

### 1. Log uniform distribution in [0, range)

我们希望在[0, range）范围内抽样整数，并且我们假设我们要抽的整数符合长尾分布。
假设离散随机变量\\(X\\), 其概率分布为：
$$P_X(x) = \theta ln(1+\frac{1}{x+1})\tag{1}$$


为了得到\\(X\\), 我们再引入连续性随机变量\\(X'\\), 设\\(X'\\)的概率密度函数为：
$$f_{X'}(x) $$
为了借助\\(X'\\)得到\\(X\\), 令：
$$P_X(x) = \int_x^{x+1}f_{X'}(x) \tag{2}$$ 

等式\\((2)\\)的意思就是，我们对连续随机变量\\(X'\\)进行抽样得到实数R, 然后对R取下界，得到符合离散分布\\(X\\)的N.

接下来求\\(X'\\), 由等式\\((1)\\)和\\((2)\\)得：
$$f_{X'}(x) = \frac\theta{x+1}\tag{3}$$
\\(f_{X'}(x)\\)需要满足：
$$\int_0^{range}f_{X'}(x) = 1\tag{4}$$

由等式\\((1)\\)和\\((2)\\)得：
$$\theta  = \frac1{ln(range + 1)}\tag{5}$$


### 2. Inverse transform sampling
经过上节分析，我们将问题转换为求连续性随机变量\\(X'\\)，而且我们也知道了\\(f_{X'}(x)\\).
本节介绍如何通过[Inverse transform sampling](https://en.wikipedia.org/wiki/Inverse_transform_sampling)方法从uniform distribution得到\\(X'\\)。
设\\(X'\\)的概率分布函数为\\(F_{X'}(x)\\):
$$F_{X'}(x) = \int_{-\infty}^xf_{X'}(x) = \theta ln(x+1) \tag{6}$$

And
$$g(x) = F^{-1}_{X'}(x) = e^\frac x\theta -1 \tag{7}$$

因为：
$$\lim_{x\to0}g(x) = 0 \tag{8}$$
$$\lim_{x\to1}g(x)  =  e^\frac 1\theta -1 = range \tag{9}$$
对于等式\\((7)(8)(9)\\), 如果\\(x\\)服从[0, 1)的uniform ditribution, 则\\(g(x)\\)服从在[0, range)的分布\\(F_{X'}(x)\\)

### 3. Implement log uniform in c++

```
float log_r = log(range);
std::random_device r;
std::mt19937 random_engine(r());
std::uniform_real_distribution<> dist(0, 1);
int64 result = static_cast<int64>(exp(dist(random_engine) * log_r) - 1)
result = value % range_;
```
