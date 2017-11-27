---
layout: post
title: Noise-contrastive estimation
date: 2017-11-27 15:32:24.000000000 +09:00
---


> 转载请注明出处：http://wanghaoshuang.github.io/2017/11/Noise-contrastive-estimation

### 0. 参考


 [Noise-contrastive estimation: A new estimation principle for unnormalized statistical models](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf)

### 1. 定义
> Denote by \\(X = (x_1,...,x_T)\\) the observed data set, consisting of T observations of the data x, and by \\(Y = (y_1 , . . . , y_T )\\) an artificially generated data set of noise y with distribution \\(p_n(.)\\). X is modeled with \\($p_m(.; θ)\\).

翻译：
 定义\\(X = (x_1,...,x_T)\\) 为我们观察到的样本集合,也就是训练数据集中的T条样本。
 \\(p_m(.; θ)\\) 表示$X$的分布，其中\\(θ\\)为我们要求的模型的参数。 
  
  定义 \\(Y = (y_1 , . . . , y_T )\\)是人工生成的样本集合，也就是通过负类别采样生成的数据集。
  \\(p_n(.)\\) 表示\\(Y\\)的分布 (一般用uniform distribution) .

>Denote by \\(U = (u_1 , . . . , u_{2T} )\\) the union of the two sets \\(X\\) and \\(Y\\) , and assign to each datapoint \\(u_t\\) a binary class label \\(C_t\\): \\(C_t =1\\) if \\(u_t ∈X\\) and \\(C_t = 0\\) if \\(u_t ∈ Y\\) . 

翻译：
定义 \\(U = (u_1 , . . . , u_{2T} )\\) 为 \\(X\\) 和 \\(Y\\)的并集 , 对于每一个样本 \\(u_t\\)定义一个二分类 label \\(C_t\\):
 \\(C_t =1\\) if \\(u_t ∈X\\) （真正的样本）
 \\(C_t = 0\\) if \\(u_t ∈ Y\\) （通过sampling生成的样本）

### 2. 公式

对于真正的样本\\(u\\)（\\(C_t =1\\)）：

$$p(u|C = 1; θ) = p_m(u; θ)$$

其中\\(θ\\)就是我们正在求解的模型，也就是整个NCE OP 的 FC部分，FC的输出就是\\(p_m(u; θ)\\), **所以FC的输出应该过sigmoid activation.**

对于生成的噪声样本u ( \\(C_t =1\\)）:

$$p(u|C = 0) = p_n(u)$$

如果我们用uniform distribute,  \\(p_n(u)\\) = number_sampled_class / number_total_class

又：

$$ P(C=1)=P(C=0)=1/2$$

根据贝叶斯原理得：

$$P(C=1|u;\theta) = \frac{p_m(u;\theta)}{p_m(u;\theta) + p_n(u)}$$

$$P(C=0|u;\theta) = 1 - P(C=1|u;\theta) =  \frac{p_n(u)}{p_m(u;\theta) + p_n(u)}$$

Log-likelihood of the parameters θ:

$$L(\theta) = \sum_tC_tlnP(C_t = 1|u_t;\theta) + (1 - C_t)lnP(C_t = 0|u_t;\theta)$$

$$=\sum_tC_tln\frac{p_m(u;\theta)}{p_m(u;\theta) + p_n(u)} + (1-C_t)ln \frac{p_n(u)}{p_m(u;\theta) + p_n(u)}$$

$$=\sum_tC_tln\frac{p_m(u;\theta)}{p_m(u;\theta) + p_n(u)} + \sum_t(1-C_t)ln \frac{p_n(u)}{p_m(u;\theta) + p_n(u)}$$



再来看paddle中的实现：

```
cost = samples_[i].target ? -log(o / (o + b)) : -log(b / (o + b)); // 计算单条样本cost
// samples_[i].target 表示是否是真是样本，也就是(C_t==1)
// o = sigmoid(fc.output) = p_m(u; \theta)
// b = 1. / numClasses_ * config_.num_neg_samples() = p_n(u)
```

### 3. 理论证明

### 4. 其它 
NCE方法也存在一个弊端，由于每个正样本词语w对应的噪声样本各不相同，这些负样本就无法以稠密矩阵的形式存储，也就无法快速地计算稠密矩阵的乘法，这就削弱了GPU的优势。Jozefowicz et al. (2016) 和 Zoph et al. (2016) 各自分别提出了在同一个批数据中共享噪声样本的想法，于是可以发挥GPU的稠密矩阵快速运算的优势。
