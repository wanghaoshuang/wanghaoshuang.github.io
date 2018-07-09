---
layout: post
title: 【metric learning】Dense triplet loss
date: 2018-7-9 14:44:24.000000000 +09:00
---

## 1. Input定义

对于一个batch_siz为n， embedding特征长度为m的input，记为A:

$$
A = 
\left[
 \begin{matrix}
   x_0^0 & x_0^1 & x_0^2 &\cdots &x_0^m \\
   x_1^0 & x_1^1 & x_1^2 &\cdots &x_1^m \\
   \vdots & \vdots & \ddots & \ddots & \vdots \\
   x_n^0 & x_n^1 & x_n^2 &\cdots &x_n^m 
  \end{matrix}
\right]  
\tag{1}
$$


## 2. 计算距离矩阵

对A进行elementwise square, 得到：

$$
A^2 = 
\left[
 \begin{matrix}
   (x_0^0)^2 & (x_0^1)^2 & (x_0^2)^2 &\cdots & (x_0^m)^2 \\
   (x_1^0)^2 & (x_1^1)^2 & (x_1^2)^2 &\cdots & (x_1^m)^2 \\
   \vdots & \vdots & \vdots & \ddots & \vdots \\
   (x_n^0)^2 & (x_n^1)^2 & (x_n^2)^2 &\cdots & (x_n^m)^2
  \end{matrix}
\right] \tag{2}  
$$

然后对\\(A^2\\)按行做`reduce sum`, 得到：

$$ A_{sum} = 
\left[
 \begin{matrix}
   \sum_{i=0}^m(x_0^i)^2 \\
   \sum_{i=0}^m(x_1^i)^2 \\
   \vdots \\
   \sum_{i=0}^m(x_n^i)^2
  \end{matrix}
\right]  \tag{3}
$$

A乘上A的转置得：

$$ A * A^T =
\left[
 \begin{matrix}
   \sum_{i=0}^m{(x_0^m * x_0^m)} & \sum_{i=0}^m{(x_0^m * x_1^m)} & \cdots & \sum_{i=0}^m{(x_0^m * x_n^m)}\\
   \sum_{i=0}^m{(x_1^m * x_0^m)} & \sum_{i=0}^m{(x_1^m * x_1^m)} & \cdots & \sum_{i=0}^m{(x_1^m * x_n^m)} \\
   \vdots & \vdots & \ddots & \vdots \\
   \sum_{i=0}^m{(x_n^m * x_0^m)} & \sum_{i=0}^m{(x_n^m * x_1^m)} & \cdots & \sum_{i=0}^m{(x_n^m * x_n^m)} 
  \end{matrix}
\right]  \tag{4}
$$

由等式（3）（4）得：


$$
A_{sum} + （A_{sum})^T - 2*  A * A^T  \\ 
= \left[
 \begin{matrix}
   2*\sum_{i=0}^m{(x_0^i * x_0^i)} + \sum_{i=0}^m(x_0^i)^2 + \sum_{i=0}^m(x_0^i)^2 &
    2*\sum_{i=0}^m{(x_0^i * x_1^i)} + \sum_{i=0}^m(x_0^i)^2 + \sum_{i=0}^m(x_1^i)^2 &
    \cdots &
    \sum_{i=0}^m{(x_0^i * x_n^i)} + \sum_{i=0}^m(x_0^i)^2 + \sum_{i=0}^m(x_n^i)^2 &\\
    2*\sum_{i=0}^m{(x_1^i * x_0^i)} + \sum_{i=0}^m(x_1^i)^2 + \sum_{i=0}^m(x_0^i)^2 &
    2*\sum_{i=0}^m{(x_1^i * x_1^i)} + \sum_{i=0}^m(x_1^i)^2 + \sum_{i=0}^m(x_1^i)^2 &
    \cdots &
    \sum_{i=0}^m{(x_1^i * x_n^i)} + \sum_{i=0}^m(x_1^i)^2 + \sum_{i=0}^m(x_n^i)^2 &\\
   \vdots & \vdots & \ddots & \vdots \\
   2*\sum_{i=0}^m{(x_n^i * x_0^i)} + \sum_{i=0}^m(x_n^i)^2 + \sum_{i=0}^m(x_0^i)^2 &
    2*\sum_{i=0}^m{(x_n^i * x_1^i)} + \sum_{i=0}^m(x_n^i)^2 + \sum_{i=0}^m(x_1^i)^2 &
    \cdots &
    \sum_{i=0}^m{(x_n^i * x_n^i)} + \sum_{i=0}^m(x_n^i)^2 + \sum_{i=0}^m(x_n^i)^2  &
  \end{matrix}
\right]  \\
=
\left[
 \begin{matrix}
   \sum_{i=0}^m{(x_0^i - x_0^i)^2} & \sum_{i=0}^m{(x_0^i - x_1^i)^2} & \cdots & \sum_{i=0}^m{(x_0^i - x_n^i)^2} \\
   \sum_{i=0}^m{(x_0^i - x_1^i)^2} & \sum_{i=0}^m{(x_1^i - x_1^i)^2} & \cdots & \sum_{i=0}^m{(x_1^i - x_n^i)^2} \\
   \vdots & \vdots & \ddots & \vdots \\
    \sum_{i=0}^m{(x_n^i - x_0^i)^2} & \sum_{i=0}^m{(x_n^i - x_1^i)^2} & \cdots & \sum_{i=0}^m{(x_n^i - x_n^i)^2} \\
  \end{matrix}
\right]
 \tag{5}
$$

等式（5）的计算结果即为我们要求的距离矩阵， 简单记为\\(D\\)：

$$
D = 
\left[
 \begin{matrix}
  d_{00} & d_{01} & \cdots & d_{0n} \\
   d_{10} & d_{11} & \cdots & d_{1n} \\
   \vdots & \vdots & \ddots & \vdots \\
   d_{n0} & d_{n1} & \cdots & d_{nn} \\
  \end{matrix}
\right]  \tag{6}
$$


## 3. 计算Loss

为了简单说明计算过程，我们假设Input的batch_size=6，而且有三个类别，其中：

- 类别0： \\(x_0, x_1\\)
- 类别1： \\(x_2\\)
- 类别2： \\(x_3, x_4, x_5\\)

其距离矩阵如图1所示, 其中彩色填充部分为positive pair distances, 未填充部分为negtive pair distance:

![图片](http://agroup-bos.su.bcebos.com/2eef74daca2965da86a630b1504c9b4b90cc5d65)

<center>图1</center>

假设我们选取\\(x_0\\)为anchor, \\(x_1\\)为positive, \\(x_2\\)为negtive, 则\\(triplet (x_0, x_1, x_2)\\)的loss为：

$$
\begin{align}
L & =  |d(x_0, x_1) - d(x_0, x_2) + margin|_+ \\
   & = relu(d_{01} - d_{02} + margin)
 \end{align} \tag{7}
 $$

根据图1的第一行，我们可以计算出所有以\\(x_0\\)为anchor的triplet的loss.

首先选出所有与\\(x_0\\)同类的samples与\\(x_0\\)的距离, 记作向量：

$$
D_0^{pos} =  D_0[0:2] = [d_{00}, d_{01}] \tag{8}
$$

然后，选出所有与\\(x_0\\)不同类的samples, 记作：

$$
D_0^{neg} = D_0[2:6] = [d_{02}, d_{03}, d_{04}, d_{05}]  \tag{9}
$$

由等式（8）（9）得：

$$
\begin{align}
(D_0^{pos})^T  - D_0^{neg} &= 
\left[
 \begin{matrix}
  d_{00} & d_{00} & d_{00} & d_{00} \\
   d_{01} & d_{01} & d_{01} & d_{01}
  \end{matrix}
\right] \tag{10}
-
\left[
 \begin{matrix}
  d_{02} & d_{03} & d_{04} & d_{05} \\
   d_{02} & d_{03} & d_{04} & d_{05}
  \end{matrix}
\right]  \\
&= 
\left[
 \begin{matrix}
  d_{00} - d_{02} & d_{00} - d_{03} & d_{00} - d_{04} & d_{00} - d_{05} \\
   d_{01} - d_{02} & d_{01} - d_{03} & d_{01} - d_{04} & d_{01} - d_{05}
  \end{matrix}
\right] 
\end{align}
$$

但是，从图1的第3行可以看出，\\(D_2^{neg}=D_2[0:2] + D_2[3:6]= [d_{20}, d_{21}, d_{23}, d_{24}, d_{25}]\\) 并不是连续的，


$$
\begin{align}
(D_0^{pos})^T  - D_0 &=
\left[
 \begin{matrix}
  d_{00} & d_{00} & d_{00} & d_{00} & d_{00} & d_{00} \\
  d_{01} & d_{01} & d_{01} & d_{01} & d_{01} & d_{01}
  \end{matrix}
\right] 
-
\left[
 \begin{matrix}
  d_{00} & d_{01} & d_{02} & d_{03} & d_{04} & d_{05} \\
  d_{00} & d_{01} & d_{02} & d_{03} & d_{04} & d_{05}
  \end{matrix}
\right]  \\
&=
\left[
 \begin{matrix}
  d_{00} - d_{00} & d_{00} - d_{01} & d_{00} - d_{02} & d_{00} - d_{03} & d_{00} - d_{04} & d_{00} - d_{05} \\
  d_{01} - d_{00} & d_{01} - d_{01} & d_{01} - d_{02} & d_{01} - d_{03} & d_{01} - d_{04} & d_{01} - d_{05}
  \end{matrix}
\right] 
\end{align} \tag{11}
$$

又：

$$
\begin{align}
(D_0^{pos})^T  -D_0^{pos} &= 
\left[
 \begin{matrix}
  d_{00} & d_{00} \\
  d_{01} & d_{01} 
  \end{matrix}
\right] 
-
\left[
 \begin{matrix}
  d_{00} & d_{01} \\
  d_{00} & d_{01}
  \end{matrix}
\right] \\
&= 
\left[
 \begin{matrix}
  d_{00} - d_{00}& d_{00} - d_{01} \\
  d_{01} - d_{00}& d_{01} - d_{01} 
  \end{matrix}
\right] 
\end{align} \tag{12}
$$

由等式（11）（12）计算得到anchor为\\(x_0\\)的所有triplet的loss的和为：

$$
\begin{align} 
l(x_0) &= sum\_all(relu((D_0^{pos})^T  - D_0^{neg})) \\
&=sum\_all(relu((D_0^{pos})^T  - D_0)) - sum\_all(relu((D_0^{pos})^T  - D_0^{pos})) 
\end{align} \tag{13}
$$

同理，可计算出\\(l(x_n)\\)， 最终的loss为：

$$
L = \sum_{i=0}^n{l(x_i)} \tag{14}
$$


## 4. Backward

TODO
