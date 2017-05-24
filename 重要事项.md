<center> 机器学习校招面试准备 <center>
======
## 机器学习基础知识
* [1）归类公理的阐述](#1)

* [2）回归模型的阐述](#2)


<h2 id="1">1.归类公理的阐述</h2>
### 1.1）类表示公理

---
**归类输入有 **内蕴表示** 和 **外部表示****


归类输入的外部表示可以表示为
$$(X,U)$$
对象特性输入表示
$$X = ( x_1,x_2,x_3,...,x_N ) $$
归为 c 个子集
$$ ( X_1,X_2,...,X_c )$$
其对应的归类输入的类外延表示由划分矩阵 $U = [u_{ik}]_{c*N}$ 其中$u_{ik}$ 表示对象 $o_k$ 属于第 $i$ 个输入类的隶属度， $u_{ik} \geq 0$。
不同的划分矩阵约束
* **硬划分**：$\sum\nolimits_{i = 1}^c {{u_{ik}}}  = 1$,  ${u_{ik}} \in \{ 0,1\}$，$\sum\nolimits_{k = 1}^N {{u_{ik}} > 1}$
* **软划分**：$\sum\nolimits_{i = 1}^c {{u_{ik}}}  = 1$,  ${u_{ik}} \geq  0$，$\sum\nolimits_{k = 1}^N {{u_{ik}} > 0}$
* **可能性划分**：$\sum\nolimits_{i = 1}^c {{u_{ik}}}  > 0$,  ${u_{ik}} \geq  0$，$\sum\nolimits_{k = 1}^N {{u_{ik}} > 0}$

因此，归类输入的外部表示可以表示为 $(X,U)$。当 $U$ 知道，一个对象总是被指派到具有最大隶属度的类中，由此可以定义指派算子 $\to$ 如下
\[
\overrightarrow X  = \{ \overrightarrow {{x_1}},\overrightarrow {{x_2}},\overrightarrow {{x_3}},...,\overrightarrow {{x_N}} \}
\]
其中，$\overrightarrow {x_k} = arg\ max_i\ u_{ik}$

$\overrightarrow {x_k}$ 可以读作 $x_k$外部指称为为第 $\overrightarrow {x_k}$ 类

这里可以**参考例题 9**

---

归类输出的外部表示可以表示为
$$(Y,V)$$
对象特性输出表示
$$Y = ( y_1,y_2,y_3,...,y_N ) $$
归为 c 个子集
$$ ( Y_1,Y_2,...,Y_c )$$
其对应的归类输出的类外延表示由划分矩阵 $U = [u_{ik}]_{c*N}$ 其中$u_{ik}$ 表示对象 $o_k$ 属于第 $i$ 个输出类的隶属度， $u_{ik} \geq 0$。
不同的划分矩阵约束
* **硬划分**：$\sum\nolimits_{i = 1}^c {{u_{ik}}}  = 1$,  ${u_{ik}} \in \{ 0,1\}$，$\sum\nolimits_{k = 1}^N {{u_{ik}} > 1}$
* **软划分**：$\sum\nolimits_{i = 1}^c {{u_{ik}}}  = 1$,  ${u_{ik}} \geq  0$，$\sum\nolimits_{k = 1}^N {{u_{ik}} > 0}$
* **可能性划分**：$\sum\nolimits_{i = 1}^c {{u_{ik}}}  > 0$,  ${u_{ik}} \geq  0$，$\sum\nolimits_{k = 1}^N {{u_{ik}} > 0}$

因此，归类输出的外部表示可以表示为 $(X,U)$。当 $U$ 知道，一个对象总是被指派到具有最大隶属度的类中，由此可以定义指派算子 $\to$ 如下
\[
\overrightarrow Y  = \{ \overrightarrow {{y_1}},\overrightarrow {{y_2}},\overrightarrow {{y_3}},...,\overrightarrow {{y_N}} \}
\]

根据输出划分矩阵的类型，归类方法可分为**硬归类**方法和**软归类**方法
硬归类方法的

**归类输出有 **内蕴表示** 和 **外部表示****
