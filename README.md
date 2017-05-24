<center> 机器学习校招面试准备 <center>
======
## 机器学习基础知识

* [1）归类公理的阐述](#1)

* [2）回归模型的阐述](#2)


<h2 id="1">1.归类公理的阐述</h2>
### 1.1）类表示公理

---
### **预备知识**
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

**归类输出的外部表示可以表示为**
$$(Y,V)$$
对象特性输出表示
$$Y = ( y_1,y_2,y_3,...,y_N ) $$
归为 $c$ 个子集
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

硬归类方法的输出划分矩阵为**硬划分矩阵**

软归类方法的输出划分矩阵为 **软划分矩阵** 和 **可能性划分矩阵**

在硬归类方法中，一个对象只属于一个类，划分矩阵直接说明了各个对象属于哪一类，只有该对象明确属于该类时，其对应的元素为1，如该对象不属于该类，其对应的元素为0。

在软归类方法中，划分矩阵说明了各个对象属于各个类的可能性。对象的具体归属由指派算子决定。

显然，指派算子是归类对象的外显指称，表现了对象与类之间的外显对应关系。

假设 $\forall i$，第i类的输入认知表示为 $\underline {X_i}$，第i类的输出认知表示为 $\underline {Y_i}$，
类的认知表示已知时，一般是对象像哪类便归哪类。因此，需要定义类与对象的相似度。考虑到输入输出表示不一定，下面分别定义了输入类相似性映射和输出类相似性映射

***

### **输入类相似性映射**
$Sim_X:X*\{ \underline{X_1},\underline{X_2},\underline{X_3},...\underline{X_c}\}\to R_+$
是输入类相似性映射，满足条件：函数 $Sim_X(x_k,\underline {X_i})$ 值增加表示 $x_k$ 和 $\underline{X_i}$ 相似性增大，函数 $Sim_X(x_k,\underline{X_i})$ 值减少表示 $x_k$ 和 $\underline{X_i}$ 相似性减少

***
### **输出类相似性映射**
$Sim_Y:Y*\{\underline{Y_1},\underline{Y_2},\underline{Y_3},...,\underline{Y_c}\} \to R_+$ 是输出类相似性映射，满足条件：函数 $Sim_Y(y_k,\underline {Y_i})$ 值增加表示 $y_k$ 和 $\underline{Y_i}$ 相似性增大，函数 $Sim_X(y_k,\underline{Y_i})$ 值减少表示 $y_k$ 和 $\underline{Y_i}$ 相似性减少

***
### **相似算子 与 内蕴指称**
对象 $o_k$ 的特性输入 $x_k$ 与哪个输入类认知表示最相似，则归为哪一类。由此定义相似算子 $\sim$，如下：$\widetilde X = \{\widetilde{x}_1,\widetilde{x}_2,\widetilde{x}_3,...,\widetilde{x}_N\}$，其中，
$\widetilde x_k = arg\ max_i\ Sim_x(x_k,\underline{X_i})$。$\widetilde{x}_k$可以读作 $x_k$内蕴指称为第 $\widetilde{x}_k$ 类，也可以读作对象 $o_k$ 内蕴指称为第 $\widetilde{x}_k$ 类
***
### **输入类相异性映射**
$D_{S_X}：X*\{\}$
