

# 异常检测——高维数据异常检测

**主要内容包括：**

* **Feature Bagging**

- **孤立森林** 

  

[TOC]

## 1、引言

在实际场景中，很多数据集都是多维度的。随着维度的增加，数据空间的大小（体积）会以指数级别增长，使数据变得稀疏，这便是维度诅咒的难题。维度诅咒不止给异常检测带来了挑战，对距离的计算，聚类都带来了难题。例如基于邻近度的方法是在所有维度使用距离函数来定义局部性，但是，在高维空间中，所有点对的距离几乎都是相等的（距离集中），这使得一些基于距离的方法失效。在高维场景下，一个常用的方法是**子空间方法**。

集成是子空间思想中常用的方法之一，可以有效提高数据挖掘算法精度。集成方法将多个算法或多个基检测器的输出结合起来。其基本思想是一些算法在某些子集上表现很好，一些算法在其他子集上表现很好，然后集成起来使得输出更加鲁棒。集成方法与基于子空间方法有着天然的相似性，子空间与不同的点集相关，而集成方法使用基检测器来探索不同维度的子集，将这些基学习器集合起来。

下面来介绍两种常见的集成方法：Feature Bagging 和Isolation Forest。

## 2、Feature Bagging

### 2.1 Feature Bagging的通用算法

Feature Bagging，基本思想与bagging相似，只是对象是feature。feature bagging属于集成方法的一种，下图是feature bagging的通用算法：

![image-20210104144520790](./img/image-20210104144520790.png)

- 给定：$S\{(x_1,y_1),...,(x_m,y_m)\}$  特征$x_i∈X^d$，标签$y_i∈Y=\{C,NC\}$，$C$为异常值，$NC$为非异常值。

- 标准化数据集$S$

- 对于$t=1,2,3,4,...T$

  1. 从$\lfloor d/2 \rfloor$和$(d-1)$之间的均匀分布中随机选择特征子集的特征数$N_t$（至少取一半的特征）
  2. 无放回的随机选择$N_t$个特征创建特征子集$F_t$
  3. 采用特征子集$F_t$应用异常检测算法$O_t$
  4. 异常检测算法$O_t$的输出是异常得分向量$AS_t$

- 整合异常得分向量$AS_t$，并输出最终异常得分向量$AS_{FINAL}$:

  $AS_{FINAL}=COMBINE(AS_t),t=1,2,...,T$



### 2.2 Feature Bagging 的设计关键点

#### 1. 选择基检测器

这些基本检测器可以彼此完全不同，或不同的参数设置，或使用不同采样的子数据集。Feature bagging常用LOF算法为基算法。（KNN、ABOD等同样ok）

#### 2. 分数标准化

不同检测器可能会在不同的尺度上产生分数。例如，平均k近邻检测器会输出原始距离分数，而LOF算法会输出归一化值。另外，尽管一般情况是输出较大的异常值分数，但有些检测器会输出较小的异常值分数。因此，需要将来自各种检测器的分数转换成可以有意义的组合的归一化值。

#### 3. 分数组合方法

分数标准化之后，还要选择一个组合函数将不同基本检测器的得分进行组合，最常见的选择包括平均和最大化组合函数。下面是两个feature bagging两个不同的组合分数方法：

**广度优先：**

> 本质上：取每条记录的最高的异常值分数
>
> 方式：对每条记录每个基检测器预测的异常值分数从高到低一起排序，遍历$Ind_t(i)$，将每条记录最高的异常值分数存入最终结果表。

- 给定$(AS_t),t=1,2,...,T$， 和$m$：数据集$S$和每个向量$AS_t$的大小size
- 将所有异常值分数向量$AS_t$排序，得到向量$SAS_t$，并返回已排序向量的索引$Ind_t$。比如，$SAS_t(1)$有最高的异常值分数，$Ind_t(1)$为这条有着最高异常值分数的数据记录$SAS_t(1)$在数据集$S$中的索引。
- 初始化$AS_{FINAL}$和$Ind_{FINAL}$为空向量。
- For $i=1$ to $m$ （最外层遍历所有数据记录）
  - For $t=1$ to $T$​ （里面一层遍历所有基检测器检测结果）
    - 如果索引为$Ind_t(i)$的数据记录被第$t$个异常检测算法排在第$i$位，并且异常得分$AS_t(i)$不在向量$Ind_{FINAL}$中
      - 将$Ind_t(i)$插入向量$Ind_{FINAL}$的尾部
      - 将$AS_t(i)$插入向量$AS_{FINAL}$的尾部
- 返回$AS_{FINAL}$和$Ind_{FINAL}$

![image-20210105140222697-1609839336763](img/image-20210105140222697-1609839336763.png)

排序取最大值的过程不大好理解，举个栗子：

![image-20210123180158183](.\img\image-20210123180158183.png)



**累计求和：**

> 本质上：取每条记录的平均异常值分数
>
> 方式：对每条记录每个基检测器得出的异常值分数求和

- 给定$(AS_t),t=1,2,...,T$， 和$m$：数据集$S$和每个向量$AS_t$的大小size

- 对异常值分数$AS_t$求和：

- For $i=1$ to $m$ （最外层遍历所有数据记录）

  $AS_{FINAL}(i)=\sum_{t=1}^TAS_t(i)$

- 返回$AS_{FINAL}$



![image-20210105140242611](./img/image-20210105140242611.png)

​													

基探测器的设计及其组合方法都取决于特定集成方法的特定目标。很多时候，我们无法得知数据的原始分布，只能通过部分数据去学习。除此以外，算法本身也可能存在一定问题使得其无法学习到数据完整的信息。这些问题造成的误差通常分为偏差和方差两种。

**方差**：是指算法输出结果与算法输出期望之间的误差，描述模型的离散程度，数据波动性。

**偏差**：是指预测值与真实值之间的差距。即使在离群点检测问题中没有可用的基本真值

Feature Bagging可以降低方差



### 2.3 Bagging的必要条件

- 基检测器之间尽量独立。

- 基检测器的准确率要大于50%。

用数学来验证下，集成方法比单个基检测器效果更好。基检测器准确率为$x$，集成模型包含了25个基检测器。

```python
import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

x = np.linspace(0,1,20)
y = []
for epsilon in np.linspace(0,1,20):
    E = np.array([comb(25,i)*(epsilon**i)*((1-epsilon)**(25-i))
    for i in range(13,26)]).sum()
    y.append(E)
plt.plot(x,y,"o-",label="基检测器不同")
plt.plot(x,x,"--",color="red",label="基检测器相同")
plt.xlabel("单个基检测器准确率")
plt.ylabel("25个基检测器集成后的准确率")
plt.legend()
plt.show()
```

![image-20210123160501287](D:.\img\image-20210123160501287.png)

由图可见：

（1）当基检测器的准确率大于0.5时，集成的效果是比基检测器要好的。相反，当基检测器的准确率小于0.5，袋装的集成算法就失效了。

（2）当基检测器之间无太大差别时，集成的效果与基检测器无差别。



## 3、Isolation Forests

孤立森林（Isolation Forest）算法是周志华教授等人于2008年提出的异常检测算法，受随机森林的思想启发。是机器学习中少见的专门针对异常检测设计的算法之一，方法因为该算法时间效率高，能有效处理高维数据和海量数据，无须标注样本，在工业界应用广泛。

iForest 适用于连续数据（Continuous numerical data）的**异常检测**，将异常定义为“容易被孤立的离群点 (more likely to be separated)”——可以理解为分布稀疏且离密度高的群体较远的点。
用统计学来解释，在数据空间里面，分布稀疏的区域表示数据发生在此区域的概率很低，因而可以认为落在这些区域里的数据是异常的。

**解决的问题**

Swamping：异常数据点和正常数据点离得很近
Masking：异常数据点聚集，伪装成正常数据点。

![cf0221ee3b9bd073dfd2cfb05919e992.png](.\img\iforest-1.png)



### 3.1 Isolation Forest算法原理

#### 1. Isolation

先思考一个小问题，如果把上面这张图，随机选择一条方向的直线进行切分，直到图中的每个点都在一个单独的区域？



这个算法呢，理解起来不是太难，这个孤立森林，其实都是一堆二叉树~	
先从iTree讲起。

#### 2. Isolation Trees
首先，iTree是一棵**真二叉树**（proper binary tree），及每个节点都有0个或2个子节点。

**构建一棵iTree的过程：**

1. 从训练数据$X$中随机选择$n$个样本作为子样本，放入树的根节点，样本属性集为$A$。
2. iTree会在$A$中随机挑一个属性$a$。
3. 这组数据在属性$a$上的最大值为$max$，最小值为$min$，iTree会再次随机在$min$到$max$中间随机选一个值$v$。（随机了两次，好随便的iTree）
4. 接下来，对这组数据属性$a$的值与$v$做比较，$a$<$v$ 则进入左子树，$a$>$v$则进入右子树，
5. 然后呢，对左子树和右子树重复进行1-4，直到停止条件的发生。

**so 停止条件是？**
    当当当当~ 来啦~
    停止条件1：子树中仅有一个元素，或子树不再可分（数据均相同）。
    停止条件2：达到了树高度的阈值 $ceil(log_2^n$)。

#### 3. 异常得分的计算

每条待测数据的异常分数(Anomaly Score)，其计算公式为：

$s(x,n) = 2^-(\frac{E(h(x))}{c(n)})$

$c(n) = 2H(n-1)-\frac{2(n-1)}{n}$
其中，$H(k) = ln(k)+\xi$，$\xi$为欧拉常数

$h(x)$：数据x在iTree树上沿对应的条件分支往下走，直到达到叶子节点，并记录这过程中经过的路径长度。
$E(h(x))$：数据x在一组iTree树的平均路径长度。
$c(n)$：二叉搜索树的平均路径长度，用来对结果进行归一化处理（iTree与二叉查找树有着相同的结构，故借用二叉查找树对搜索不成功路径的平均长度的分析）
$\xi$:是欧拉常数，其值为0.577215664

• when $E(h(x))$ → $c(n)$, $s$ → 0.5;
• when $E(h(x))$→0, $s$→1;
• and when $E(h(x))$→$n−1$, $s$→0.

异常分数，具有以下性质：
1. 如果分数越接近1，其是异常点的可能性越高；

2. 如果分数都比0.5要小，那么基本可以确定为正常数据；

3. 如果所有分数都在0.5附近，那么数据不包含明显的异常样本。

  ![3c34461adfa6c9dd9a24e4bc0583963b.png](.\img\3c34461adfa6c9dd9a24e4bc0583963b.png)
  **$E(h(x))$与$s$的关系，单调递减**

#### 4. 模型训练与评估

经过对原始训练数据多次不放回抽样，构建了一堆iTree后，就组成了iForest。


### 3.2 iTree如何解决swamping和masking的问题

通过sub-sampling 无放回的随机采样，解决正常样本与异常点分界不清，交织在一起的问题。

![cf0221ee3b9bd073dfd2cfb05919e992.png](.\img\iforest-1.png)

(a) Original sample (4096 instances)

![cf0221ee3b9bd073dfd2cfb05919e992.png](.\img\iforest-2.png)

(b) Sub-sample (128 instances)






### 3.3 iForest总结
**特点**

- Non-parametric（两个参数，还是已经给了优秀的默认值的）
- unsupervised
- iForest仅对Global Anomaly敏感，即全局稀疏点敏感，不擅长处理局部的相对稀疏点（Local Anomaly）。

**优点**

- 计算成本相比基于距离或基于密度的算法更小
- 线性的时间复杂度
- 对内存需求极低
- 处理大数据集上有优势

**适用的数据：**

- 连续数据
- 异常数据远少于正常数据
- 异常数据与正常数据的差异较大
- 非超高维数据（高维需要降维）



## 4、练习

### 1. 调用feature bagging

PyOD库中的FeatureBagging算法的默认基检测器为LOF，但也可以通过estimator_params自行指定。（但是还不清楚怎么写）

```python
estimator_params : dict, optional (default=None)
    The list of attributes to use as parameters
    when instantiating a new base estimator. If none are given,
    default parameters are used.
```

#### 1.1 使用PyOD库生成toy example并调用feature bagging

**调用方式1：单个LOF基检测器**

```python
clf_name = 'LOF'
clf = LOF(n_neighbors=35)
clf.fit(X_train)
```

<img src=".\img\image-20210123175203307.png" alt="image-20210123175203307" style="zoom:50%;" />

**调用方式2：多个LOF的FeatureBagging**

对比图可见：多个LOF基检测器的FeatureBagging效果好于单个LOF检测器

```python
clf_name = 'FeatureBagging'
clf = FeatureBagging(check_estimator=False)	
# clf = FeatureBagging(LOF(n_neighbors=35))	
clf.fit(X_train)
```

<img src=".\img\image-20210123174400481.png" alt="image-20210123174400481" style="zoom:50%;" />

**调用方式3：多个HBOS的FeatureBagging**

对比图可见：基检测器为HBOS时识别准确率比LOF差

```python
clf_name = 'FeatureBagging'
clf = FeatureBagging(HBOS())
clf.fit(X_train)
```

<img src=".\img\image-20210123174616267.png" alt="image-20210123174616267" style="zoom:50%;" />



#### 1.2 采用信用卡欺诈数据调用feature bagging和isolation forest



基于LOF基检测器的Feature Bagging：

```
On Training Data:
Feature Bagging ROC:0.8564, precision @ rank n:0.0457

On Test Data:
Feature Bagging ROC:0.876, precision @ rank n:0.0352
```

基于HBOS基检测器的Feature Bagging：

```
On Training Data:
Feature Bagging ROC:0.9475, precision @ rank n:0.3029

On Test Data:
Feature Bagging ROC:0.9533, precision @ rank n:0.338
```

单个HBOS：

```
On Training Data:
HBOS ROC:0.947, precision @ rank n:0.3771

On Test Data:
HBOS ROC:0.9659, precision @ rank n:0.3239
```

iforest：

```
On Training Data:
IForest ROC:0.9474, precision @ rank n:0.22

On Test Data:
IForest ROC:0.9433, precision @ rank n:0.2746
```

总结：截止目前，在信用卡欺诈数据上单个检测器准确率最高的是HBOS，但是准确率依然不足50%，故基于HBOS的FeatureBagging准确率还不如单个HBOS。



**3.(思考题：feature bagging为什么可以降低方差？)**



**4.(思考题：feature bagging存在哪些缺陷，有什么可以优化的idea？)**



## 5、参考文献

[1]Goldstein, M. and Dengel,  A., 2012. Histogram-based outlier score (hbos):A fast unsupervised anomaly detection algorithm . In*KI-2012: Poster and Demo Track*, pp.59-63.

[2]https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

[3]《Outlier Analysis》——Charu C. Aggarwal



