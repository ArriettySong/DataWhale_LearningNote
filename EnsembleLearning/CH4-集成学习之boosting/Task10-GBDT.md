

参考：[DataWhale教程链接](https://github.com/datawhalechina/team-learning-data-mining/tree/master/EnsembleLearning)

集成学习（上）所有Task：

[（一）集成学习上——机器学习三大任务](https://blog.csdn.net/youyoufengyuhan/article/details/114853640)

[（二）集成学习上——回归模型](https://blog.csdn.net/youyoufengyuhan/article/details/114994155)

[（三）集成学习上——偏差与方差](https://blog.csdn.net/youyoufengyuhan/article/details/115080030)

[（四）集成学习上——回归模型评估与超参数调优](https://blog.csdn.net/youyoufengyuhan/article/details/115136244)

[（五）集成学习上——分类模型](https://blog.csdn.net/youyoufengyuhan/article/details/115271877)

[（六）集成学习上——分类模型评估与超参数调优](https://blog.csdn.net/youyoufengyuhan/article/details/115282143)

[（七）集成学习中——投票法](https://blog.csdn.net/youyoufengyuhan/article/details/115706397)

[（八）集成学习中——bagging](https://blog.csdn.net/youyoufengyuhan/article/details/115710507)

[（九）集成学习中——Boosting简介&AdaBoost](https://blog.csdn.net/youyoufengyuhan/article/details/115919031)

[（十）集成学习中——GBDT](https://blog.csdn.net/youyoufengyuhan/article/details/115956788)

（十一）集成学习中——XgBoost、LightGBM



[TOC]

## 4. 前向分步算法

我对前向分布算法的理解：

Adaboost是每次学习一个分类器以及该分类器的参数，基于前一个分类器的效果调整下一次分类器输入样本的权重。

**加法模型**





回看Adaboost的算法内容，我们需要通过计算M个基本分类器，每个分类器的错误率、样本权重以及模型权重。我们可以认为：Adaboost每次学习单一分类器以及单一分类器的参数(权重)。接下来，我们抽象出Adaboost算法的整体框架逻辑，构建集成学习的一个非常重要的框架----前向分步算法，有了这个框架，我们不仅可以解决分类问题，也可以解决回归问题。                               

**(1) 加法模型：**

在Adaboost模型中，我们把每个基本分类器合成一个复杂分类器的方法是每个基本分类器的加权和，即：$f(x)=\sum_{m=1}^{M} \beta_{m} b\left(x ; \gamma_{m}\right)$，其中，$b\left(x ; \gamma_{m}\right)$为基本分类器，$\gamma_{m}$为基本分类器的参数，$\beta_m$为基本分类器的权重

<font color=red>其中$\sum_{m=1}^{M} \beta_{m}=1$ </font>

在给定训练数据以及损失函数$L(y, f(x))$的条件下，学习加法模型$f(x)$就是：                        
$$
\min _{\beta_{m}, \gamma_{m}} \sum_{i=1}^{N} L\left(y_{i}, \sum_{m=1}^{M} \beta_{m} b\left(x_{i} ; \gamma_{m}\right)\right)
$$
通常这是一个复杂的优化问题，很难通过简单的凸优化的相关知识进行解决。前向分步算法可以用来求解这种方式的问题.

前向分布算法的基本思路是：**因为学习的是加法模型，如果从前向后，每一步只优化一个基函数及其系数，逐步逼近目标函数，那么就可以降低优化的复杂度。**具体而言，每一步只需要优化：                    
$$
\min _{\beta, \gamma} \sum_{i=1}^{N} L\left(y_{i}, \beta b\left(x_{i} ; \gamma\right)\right)
$$
**(2) 前向分步算法：**                              
给定数据集$T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}$，$x_{i} \in \mathcal{X} \subseteq \mathbf{R}^{n}$，$y_{i} \in \mathcal{Y}=\{+1,-1\}$。损失函数$L(y, f(x))$，基函数集合$\{b(x ; \gamma)\}$，我们需要输出加法模型$f(x)$。                         

   - 初始化：$f_{0}(x)=0$ 
   - 对m = 1,2,...,M:                     
      - (a) 极小化损失函数：
      $$
      \left(\beta_{m}, \gamma_{m}\right)=\arg \min _{\beta, \gamma} \sum_{i=1}^{N} L\left(y_{i}, f_{m-1}\left(x_{i}\right)+\beta b\left(x_{i} ; \gamma\right)\right)
      $$
      得到参数$\beta_{m}$与$\gamma_{m}$  <font color=red>模型$m$拟合模型$m-1$的残差 </font>
      - (b) 更新：
      $$
      f_{m}(x)=f_{m-1}(x)+\beta_{m} b\left(x ; \gamma_{m}\right)
      $$
   - 通过加法模型得到$f(x)$：                           
$$
f(x)=f_{M}(x)=\sum_{m=1}^{M} \beta_{m} b\left(x ; \gamma_{m}\right)
$$

这样，前向分步算法将同时求解**从m=1到M的所有参数**$\beta_{m}$，$\gamma_{m}$的优化问题简化为**逐次求解各个**$\beta_{m}$，$\gamma_{m}$的问题。                           
**(3) 前向分步算法与Adaboost的关系：**                                 
Adaboost算法是前向分步算法的特例，**Adaboost算法是由基本分类器组成的加法模型，损失函数为指数损失函数**。



## 5. 梯度提升决策树(GBDT)

GBDT是回归树，不是分类树

GBDT的核心在于累加所有树的结果作为最终结果。

GBDT的关键点就是利用损失函数的负梯度去模拟（代替）残差，这样对于一般的损失函数，只要是一阶可导就可以。



### 5.1 基于残差学习的提升树算法

框架内的基函数：决策树

分类树判断最佳划分点：信息增益（ID3算法），信息增益比（C4.5算法），基尼系数（CART分类树）

回归树判断最佳划分点：平方误差

分类问题权重更新：分类错误率

回归问题权重更新：样本残差

因此，我们可以得出如下算法：

输入数据集$T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}, x_{i} \in \mathcal{X} \subseteq \mathbf{R}^{n}, y_{i} \in \mathcal{Y} \subseteq \mathbf{R}$，输出最终的提升树$f_{M}(x)$                             

   - 初始化$f_0(x) = 0$                        
   - 对m = 1,2,...,M：                  
      - 计算每个样本的残差:$r_{m i}=y_{i}-f_{m-1}\left(x_{i}\right), \quad i=1,2, \cdots, N$                                    
      - 拟合残差$r_{mi}$学习一棵回归树，得到$T\left(x ; \Theta_{m}\right)$                        
      - 更新$f_{m}(x)=f_{m-1}(x)+T\left(x ; \Theta_{m}\right)$
   - 得到最终的回归问题的提升树：$f_{M}(x)=\sum_{m=1}^{M} T\left(x ; \Theta_{m}\right)$                         

实际案例见李航老师《统计学习方法》。

<font color=green>【看懂再补】</font>

下面对提升树继续优化。



### 5.2 梯度提升决策树算法(GBDT)： 

提升树利用加法模型和前向分步算法实现学习的过程，当损失函数为平方损失和指数损失时，每一步优化是相当简单的，也就是我们前面探讨的提升树算法和Adaboost算法。

但是对于一般的损失函数而言，往往每一步的优化不是那么容易，针对这一问题，我们得分析问题的本质，也就是是什么导致了在一般损失函数条件下的学习困难。对比以下损失函数：                          
$$
\begin{array}{l|l|l}
\hline \text { Setting } & \text { Loss Function } & -\partial L\left(y_{i}, f\left(x_{i}\right)\right) / \partial f\left(x_{i}\right) \\
\hline \text { Regression } & \frac{1}{2}\left[y_{i}-f\left(x_{i}\right)\right]^{2} & y_{i}-f\left(x_{i}\right) \\
\hline \text { Regression } & \left|y_{i}-f\left(x_{i}\right)\right| & \operatorname{sign}\left[y_{i}-f\left(x_{i}\right)\right] \\
\hline \text { Regression } & \text { Huber } & y_{i}-f\left(x_{i}\right) \text { for }\left|y_{i}-f\left(x_{i}\right)\right| \leq \delta_{m} \\
& & \delta_{m} \operatorname{sign}\left[y_{i}-f\left(x_{i}\right)\right] \text { for }\left|y_{i}-f\left(x_{i}\right)\right|>\delta_{m} \\
& & \text { where } \delta_{m}=\alpha \text { th-quantile }\left\{\left|y_{i}-f\left(x_{i}\right)\right|\right\} \\
\hline \text { Classification } & \text { Deviance } & k \text { th component: } I\left(y_{i}=\mathcal{G}_{k}\right)-p_{k}\left(x_{i}\right) \\
\hline
\end{array}
$$
观察Huber损失函数：                            
$$
L_{\delta}(y, f(x))=\left\{\begin{array}{ll}
\frac{1}{2}(y-f(x))^{2} & \text { for }|y-f(x)| \leq \delta \\
\delta|y-f(x)|-\frac{1}{2} \delta^{2} & \text { otherwise }
\end{array}\right.
$$
针对上面的问题，Freidman提出了梯度提升算法(gradient boosting)，这是利用最速下降法的近似方法，利用损失函数的负梯度在当前模型的值$-\left[\frac{\partial L\left(y, f\left(x_{i}\right)\right)}{\partial f\left(x_{i}\right)}\right]_{f(x)=f_{m-1}(x)}$作为回归问题提升树算法中的残差的近似值，拟合回归树。**与其说负梯度作为残差的近似值，不如说残差是负梯度的一种特例。**

以下开始具体介绍梯度提升算法：

输入训练数据集$T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}, x_{i} \in \mathcal{X} \subseteq \mathbf{R}^{n}, y_{i} \in \mathcal{Y} \subseteq \mathbf{R}$和损失函数$L(y, f(x))$，输出回归树$\hat{f}(x)$                              

   - 初始化$f_{0}(x)=\arg \min _{c} \sum_{i=1}^{N} L\left(y_{i}, c\right)$                     
   - 对于m=1,2,...,M：                   
      - 对i = 1,2,...,N计算：$r_{m i}=-\left[\frac{\partial L\left(y_{i}, f\left(x_{i}\right)\right)}{\partial f\left(x_{i}\right)}\right]_{f(x)=f_{m-1}(x)}$                
      - 对$r_{mi}$拟合一个回归树，得到第m棵树的叶结点区域$R_{m j}, j=1,2, \cdots, J$                           
      - 对j=1,2,...J，计算：$c_{m j}=\arg \min _{c} \sum_{x_{i} \in R_{m j}} L\left(y_{i}, f_{m-1}\left(x_{i}\right)+c\right)$                      
      - 更新$f_{m}(x)=f_{m-1}(x)+\sum_{j=1}^{J} c_{m j} I\left(x \in R_{m j}\right)$                    
   - 得到回归树：$\hat{f}(x)=f_{M}(x)=\sum_{m=1}^{M} \sum_{j=1}^{J} c_{m j} I\left(x \in R_{m j}\right)$

下面，我们来使用一个具体的案例来说明GBDT是如何运作的(案例来源：https://blog.csdn.net/zpalyq110/article/details/79527653 )：

<font color=green>【看懂再补】</font>

#### 5.2.1 GBDT的使用（基于sklearn）

[GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor) 及 [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html?highlight=gra#sklearn.ensemble.GradientBoostingClassifier) 参数解释：
`loss`：待优化的损失函数

"ls": 最小二乘回归，默认值

"lad":（最小绝对偏差）是仅基于输入变量的阶数信息的高度鲁棒的损失函数。

"buber": 上面两者的结合

"quantile":允许分位数回归（用`alpha`指定分位数
`learning_rate`：学习率可以缩小每棵树的贡献。需要在learning_rate和n_estimators之间权衡。
`n_estimators`：基模型数量，即要执行的提升次数。梯度提升对于过度拟合具有相当强的鲁棒性，因此大量提升通常会带来更好的性能。
`subsample`：用于拟合各个基学习器的样本比例。如果小于1.0，则将导致随机梯度增强。subsample与参数n_estimators共同作用。选择subsample < 1.0会导致方差减少和偏差增加。
`criterion`：衡量切分点的切分能力。

"friedman_mse"：friedman均方误差，默认值

"mse"：均方误差

"mae"：平均绝对误差
`min_samples_split`：切分节点所需的最少样本数
`min_samples_leaf`：在叶节点处需要的最小样本数。
`min_weight_fraction_leaf`：在所有叶节点处（所有输入样本）的权重总和中的最小加权分数。如果未提供sample_weight，则样本的权重相等。
`max_depth`：各个回归模型的最大深度。最大深度限制了树中节点的数量。调整此参数以获得最佳性能；最佳值取决于输入变量的相互作用。默认为3。
`min_impurity_decrease`：如果节点分裂会导致熵的减少大于或等于该值，则该节点将被分裂。
`min_impurity_split`：提前停止树木生长的阈值。如果节点的熵高于阈值，则该节点将分裂。
`max_features`：寻找最佳分割点时要考虑的特征数量

如果该值为int值，则在每个分割处考虑max_features个特征。

如果该值为float值，max_features则为小数，那在每次拆分时考虑int(max_features * n_features)个特征。

如果该值为“auto”，则max_features=n_features。

如果该值为“ sqrt”，则max_features=sqrt(n_features)。

如果该值为“ log2”，则为max_features=log2(n_features)。

如果没有，则max_features=n_features。




```python
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor

X, y = make_friedman1(n_samples=1200, random_state=0, noise=1.0)
X_train, X_test = X[:200], X[200:]
y_train, y_test = y[:200], y[200:]
est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
    max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)
mean_squared_error(y_test, est.predict(X_test))
```


    5.009154859960321


```python
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
X, y = make_regression(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)
reg = GradientBoostingRegressor(random_state=0)
reg.fit(X_train, y_train)
reg.score(X_test, y_test)

```


    0.43848663277068134



