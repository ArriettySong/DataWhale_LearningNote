参考：[DataWhale教程链接](https://github.com/datawhalechina/team-learning-data-mining/tree/master/EnsembleLearning)



前面的Task：

[（一）集成学习上——机器学习三大任务](https://blog.csdn.net/youyoufengyuhan/article/details/114853640)

[（二）集成学习上——回归模型](https://blog.csdn.net/youyoufengyuhan/article/details/114994155)

[（三）集成学习上——偏差与方差](https://blog.csdn.net/youyoufengyuhan/article/details/115080030)

[（四）集成学习上——回归模型评估与超参数调优](https://blog.csdn.net/youyoufengyuhan/article/details/115136244)

[TOC]

### 2.1.6 对模型超参数进行<font color=red>调优</font>(调参)

​	在刚刚的讨论中，我们似乎对模型的优化都是对模型算法本身的改进，比如：岭回归对线性回归的优化在于线性回归的损失函数中加入L2正则化项从而牺牲无偏性降低方差。但是，大家是否想过这样的问题：在L2正则化中参数$\lambda$应该选择多少？是0.01、0.1、还是1？到目前为止，我们只能凭经验或者瞎猜，能不能找到一种方法找到最优的参数$\lambda$？事实上，找到最佳参数的问题本质上属于最优化的内容，因为从一个参数集合中找到最佳的值本身就是最优化的任务之一，我们脑海中浮现出来的算法无非就是：梯度下降法、牛顿法等无约束优化算法或者约束优化算法，但是在具体验证这个想法是否可行之前，我们必须先认识两个最本质概念的区别。                                     

#### （1） 参数与超参数

​	我们很自然的问题就是岭回归中的参数$\lambda$和参数$w$之间有什么不一样？

```
我的简单理解：
参数：求解得到的最优解，人工通过无法干预
超参数：在设定超参数的前提下，进行求解，人工可以干预
```

​	事实上，参数$w$是我们通过设定某一个具体的$\lambda$后使用类似于最小二乘法、梯度下降法等方式优化出来的，我们总是设定了$\lambda$是多少后才优化出来的参数$w$。因此，类似于参数$w$一样，使用最小二乘法或者梯度下降法等最优化算法优化出来的数我们称为参数，类似于$\lambda$一样，无法使用最小二乘法或者梯度下降法等最优化算法优化出来的数我们称为超参数。 

- **参数** 
  ​	参数，模型内部的配置，其值需要从数据中估计。

  - 参数定义了可使用的模型。                        
  - 参数是从数据中估计的。                       
  - 参数通常不由编程者手动设置。                     
  - 参数通常被保存为学习模型的一部分。                      
  - 参数是机器学习算法的关键，它们通常由过去的训练数据中总结得出 。                          

   - **超参数**

     ​	超参数，不直接在估计器内学习的参数。在 scikit-learn 包中，它们作为估计器类中构造函数的参数进行传递。典型的例子有：用于支持向量分类器的 `C` 、`kernel` 和 `gamma` ，用于Lasso的 `alpha` 等

     - 超参数通常用于帮助估计模型参数。
     - 超参数通常由人工指定。
     - 超参数通常可以使用启发式设置。<font color=green>什么是启发式设置</font>
     - 超参数经常被调整为给定的预测建模问题。                            

我们在[（三）集成学习上——偏差与方差](https://blog.csdn.net/youyoufengyuhan/article/details/115080030)讨论的优化都是基于模型本身的具体形式的优化，那本次讨论的调整优化的内容是超参数，也就是取不同的超参数的值对于模型的性能有不同的影响。   

​           

#### （2） 寻找最优超参数

##### ① 网格搜索 GridSearchCV              

​	网格搜索在sklearn中，有两个接口可用。分别为：[网格搜索GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html?highlight=gridsearchcv#sklearn.model_selection.GridSearchCV ) 、[网格搜索结合管道Pipeline and GridSearchCV](https://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html?highlight=gridsearchcv  )

​	网格搜索的思想非常简单，比如你有2个超参数需要去选择，那你就把所有的超参数选择列出来分别做排列组合。举个例子：$\lambda = 0.01,0.1,1.0$和$\alpha = 0.01,0.1,1.0$,你可以做一个排列组合，即：${[0.01,0.01],[0.01,0.1],[0.01,1],[0.1,0.01],[0.1,0.1],[0.1,1.0],[1,0.01],[1,0.1],[1,1]}$  ，然后针对每组超参数分别建立一个模型，选择测试误差最小的那组超参数。换句话说，我们需要从超参数空间中寻找最优的超参数，很像一个网格中找到一个最优的节点，因此叫网格搜索。 

##### ② 随机搜索 RandomizedSearchCV              

​	 [随机搜索RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html?highlight=randomizedsearchcv#sklearn.model_selection.RandomizedSearchCV ) 

​	网格搜索相当于暴力地从参数空间中每个都尝试一遍，然后选择最优的那组参数，这样的方法显然是不够高效的，因为随着参数类别个数的增加，需要尝试的次数呈指数级增长。有没有一种更加高效的调优方式呢？那就是使用随机搜索的方式，这种方式不仅仅高效，而且实验证明，随机搜索法结果比稀疏化网格法稍好(有时候也会极差，需要权衡)。参数的随机搜索中的每个参数都是从可能的参数值的分布中采样的。与网格搜索相比，这有两个主要优点：        

- 可以独立于参数数量和可能的值来选择计算成本。                 
- 添加不影响性能的参数不会降低效率。                      <font color=green>有点拗口，啥意思？</font>     



[GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) 考虑了所有参数组合，而 [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV) 可以从具有指定分布的参数空间中抽取给定数量的候选

#### （3） 寻找最优超参数实践

下面我们使用SVR的例子，结合管道来进行调优：

##### ① baseline


```python
# 我们先来对未调参的SVR进行评价： 
import numpy as np
from sklearn.svm import SVR     # 引入SVR类
from sklearn.pipeline import make_pipeline   # 引入管道简化学习流程
from sklearn.preprocessing import StandardScaler # 由于SVR基于距离计算，引入对数据进行标准化的类
from sklearn.model_selection import GridSearchCV  # 引入网格搜索调优
from sklearn.model_selection import cross_val_score # 引入K折交叉验证
from sklearn import datasets


boston = datasets.load_boston()     # 返回一个类似于字典的类
X = boston.data
y = boston.target
features = boston.feature_names
pipe_SVR = make_pipeline(StandardScaler(),SVR())
score1 = cross_val_score(estimator=pipe_SVR,
                                                     X = X,
                                                     y = y,
                                                     scoring = 'r2',
                                                      cv = 10)       # 10折交叉验证
print("CV accuracy: %.3f +/- %.3f" % ((np.mean(score1)),np.std(score1)))
```

    CV accuracy: 0.187 +/- 0.649

##### ② 网格搜索`GridSearchCV`

*class* `sklearn.model_selection.GridSearchCV`(*estimator*, *param_grid*, *, *scoring=None*, *n_jobs=None*, *refit=True*, *cv=None*, *verbose=0*, *pre_dispatch='2\*n_jobs'*, *error_score=nan*, *return_train_score=False*)

`param_grid` 参数：在指定的网格参数值中全面生成候选。

比如，在下面的例子中，候选集包括：

第一组：`kernel`为'linear'，C取值为`param_range`，共8个候选

第二组：`kernel`为'rbf'，C取值为`param_range`，`gamma`取值为`param_range`，共8*8=64个候选

共64+8=72个候选超参数集合。

```python
# 下面我们使用网格搜索来对SVR调参：
from sklearn.pipeline import Pipeline
pipe_svr = Pipeline([("StandardScaler",StandardScaler()),("svr",SVR())])
param_range = [0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]
param_grid = [{"svr__C":param_range,"svr__kernel":["linear"]},  # 注意__是指两个下划线，一个下划线会报错的
                            {"svr__C":param_range,"svr__gamma":param_range,"svr__kernel":["rbf"]}]
gs = GridSearchCV(estimator=pipe_svr,
                                                     param_grid = param_grid,
                                                     scoring = 'r2',
                                                      cv = 10)       # 10折交叉验证
gs = gs.fit(X,y)
print("网格搜索最优得分：",gs.best_score_)
print("网格搜索最优参数组合：\n",gs.best_params_)
```

    网格搜索最优得分： 0.6081303070817127
    网格搜索最优参数组合：
     {'svr__C': 1000.0, 'svr__gamma': 0.001, 'svr__kernel': 'rbf'}

##### ③ 随机搜索RandomizedSearchCV

*class* `sklearn.model_selection.``RandomizedSearchCV`(*estimator*, *param_distributions*, ***, *n_iter=10*, *scoring=None*, *n_jobs=None*, *refit=True*, *cv=None*, *verbose=0*, *pre_dispatch='2\*n_jobs'*, *random_state=None*, *error_score=nan*, *return_train_score=False*)

 `n_iter` 参数指定计算预算, 即取样候选项数或取样迭代次数。

```python
# 下面我们使用随机搜索来对SVR调参：
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform  # 引入均匀分布设置参数
pipe_svr = Pipeline([("StandardScaler",StandardScaler()),("svr",SVR())])
distributions = dict(svr__C=uniform(loc=1.0, scale=4),    # 构建连续参数的分布
                     svr__kernel=["linear","rbf"],   # 离散参数的集合
                    svr__gamma=uniform(loc=0, scale=4))

rs = RandomizedSearchCV(estimator=pipe_svr,
                                                     param_distributions = distributions,
                                                     scoring = 'r2',
                                                      cv = 10)       # 10折交叉验证
rs = rs.fit(X,y)
print("随机搜索最优得分：",rs.best_score_)
print("随机搜索最优参数组合：\n",rs.best_params_)
```

    随机搜索最优得分： 0.4522636151908035
    随机搜索最优参数组合：
     {'svr__C': 4.645600191609386, 'svr__gamma': 0.06758542185532646, 'svr__kernel': 'rbf'}


​	

拓展学习资料：

机器学习模型评估与超参数调优详解https://zhuanlan.zhihu.com/p/140040705
机器学习优化算法之贝叶斯优化 https://zhuanlan.zhihu.com/p/146329121 
最优化理论之无约束优化基本结构及其python应用 https://zhuanlan.zhihu.com/p/163405865