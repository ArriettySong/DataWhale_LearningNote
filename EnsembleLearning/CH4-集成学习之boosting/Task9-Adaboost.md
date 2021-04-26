

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

（十）集成学习中——GBDT

（十一）集成学习中——XgBoost、LightGBM



[TOC]

## 1. 导论

Bagging：通过Bootstrap 的方式对全样本数据集进行抽样得到抽样子集，对不同的子集使用同一种基本模型进行拟合，然后投票得出最终的预测。Bagging主要通过降低方差的方式减少预测误差。

Boosting：使用同一组数据集进行反复学习，基于前一个模型的效果不断调整训练集的分布，得到一系列简单模型，这些简单模型都是按顺利生成的，然后组合这些模型构成一个预测性能十分强大的机器学习模型。Boosting主要是通过不断减少偏差来提高最终的预测效果。

在Boosting这一大类方法中，主要介绍两类常用的Boosting方式：Adaptive Boosting 和 Gradient Boosting 以及它们的变体Xgboost、LightGBM以及Catboost。

## 2. Boosting方法的基本思路

Boosting的提出与发展离不开Valiant和 Kearns的努力，历史上正是Valiant和 Kearns提出了"强可学习"和"弱可学习"的概念。那什么是"强可学习"和"弱可学习"呢？

在概率近似正确PAC学习的框架下：            

  - 弱学习：识别错误率小于1/2（准确率仅比随机猜测略高的学习算法）  <font color="red">accuracy大于0.5即可</font>
  - 强学习：识别准确率很高并能在多项式时间内完成的学习算法      <font color="red">accuracy可能需要大于0.9才能称之为好/强学习</font>

我们面对一个问题，用算法求解，想直接达到强学习的标准有些难，而达到弱学习的标准还是很容易的。我们可以从弱学习算法出发，反复学习，得到一系列弱分类器(又称为基本分类器)，然后通过一定的形式去组合这些弱分类器构成一个强分类器。

大多数的Boosting方法都是通过改变训练数据集的概率分布(训练数据不同样本的权值)，针对不同概率分布的数据调用弱分类算法学习一系列的弱分类器。 对于Boosting方法来说，有两个问题需要给出答案：

- 每一轮学习应该如何改变数据的概率分布

- 如何将各个弱分类器组合起来

关于第一个问题，有一类通俗的方法是：① 无差别选择数据集D1，训练分类器C1，将C1放回到D中，看哪些样本会分对，哪些会分错，然后对分错的样本加权，分对的样本降权，选择出数据集D2，训练分类器C2，以此类推。② 对D中的每个样本都用C1和C2进行预测，将C1和C2预测结果不同的数据进行抽取组装为数据集D3，训练分类器C3。

关于第二个问题，通俗的做法，可以采用投票法。



而不同的Boosting算法会有不同的答案，我们接下来介绍一种最经典的Boosting算法----Adaboost，我们需要理解Adaboost是怎么处理这两个问题以及为什么这么处理的。



> **关于PAC（Probably Approximately Correct）：**
>
>  PAC代表“可能近似正确”，而我们的数字猜谜游戏清楚地表明了这意味着什么。 近似正确意味着时间间隔与真实时间间隔足够接近，因此在新样本上误差会很小，并且可能意味着如果我们反复进行游戏，通常可以得到一个很好的近似值。 就是说，我们会找到一个概率很高的近似间隔。
>  
> 参考资料：https://jeremykun.com/2014/01/02/probably-approximately-correct-a-formal-theory-of-learning/



## 3. Adaboost算法

**Adaboost的基本原理**（待补充）

Adaboost训练误差的上界是可以推导的。

Adaboost的参数α是可以学习以及推导出来的，而不是超参数。

清华的这位老师讲解推导讲得非常非常好。

https://www.bilibili.com/video/BV1mt411a7FT

优点：

- 简洁、简单、有效
- 几乎不需要调参
- 可以证明在训练集上训练误差是有上界的
- 对过拟合免疫

缺点：

- 次优α值
- 最陡下降
- 对噪音敏感

### 3.1 Adaboost实践（基于sklearn）

数据集：乳腺癌数据集/生成数据集

`AdaBoostClassifier`相关参数：
	`base_estimator`：基本分类器，默认为DecisionTreeClassifier(max_depth=1)
	`n_estimators`：终止迭代的次数
	`learning_rate`：学习率
	`algorithm`：训练的相关算法，{'SAMME'，'SAMME.R'}，默认='SAMME.R'
	`random_state`：随机种子

代码：


```python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier

class AdaboostClassify:

    def __init__(self,dataset):
        """:arg
        dataset: 数据集，generate生成数据  breastcancer 乳腺癌数据
        """
        self.dataset = dataset

    def get_dataset(self):
        if self.dataset == "generate":
            X,y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                                      n_redundant=5, random_state=2)
            return X, y
        elif self.dataset == "breastcancer":
            # 乳腺癌数据集
            breast_cancer = datasets.load_breast_cancer()
            # print(breast_cancer.feature_names)
            df = pd.concat([pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)
                               ,pd.DataFrame(breast_cancer.target,columns=["label"])
                            ]
                           ,axis=1)
            # print(df.head())
            X=breast_cancer.data
            y=breast_cancer.target
            # print(X.shape)
            # print(df['label'].value_counts())
            return X, y


data = AdaboostClassify(dataset="breastcancer")
# data = AdaboostClassify(dataset="generate")


# define dataset
X, y = data.get_dataset()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)

# 使用单棵决策树
# define the model
tree = DecisionTreeClassifier(criterion='entropy',random_state=1,max_depth=None)   #选择决策树为基本分类器
# evaluate the model
n_scores1 = cross_val_score(tree, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Decision tree Accuracy: %.3f ,%.3f' % (np.mean(n_scores1), np.std(n_scores1)))

# 使用Adaboost(基分类器为决策树)
# define the model

from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(base_estimator=tree,n_estimators=500,learning_rate=0.1,random_state=1)
# evaluate the model
n_scores2 = cross_val_score(ada, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Adaboost Accuracy: %.3f ,%.3f' % (np.mean(n_scores2), np.std(n_scores2)))

```

    Decision tree Accuracy: 0.928 ,0.020
    Adaboost Accuracy: 0.939 ,0.024


​    

总结：

Adaboost模型的决策边界比单层决策树的决策边界要复杂的多。

与单个分类器相比，Adaboost等Boosting模型增加了计算的复杂度，在实践中需要仔细思考是否愿意为预测性能的相对改善而增加计算成本，而且Boosting方式无法做到现在流行的并行计算的方式进行训练，因为每一步迭代都要基于上一步的基本分类器。



参考资料：

https://www.bilibili.com/video/BV1Cs411c7Zt

https://towardsdatascience.com/understanding-adaboost-2f94f22d5bfe

https://www.bilibili.com/video/BV1xK411c7wt

https://www.bilibili.com/video/BV1mt411a7FT



--删掉

https://baijiahao.baidu.com/s?id=1633580172255481867&wfr=spider&for=pc

http://vqclabeling.58corp.com/#/checkView/E95AD3693E2F83B899320B781738C9BD240020210415/13/default