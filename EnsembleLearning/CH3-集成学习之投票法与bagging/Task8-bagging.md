

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



[TOC]

## bagging VS Voting

Voting：基学习器可以是多种模型，最终基于基学习器的分类结果或者概率进行最终结果的投票。该集成方法的目的主要降低偏差。

Bagging：基学习器一般是单个模型，在采样上做文章，从训练样本中又放回的采样，对每一个基学习器采用不同的训练数据，并基于每个基学习器的结果进行投票。该集成方法的目的主要是降低方差。

## bagging的原理分析

Bagging的核心在于自助采样(bootstrap)这一概念，即有放回的从数据集中进行采样，也就是说，同样的一个样本可能被多次进行采样。

Bagging方法之所以有效，是因为每个模型都是在略微不同的训练数据集上拟合完成的，这又使得每个基模型之间存在略微的差异，使每个基模型拥有略微不同的训练能力。

Bagging同样是一种降低方差的技术，因此它在不剪枝决策树、神经网络等易受样本扰动的学习器上效果更加明显。在实际的使用中，加入列采样的Bagging技术对高维小样本往往有神奇的效果。

## bagging的案例分析

Sklearn为我们提供了 [BaggingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html) 与 [BaggingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html) 两种Bagging方法的API。我们来尝试使用一下分类的Bagging方法。

[BaggingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html) 的一些参数：

`base_estimator`: 基模型， 默认的分类基模型为决策树模型`sklearn.tree.DecisionTreeClassifier`

`n_estimators`：基学习器的数量，默认为10

`max_samples`：每个基学习器使用的最多的样本数，默认为全部样本

`max_features`：每个基学习器使用的最多的特征数，默认为全部特征

`bootstrap`：样本是否为有放回抽样，默认为True

`bootstrap_features`：特征是否为有放回抽样，默认为False

`oob_score`：是否使用袋外数据评估泛化误差，默认为False

`n_jobs`：为`fit`和`predict`设置的并行运行的作业数，默认为1，如果设置为-1，则表示全部可用的处理器



[BaggingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html) 模型的一些属性：

`base_estimator_`：基模型

`n_features_`：训练时使用的特征数量

`estimators_`：所有的基学习器

`estimators_samples_`：每个基学习器抽取的样本index 集合

`estimators_features_`：基学习器使用的特征子集

`classes_`：分类标签

`n_classes_`：分类的类别数

`oob_score_`：袋外估计评分

`oob_decision_function_`：袋外估计的决策函数



上代码：

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

class BaggingClassify:

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


# data = BaggingClassify(dataset="breastcancer")
data = BaggingClassify(dataset="generate")


# 使用Bagging分类的默认参数
# define dataset
X, y = data.get_dataset()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# 使用单棵决策树
# define the model
tree = DecisionTreeClassifier(criterion='entropy',random_state=1,max_depth=None)   #选择决策树为基本分类器
# evaluate the model
n_scores1 = cross_val_score(tree, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Decision tree Accuracy: %.3f ,%.3f' % (np.mean(n_scores1), np.std(n_scores1)))

# 使用决策树进行bagging
# define the model
bag = BaggingClassifier(base_estimator=tree,n_estimators=100,max_samples=1.0,max_features=1.0,
                        bootstrap=True,
                        bootstrap_features=False,n_jobs=1,random_state=1)
# evaluate the model
n_scores2 = cross_val_score(bag, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Bagging Accuracy: %.3f ,%.3f' % (np.mean(n_scores2), np.std(n_scores2)))

```

运行结果：

采用乳腺癌数据时：

```python
Decision tree Accuracy: 0.933 ,0.030
Bagging Accuracy: 0.958 ,0.026
```

采用生成数据时：

```python
Decision tree Accuracy: 0.782 ,0.042
Bagging Accuracy: 0.878 ,0.030
```

bagging的效果远远好于单个模型。



