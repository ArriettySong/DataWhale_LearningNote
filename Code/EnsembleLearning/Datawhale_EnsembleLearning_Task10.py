# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier

class GBDT:

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


from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor

'''
GradientBoostingRegressor参数解释：
loss：{‘ls’, ‘lad’, ‘huber’, ‘quantile’}, default=’ls’：‘ls’ 指最小二乘回归. ‘lad’ (最小绝对偏差) 是仅基于输入变量的顺序信息的高度鲁棒的损失函数。. ‘huber’ 是两者的结合. ‘quantile’允许分位数回归（用于alpha指定分位数）
learning_rate：学习率缩小了每棵树的贡献learning_rate。在learning_rate和n_estimators之间需要权衡。
n_estimators：要执行的提升次数。
subsample：用于拟合各个基础学习者的样本比例。如果小于1.0，则将导致随机梯度增强。subsample与参数n_estimators。选择会导致方差减少和偏差增加。subsample < 1.0
criterion：{'friedman_mse'，'mse'，'mae'}，默认='friedman_mse'：“ mse”是均方误差，“ mae”是平均绝对误差。默认值“ friedman_mse”通常是最好的，因为在某些情况下它可以提供更好的近似值。
min_samples_split：拆分内部节点所需的最少样本数
min_samples_leaf：在叶节点处需要的最小样本数。
min_weight_fraction_leaf：在所有叶节点处（所有输入样本）的权重总和中的最小加权分数。如果未提供sample_weight，则样本的权重相等。
max_depth：各个回归模型的最大深度。最大深度限制了树中节点的数量。调整此参数以获得最佳性能；最佳值取决于输入变量的相互作用。
min_impurity_decrease：如果节点分裂会导致杂质的减少大于或等于该值，则该节点将被分裂。
min_impurity_split：提前停止树木生长的阈值。如果节点的杂质高于阈值，则该节点将分裂
max_features{‘auto’, ‘sqrt’, ‘log2’}，int或float：寻找最佳分割时要考虑的功能数量：

如果为int，则max_features在每个分割处考虑特征。

如果为float，max_features则为小数，并 在每次拆分时考虑要素。int(max_features * n_features)

如果“auto”，则max_features=n_features。

如果是“ sqrt”，则max_features=sqrt(n_features)。

如果为“ log2”，则为max_features=log2(n_features)。

如果没有，则max_features=n_features。
'''

X, y = make_friedman1(n_samples=1200, random_state=0, noise=1.0)
X_train, X_test = X[:200], X[200:]
y_train, y_test = y[:200], y[200:]
est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                max_depth=3, random_state=0, loss='ls').fit(X_train, y_train)
print("",mean_squared_error(y_test, est.predict(X_test)))

from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
X, y = make_regression(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
reg = GradientBoostingRegressor(random_state=0)
reg.fit(X_train, y_train)
print("",reg.score(X_test, y_test))
