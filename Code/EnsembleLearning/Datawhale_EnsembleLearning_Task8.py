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


data = BaggingClassify(dataset="breastcancer")
# data = BaggingClassify(dataset="generate")


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

