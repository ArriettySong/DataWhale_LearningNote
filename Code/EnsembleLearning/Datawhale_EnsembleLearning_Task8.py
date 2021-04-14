# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier

class BaggingClassify:
    # test classification dataset
    # define dataset

    def __init__(self,dataset,basemodel = "diff",voting = "hard"):
        """:arg
        dataset: 数据集，generate生成数据  breastcancer 乳腺癌数据
        """
        self.basemodel = basemodel
        self.voting = voting
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

# evaluate bagging algorithm for classification

bc = BaggingClassify(dataset="breastcancer")

# define dataset
X, y = bc.get_dataset()
# define the model
model = BaggingClassifier()
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))