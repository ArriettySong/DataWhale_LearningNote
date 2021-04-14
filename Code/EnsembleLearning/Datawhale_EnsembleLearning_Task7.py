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

class VoteClassify:
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

    # get a voting ensemble of models, the base models are same
    def get_voting_samebasemodel(self):
        # define the base models
        model_dict = dict()
        # model_dict['knn1'] = KNeighborsClassifier(n_neighbors=1)
        # model_dict['knn3'] = KNeighborsClassifier(n_neighbors=3)
        # model_dict['knn5'] = KNeighborsClassifier(n_neighbors=5)
        # model_dict['knn7'] = KNeighborsClassifier(n_neighbors=7)
        # model_dict['knn9'] = KNeighborsClassifier(n_neighbors=9)

        model_dict['svm1_with_std'] = make_pipeline(StandardScaler(),
                                                    SVC(C=0.1,kernel='rbf',gamma=0.05,probability=True))
        model_dict['svm2_with_std'] = make_pipeline(StandardScaler(),
                                                    SVC(C=0.1,kernel='rbf',gamma=0.005,
                                                        probability=True))
        model_dict['svm3_with_std'] = make_pipeline(StandardScaler(),
                                                    SVC(C=0.01,kernel='linear',probability=True))
        model_dict['svm4_with_std'] = make_pipeline(StandardScaler(),
                                                    SVC(C=1.0,kernel='linear',probability=True))
        model_dict['svm5_with_std'] = make_pipeline(StandardScaler(),
                                                    SVC(C=10.0,kernel='linear',probability=True))

        # define the voting ensemble
        ensemble = VotingClassifier(estimators=self.model_dict2list(model_dict), voting=self.voting)  #
        # voting="soft"为软投票
        # hard为硬投票
        return ensemble,model_dict

    # get a voting ensemble of models, the base models are different
    def get_voting_diffbasemodel(self):
        # define the base models
        model_dict = dict()
        # model_dict['lr'] = LogisticRegression(max_iter=3000) #迭代次数少会报错
        model_dict['lr_with_std'] = make_pipeline(StandardScaler(),LogisticRegression())
        model_dict['svm'] = SVC(probability=True)
        model_dict['svm_with_std'] = make_pipeline(StandardScaler(),SVC(probability=True))
        model_dict['dt3'] = DecisionTreeClassifier(max_depth=3)
        model_dict['dt3_with_std'] = make_pipeline(StandardScaler(),DecisionTreeClassifier(max_depth=3))
        model_dict['knn5'] = KNeighborsClassifier(n_neighbors=5)
        model_dict['knn5_with_std'] = make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=5))
        model_dict['nb'] = GaussianNB()
        model_dict['nb_with_std'] = make_pipeline(StandardScaler(), GaussianNB())
        model_dict['rf'] = RandomForestClassifier(n_estimators=30)
        model_dict['rf_with_std'] = make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators=30))

    # define the voting ensemble
        ensemble = VotingClassifier(estimators=self.model_dict2list(model_dict), voting=self.voting)  # voting="soft"为软投票  hard为硬投票
        return ensemble,model_dict

    def model_dict2list(self,model_dict):
        models = list()
        for key in model_dict.keys():
            models.append((key,model_dict[key]))
        return models

    # get a list of models to evaluate
    def get_models(self):
        models = dict()
        if(self.basemodel == "diff"):
            voting_model,models = self.get_voting_diffbasemodel()
        else :
            voting_model,models = self.get_voting_samebasemodel()

        if(self.voting == "hard"):
            models['hard_voting'] = voting_model
        else :
            models['soft_voting'] = voting_model
        return models

    # evaluate a give model using cross-validation
    def evaluate_model(self,model, X, y):
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=1)
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        return scores



# 设置投票模型采用的数据集、基模型和投票方式
# vc = VoteClassify(dataset = "generate",basemodel = "diff",voting = "soft")
# vc = VoteClassify(dataset = "generate",basemodel = "same",voting = "soft")
vc = VoteClassify(dataset = "breastcancer",basemodel = "diff",voting = "hard")
# vc = VoteClassify(dataset = "breastcancer",basemodel = "diff",voting = "soft")
# vc = VoteClassify(dataset = "breastcancer",basemodel = "same",voting = "hard")
# vc = VoteClassify(dataset = "breastcancer",basemodel = "same",voting = "soft") # svm比knn效果要好

# define dataset
X, y = vc.get_dataset()
# get the models to evaluate
models = vc.get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = vc.evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()

# 对于基模型为同一类型模型的投票集成来说，硬投票效果好于软投票
# 基模型的参数怎么调，单独调吗？有必要在投票模型中整体调参吗？
# 如果要采用软投票，svc务必要设置probability=True
# 硬投票，如果1:1平怎么办？
