# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 13:16:58 2021

@author: huyaxue
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# 1. 加载乳腺癌数据，并做数据认知
# 乳腺癌数据集，特征均为连续型，适合用SVM
breast_cancer = datasets.load_breast_cancer()
# print(breast_cancer.feature_names)
df = pd.concat([pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)
                   ,pd.DataFrame(breast_cancer.target,columns=["label"])
                ]
               ,axis=1)
# print(df.head())
X=breast_cancer.data
y=breast_cancer.target
print(X.shape)
print(df['label'].value_counts())

# 2. 使用SVM分类
# 使用网格搜索进行超参数调优：
# 方式1：网格搜索GridSearchCV()
from sklearn.pipeline import make_pipeline   # 引入管道简化学习流程
from sklearn.preprocessing import StandardScaler # 引入对数据进行标准化的类
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import time

start_time = time.time()
pipe_svc = make_pipeline(StandardScaler(),SVC(random_state=1))
param_range = [.000001,0.00001,0.0001,0.001,0.01,0.05,0.1,0.5,1.0,5,10.0,50,100.0,1000.0,10000.0]
param_grid = [{'svc__C':param_range,'svc__kernel':['linear']},{'svc__C':param_range,'svc__gamma':param_range,'svc__kernel':['rbf']}]
gs = GridSearchCV(estimator=pipe_svc
                  ,param_grid=param_grid
                  ,scoring='accuracy'
                  ,cv=10
                  ,n_jobs=-1)
gs = gs.fit(X,y)
end_time = time.time()
print("SVM分类，网格搜索调参经历时间：%.3f S" % float(end_time-start_time))
print(gs.best_score_)
print(gs.best_params_)


# 随机网格搜索
from sklearn.model_selection import RandomizedSearchCV
start_time = time.time()
pipe_svc = make_pipeline(StandardScaler(),SVC(random_state=1))
param_range = [.000001,0.00001,0.0001,0.001,0.01,0.05,0.1,0.5,1.0,5,10.0,50,100.0,1000.0,10000.0]
param_grid = [{'svc__C':param_range,'svc__kernel':['linear']},{'svc__C':param_range,'svc__gamma':param_range,'svc__kernel':['rbf']}]
# param_grid = [{'svc__C':param_range,'svc__kernel':['linear','rbf'],'svc__gamma':param_range}]
gs = RandomizedSearchCV(estimator=pipe_svc
                        ,param_distributions=param_grid
                        ,scoring='accuracy'
                        ,cv=10
                        ,n_jobs=-1)
gs = gs.fit(X,y)
end_time = time.time()
print("SVM分类，随机网格搜索调参经历时间：%.3f S" % float(end_time-start_time))
print(gs.best_score_)
print(gs.best_params_)


# 绘制混淆矩阵,输出分类报告
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=1)
# stratify=y，表示切分数据按照y中的比例分配
pipe_svc = make_pipeline(StandardScaler(),SVC(kernel="rbf"
                                              ,gamma=0.01
                                              ,C=1.0))
pipe_svc.fit(X_train,y_train)
y_pred = pipe_svc.predict(X_test)
print("classification_report:\n",classification_report(y_test,y_pred))
print("confusion_matrix:\n",confusion_matrix(y_test,y_pred))
# confmat = confusion_matrix(y_true=y_test,y_pred=y_pred)
# fig,ax = plt.subplots(figsize=(3,3))
# ax.matshow(confmat, cmap=plt.cm.Blues,alpha=0.3)
# for i in range(confmat.shape[0]):
#     for j in range(confmat.shape[1]):
#         ax.text(x=j,y=i,s=confmat[i,j],va='center',ha='center')
# plt.xlabel('predicted label')
# plt.ylabel('true label')
# plt.show()


# 绘制ROC曲线
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import make_scorer,f1_score
scorer = make_scorer(f1_score,pos_label=0)
gs = GridSearchCV(estimator=pipe_svc,param_grid=param_grid,scoring=scorer,cv=10)
y_pred = gs.fit(X_train,y_train).decision_function(X_test)
#y_pred = gs.predict(X_test)
fpr,tpr,threshold = roc_curve(y_test, y_pred) ###计算真阳率和假阳率
roc_auc = auc(fpr,tpr) ###计算auc的值
plt.figure()
lw = 2
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, color='green',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假阳率为横坐标，真阳率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ')
plt.legend(loc="lower right")
plt.show()


# 毒蘑菇数据集，特征均为离散型，需要做处理。
df_mr = pd.read_csv(r"D:\WorkSpace\GitHub\dataset\mushrooms.csv")
df_mr.head()
X_mr=df_mr.iloc[:,1:]
y_mr=df_mr.iloc[:,0]
print(X_mr.shape)
print(y_mr.shape)
print(y_mr.value_counts())


