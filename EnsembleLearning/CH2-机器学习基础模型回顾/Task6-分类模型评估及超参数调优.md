

参考：[DataWhale教程链接](https://github.com/datawhalechina/team-learning-data-mining/tree/master/EnsembleLearning)

集成学习（上）所有Task：

[（一）集成学习上——机器学习三大任务](https://blog.csdn.net/youyoufengyuhan/article/details/114853640)

[（二）集成学习上——回归模型](https://blog.csdn.net/youyoufengyuhan/article/details/114994155)

[（三）集成学习上——偏差与方差](https://blog.csdn.net/youyoufengyuhan/article/details/115080030)

[（四）集成学习上——回归模型评估与超参数调优](https://blog.csdn.net/youyoufengyuhan/article/details/115136244)

[（五）集成学习上——分类模型](https://blog.csdn.net/youyoufengyuhan/article/details/115271877)

[（六）集成学习上——分类模型评估与超参数调优](https://blog.csdn.net/youyoufengyuhan/article/details/115282143)



[TOC]



### 2.1.5 <font color=red>评估</font>模型的性能并调参

​	教程给出了基于iris数据的不少参考代码，让我们换个数据集来试试。

#### 基于sklearn中乳腺癌数据

直接上代码：

````python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# 1. 加载乳腺癌数据，并做数据认知
# 乳腺癌数据集，特征均为连续型
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
param_range = [0.000001,0.00001,0.0001,0.001,0.01,0.05,0.1,0.5,1.0,5,10.0,50,100.0,1000.0,10000.0]
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
param_range = [0.000001,0.00001,0.0001,0.001,0.01,0.05,0.1,0.5,1.0,5,10.0,50,100.0,1000.0,10000.0]
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
````

结果：

```
(569, 30)
1    357
0    212
Name: label, dtype: int64
SVM分类，网格搜索调参经历时间：15.339 S
0.9788847117794486
{'svc__C': 0.1, 'svc__kernel': 'linear'}
SVM分类，随机网格搜索调参经历时间：0.615 S
0.9718984962406015
{'svc__kernel': 'rbf', 'svc__gamma': 0.01, 'svc__C': 50}
classification_report:
               precision    recall  f1-score   support

           0       1.00      0.91      0.95        64
           1       0.95      1.00      0.97       107

    accuracy                           0.96       171
   macro avg       0.97      0.95      0.96       171
weighted avg       0.97      0.96      0.96       171

confusion_matrix:
 [[ 58   6]
 [  0 107]]
```



![6-2](.\第二章：机器学习基础_files\6-2.png)

一些比较疑惑的点：

- 多次运行代码，网格搜索出来的最优解不同，该相信哪个？或者说，该如何进一步确认？

#### 基于sklearn中的lfw_people数据

参考了不少网上的代码，慢慢熟悉整个流程及环节。

```python
from time import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC


lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw_people.data
n_samples = X.shape[0]
n_features = X.shape[1]
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# 切分数据 训练：测试=7:3
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=12)

# 使用PCA降维，从1850维降至200维
n_components = 200
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))

print("before PCA:",X_train.shape)
print("after PCA:",X_train_pca.shape)


# 训练SVM分类模型，使用网格搜索寻找最优参数
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced'), param_grid
)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("svm最优参数:",clf.best_estimator_)


# 模型评估
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
```



```
n_samples: 1288
n_features: 1850
n_classes: 7
done in 0.216s
done in 0.014s
before PCA: (901, 1850)
after PCA: (901, 200)
done in 71.495s
svm最优参数: SVC(C=1000.0, class_weight='balanced', gamma=0.001)
done in 0.957s
                   precision    recall  f1-score   support

     Ariel Sharon       0.81      0.59      0.68        22
     Colin Powell       0.78      0.84      0.81        70
  Donald Rumsfeld       0.67      0.84      0.74        31
    George W Bush       0.90      0.92      0.91       171
Gerhard Schroeder       0.85      0.74      0.79        31
      Hugo Chavez       0.88      0.79      0.83        19
       Tony Blair       0.82      0.72      0.77        43

         accuracy                           0.84       387
        macro avg       0.82      0.78      0.79       387
     weighted avg       0.84      0.84      0.84       387

[[ 13   4   4   1   0   0   0]
 [  0  59   4   5   0   1   1]
 [  0   2  26   2   0   0   1]
 [  3   7   2 157   1   0   1]
 [  0   1   1   4  23   1   1]
 [  0   0   0   1   0  15   3]
 [  0   3   2   4   3   0  31]]
```



参考资料：

https://blog.csdn.net/cwlseu/article/details/52356665

https://blog.csdn.net/jasonzhoujx/article/details/81905923

https://scikit-learn.org.cn/view/187.html