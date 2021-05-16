

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

[（十一）集成学习中——XgBoost、LightGBM](https://blog.csdn.net/youyoufengyuhan/article/details/116179645)

[（十二）集成学习（下）——Blending](https://blog.csdn.net/youyoufengyuhan/article/details/116679272)



# Blending集成学习算法

Blending集成学习方式：

- (1) 将数据划分为训练集TrainData和测试集TestData，其中训练集需要再次划分为训练集Train_TrainData和验证集Train_ValData；

- (2) 构建第一层模型：选择$M$个基模型（对Train_TrainData数据集进行训练），这些模型可以使同质的也可以是异质的；

- (3) 训练第一层模型：使用Train_TrainData训练步骤2中的$M$个模型，然后用训练好的$M$个模型预测Train_ValData得到val_predict；

- (4) 构建第二层的模型：一般是逻辑回归；

- (5) 训练第二层的模型：以Train_ValData的特征为输入，以val_predict为因变量训练第二层的模型；

  <font color="red"> 至此，模型训练完成</font>
  <font color="red"> 接下来是模型预测</font>

- (6) 模型预测：用TestData走一遍第一层模型，得到test_predict1，再用test_predict1作为输入走一遍第二层模型进行预测，该结果为整个测试集的结果。

Blending集成方式的优劣：

- 优点：实现简单粗暴，没有太多的理论的分析。
- 缺点：只使用了一部分数据集作为留出集进行验证，也就是只能用上数据中的一部分，实际上这对数据来说是很奢侈浪费的。


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from itertools import product
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris

data = load_iris()
iris_data = data.data
x = pd.DataFrame(data.data, columns=data.feature_names)[['sepal length (cm)', 'sepal width (cm)']]
x = data.data[:, :2]
y = data.target
# 创建训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)
# 创建训练集和验证集
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=1)

x_min = x_train[:, 0].min() - 1
x_max = x_train[:, 0].max() + 1
y_min = x_train[:, 1].min() - 1
y_max = x_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# blending
# 第一分类器
# clfs = [SVC(probability=True), RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),KNeighborsClassifier()]
clfs = [SVC(probability=True), RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='gini')]
# 设置第二层分类器
# dtc = DecisionTreeClassifier(max_depth=5)
knc=KNeighborsClassifier()
# 输出第一层的验证集结果与测试集结果
val_features = np.zeros((x_val.shape[0], len(clfs)))  # 初始化验证集结果
test_features = np.zeros((x_test.shape[0], len(clfs)))  # 初始化测试集结果
for i, clf in enumerate(clfs):
    clf.fit(x_train, y_train)
    val_feature = clf.predict_proba(x_val)[:, 1]
    test_feature = clf.predict_proba(x_test)[:, 1]
    val_features[:, i] = val_feature
    test_features[:, i] = test_feature
# 将第一层的验证集的结果输入第二层训练第二层分类器
knc.fit(val_features, y_val)
Z = knc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
colors = ListedColormap(['#AAAAFF', '#AAFFAA', '#FFAAAA'])
plt.figure()
plt.contourf(xx, yy, Z, cmap=colors, alpha=0.3)
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
```

