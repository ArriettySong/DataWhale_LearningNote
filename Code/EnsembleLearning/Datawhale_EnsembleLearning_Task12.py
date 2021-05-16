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