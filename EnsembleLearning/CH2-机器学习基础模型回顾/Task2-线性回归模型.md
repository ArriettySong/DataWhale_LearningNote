参考资料：

拓展学习资料：线性回归模型123
https://zhuanlan.zhihu.com/p/109808497 
https://zhuanlan.zhihu.com/p/109815760
https://zhuanlan.zhihu.com/p/109819252
R语言实战之回归分析https://zhuanlan.zhihu.com/p/184923047 
线性回归https://zhuanlan.zhihu.com/p/49480391

# 2. 使用sklearn构建完整的机器学习项目流程

一般来说，一个完整的机器学习项目分为以下步骤：
   1. 明确解决问题的模型类型：<font color=red>回归/分类</font>
   2. 收集数据集并选择合适的<font color=red>特征</font>。
   3. 选择度量模型性能的<font color=red>指标</font>。
   4. 选择具体的<font color=red>模型</font>并进行训练以优化模型。
   5. <font color=red>评估</font>模型的性能并调参。

## 2.1 使用sklearn构建完整的回归项目

   ### 2.1.1 明确解决问题的模型类型：<font color=red>回归/分类</font>
   本次实践项目为回归项目。    
   ### 2.1.2 收集数据集并选择合适的<font color=red>特征</font>
   数据集上我们使用上一节尝试过的Boston房价数据集。    


```python
# 引入相关科学计算包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline 
plt.style.use("ggplot")      # 样式美化
import seaborn as sns
from sklearn import datasets
boston = datasets.load_boston()     # 返回一个类似于字典的类
X = boston.data
y = boston.target
features = boston.feature_names
boston_data = pd.DataFrame(X,columns=features)
boston_data["Price"] = y
boston_data.head()
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>


各个特征的相关解释：
   - CRIM：各城镇的人均犯罪率
   - ZN：规划地段超过25,000平方英尺的住宅用地比例
   - INDUS：城镇非零售商业用地比例
   - CHAS：是否在查尔斯河边(=1是)
   - NOX：一氧化氮浓度(/千万分之一)
   - RM：每个住宅的平均房间数
   - AGE：1940年以前建造的自住房屋的比例
   - DIS：到波士顿五个就业中心的加权距离
   - RAD：放射状公路的可达性指数
   - TAX：全部价值的房产税率(每1万美元)
   - PTRATIO：按城镇分配的学生与教师比例
   - B：1000(Bk - 0.63)^2其中Bk是每个城镇的黑人比例
   - LSTAT：较低地位人口
   - Price：房价

  ### 2.1.3 选择度量模型性能的<font color=red>指标</font>
   - MSE均方误差： $\text{MSE}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (y_i - \hat{y}_i)^2.$
   - MAE平均绝对误差:$\text{MAE}(y, \hat{y}) = \frac{1}{n_{\text{samples}}} \sum_{i=0}^{n_{\text{samples}}-1} \left| y_i - \hat{y}_i \right|$
   - $R^2$决定系数：$R^2(y, \hat{y}) = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$
   - 解释方差得分:$explained\_{}variance(y, \hat{y}) = 1 - \frac{Var\{ y - \hat{y}\}}{Var\{y\}}$

https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics

![jupyter](./1.3.png)              
在这个案例中，我们使用<font color=red>MSE均方误差</font>>为模型的性能度量指标。



   ### 2.1.4 选择具体的<font color=red>模型</font>并进行训练以优化模型。

#### 1. 线性回归模型（[sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)）

##### 1.1 最小二乘估计

##### 1.2 几何视角

##### 1.3 概率视角

下面，我们使用sklearn的线性回归实例来演示：                   
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression


```python
from sklearn import linear_model      # 引入线性回归方法
lin_reg = linear_model.LinearRegression()       # 创建线性回归的类
lin_reg.fit(X,y)        # 输入特征X和因变量y进行训练
print("模型系数：",lin_reg.coef_)             # 输出模型的系数
print("模型得分：",lin_reg.score(X,y))    # 输出模型的决定系数R^2
```

    模型系数： [-1.08011358e-01  4.64204584e-02  2.05586264e-02  2.68673382e+00
     -1.77666112e+01  3.80986521e+00  6.92224640e-04 -1.47556685e+00
      3.06049479e-01 -1.23345939e-02 -9.52747232e-01  9.31168327e-03
     -5.24758378e-01]
    模型得分： 0.7406426641094095



#### 2. 线性回归的推广

##### 2.1 多项式回归（[sklearn.preprocessing.PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html?highlight=poly#sklearn.preprocessing.PolynomialFeatures)）

##### 2.2 广义可加模型GAM（[`pyGam`](https://github.com/dswah/pyGAM/blob/master/doc/source/notebooks/quick_start.ipynb)）

多项式回归实例介绍：      

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html?highlight=poly#sklearn.preprocessing.PolynomialFeatures     

sklearn.preprocessing.PolynomialFeatures(degree=2, *, interaction_only=False, include_bias=True, order='C'):               

   - 参数：

     degree：特征转换的阶数。  

     interaction_onlyboolean：是否只包含交互项，默认False 。

     include_bias：是否包含截距项，默认True。

     order：str in {‘C’, ‘F’}, default ‘C’，输出数组的顺序。    


```python
from sklearn.preprocessing import PolynomialFeatures
X_arr = np.arange(12).reshape(3, 4)
print("原始X为：\n",X_arr)

poly = PolynomialFeatures(2)
print("2次转化X：\n",poly.fit_transform(X_arr))

poly = PolynomialFeatures(interaction_only=True)
print("2次转化X：\n",poly.fit_transform(X_arr))
```

    原始X为：
     [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    2次转化X：
     [[  1.   0.   1.   2.   3.   0.   0.   0.   0.   1.   2.   3.   4.   6.
        9.]
     [  1.   4.   5.   6.   7.  16.  20.  24.  28.  25.  30.  35.  36.  42.
       49.]
     [  1.   8.   9.  10.  11.  64.  72.  80.  88.  81.  90.  99. 100. 110.
      121.]]
    2次转化X：
     [[  1.   0.   1.   2.   3.   0.   0.   0.   2.   3.   6.]
     [  1.   4.   5.   6.   7.  20.  24.  28.  30.  35.  42.]
     [  1.   8.   9.  10.  11.  72.  80.  88.  90.  99. 110.]]

(b) GAM模型实例介绍：          
安装pygam：pip install pygam               
https://github.com/dswah/pyGAM/blob/master/doc/source/notebooks/quick_start.ipynb                     


```python
from pygam import LinearGAM
gam = LinearGAM().fit(boston_data[boston.feature_names], y)
gam.summary()
```

    LinearGAM                                                                                                 
    =============================================== ==========================================================
    Distribution:                        NormalDist Effective DoF:                                    103.2423
    Link Function:                     IdentityLink Log Likelihood:                                 -1589.7653
    Number of Samples:                          506 AIC:                                             3388.0152
                                                    AICc:                                            3442.7649
                                                    GCV:                                               13.7683
                                                    Scale:                                              8.8269
                                                    Pseudo R-Squared:                                   0.9168
    ==========================================================================================================
    Feature Function                  Lambda               Rank         EDoF         P > x        Sig. Code   
    ================================= ==================== ============ ============ ============ ============
    s(0)                              [0.6]                20           11.1         2.20e-11     ***         
    s(1)                              [0.6]                20           12.8         8.15e-02     .           
    s(2)                              [0.6]                20           13.5         2.59e-03     **          
    s(3)                              [0.6]                20           3.8          2.76e-01                 
    s(4)                              [0.6]                20           11.4         1.11e-16     ***         
    s(5)                              [0.6]                20           10.1         1.11e-16     ***         
    s(6)                              [0.6]                20           10.4         8.22e-01                 
    s(7)                              [0.6]                20           8.5          4.44e-16     ***         
    s(8)                              [0.6]                20           3.5          5.96e-03     **          
    s(9)                              [0.6]                20           3.4          1.33e-09     ***         
    s(10)                             [0.6]                20           1.8          3.26e-03     **          
    s(11)                             [0.6]                20           6.4          6.25e-02     .           
    s(12)                             [0.6]                20           6.5          1.11e-16     ***         
    intercept                                              1            0.0          2.23e-13     ***         
    ==========================================================================================================
    Significance codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    
    WARNING: Fitting splines and a linear function to a feature introduces a model identifiability problem
             which can cause p-values to appear significant when they are not.
    
    WARNING: p-values calculated in this manner behave correctly for un-penalized models or models with
             known smoothing parameters, but when smoothing parameters have been estimated, the p-values
             are typically lower than they should be, meaning that the tests reject the null too readily.



#### 3. 回归树（[sklearn.tree.DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html?highlight=tree#sklearn.tree.DecisionTreeRegressor)）

回归树与线性模型的比较：              
线性模型的模型形式与树模型的模型形式有着本质的区别，具体而言，线性回归对模型形式做了如下假定：$f(x) = w_0 + \sum\limits_{j=1}^{p}w_jx^{(j)}$，而回归树则是$f(x) = \sum\limits_{m=1}^{J}\hat{c}_mI(x \in R_m)$。那问题来了，哪种模型更优呢？这个要视具体情况而言，如果特征变量与因变量的关系能很好的用线性关系来表达，那么线性回归通常有着不错的预测效果，拟合效果则优于不能揭示线性结构的回归树。反之，如果特征变量与因变量的关系呈现高度复杂的非线性，那么树方法比传统方法更优。                     
![jupyter](./1.9.1.png)                        
树模型的优缺点：                 

- 树模型的解释性强，在解释性方面可能比线性回归还要方便。
- 树模型更接近人的决策方式。
- 树模型可以用图来表示，非专业人士也可以轻松解读。
- 树模型可以直接做定性的特征而不需要像线性回归一样哑元化。
- 树模型能很好处理缺失值和异常值，对异常值不敏感，但是这个对线性模型来说却是致命的。
- 树模型的预测准确性一般无法达到其他回归模型的水平，但是改进的方法很多。



sklearn使用回归树的实例：https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html?highlight=tree#sklearn.tree.DecisionTreeRegressor 

sklearn.tree.DecisionTreeRegressor(*, criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort='deprecated', ccp_alpha=0.0）  

   - 参数：(列举几个重要的，常用的，详情请看上面的官网)  

     criterion：{“ mse”，“ friedman_mse”，“ mae”}，默认=“ mse”。衡量分割标准的函数 。

     splitter：{“best”, “random”}, default=”best”。分割方式。

     max_depth：树的最大深度。

     min_samples_split：拆分内部节点所需的最少样本数，默认是2。 

     min_samples_leaf：在叶节点处需要的最小样本数。默认是1。

     min_weight_fraction_leaf：在所有叶节点处（所有输入样本）的权重总和中的最小加权分数。如果未提供sample_weight，则样本的权重相等。默认是0。                      


```python
from sklearn.tree import DecisionTreeRegressor    
reg_tree = DecisionTreeRegressor(criterion = "mse",min_samples_leaf = 5)
reg_tree.fit(X,y)
reg_tree.score(X,y)
```


    0.9376307599929274


```python
from sklearn.tree import DecisionTreeRegressor    
reg_tree = DecisionTreeRegressor(criterion = "mse",max_depth=7)
reg_tree.fit(X,y)
reg_tree.score(X,y)
```


    0.9639364682827993

#### 4. 支撑向量机回归SVR（[sklearn.svm.SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html?highlight=svr#sklearn.svm.SVR)）

sklearn中使用SVR实例：

sklearn.svm.SVR(*, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1) 

https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html?highlight=svr#sklearn.svm.SVR                        

   - 参数：

     kernel：核函数，{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, 默认=’rbf’。(后面会详细介绍) 

     degree：多项式核函数的阶数。默认 = 3。

     C：正则化参数，默认=1.0。(后面会详细介绍) 

     epsilon：SVR模型允许的不计算误差的邻域大小。默认0.1。              



```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler     # 标准化数据
from sklearn.pipeline import make_pipeline   # 使用管道，把预处理和模型形成一个流程

reg_svr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
reg_svr.fit(X, y)
reg_svr.score(X,y)
```


    0.7024525421955277


```python
reg_svr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1))
reg_svr.fit(X, y)
reg_svr.score(X,y)
```


    0.7028285706092579

