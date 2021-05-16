

## 「幸福感预测」

### 背景介绍

幸福感预测，是一个数据挖掘类型的比赛，我们需要使用包括个体变量（性别、年龄、地域、职业、健康、婚姻与政治面貌等等）、家庭变量（父母、配偶、子女、家庭资本等等）、社会态度（公平、信用、公共服务等等）等139维度的信息来预测其对幸福感的影响。

数据来源于国家官方的《中国综合社会调查（CGSS）》文件中的调查结果中的数据，数据来源可靠。

更详细的背景介绍见DataWhale组队学习教程。

### 数据信息

特征139维，样本8000+，目标预测值为1，2，3，4，5，其中1代表幸福感最低，5代表幸福感最高。该问题为回归类问题。

### 评价指标

最终的评价指标为均方误差MSE，即：$Score = \frac{1}{n} \sum_1 ^n (y_i - y ^*)^2$

### 导入package


```python
import os
import time 
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error,mean_absolute_error, f1_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.ensemble import ExtraTreesRegressor as etr
from sklearn.linear_model import BayesianRidge as br
from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression as lr
from sklearn.linear_model import ElasticNet as en
from sklearn.kernel_ridge import KernelRidge as kr
from sklearn.model_selection import  KFold, StratifiedKFold,GroupKFold, RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import logging
import warnings

warnings.filterwarnings('ignore') #消除warning
```

### 导入数据集


```python
train = pd.read_csv("train.csv", parse_dates=['survey_time'],encoding='latin-1') 
test = pd.read_csv("test.csv", parse_dates=['survey_time'],encoding='latin-1') #latin-1向下兼容ASCII
train = train[train["happiness"]!=-8].reset_index(drop=True)
train_data_copy = train.copy() #删去"happiness" 为-8的行
target_col = "happiness" #目标列
target = train_data_copy[target_col]
del train_data_copy[target_col] #去除目标列

data = pd.concat([train_data_copy,test],axis=0,ignore_index=True)
```

### 查看数据的基本信息


```python
train.happiness.describe() #数据的基本信息
```




    count    7988.000000
    mean        3.867927
    std         0.818717
    min         1.000000
    25%         4.000000
    50%         4.000000
    75%         4.000000
    max         5.000000
    Name: happiness, dtype: float64



### 数据预处理

首先需要对于数据中的连续出现的负数值进行处理。由于数据中的负数值只有-1，-2，-3，-8这几种数值，所以它们进行分别的操作，实现代码如下：


```python
#make feature +5
#csv中有复数值：-1、-2、-3、-8，将他们视为有问题的特征，但是不删去
def getres1(row):
    return len([x for x in row.values if type(x)==int and x<0])

def getres2(row):
    return len([x for x in row.values if type(x)==int and x==-8])

def getres3(row):
    return len([x for x in row.values if type(x)==int and x==-1])

def getres4(row):
    return len([x for x in row.values if type(x)==int and x==-2])

def getres5(row):
    return len([x for x in row.values if type(x)==int and x==-3])

#检查数据
data['neg1'] = data[data.columns].apply(lambda row:getres1(row),axis=1)
data.loc[data['neg1']>20,'neg1'] = 20  #平滑处理,最多出现20次

data['neg2'] = data[data.columns].apply(lambda row:getres2(row),axis=1)
data['neg3'] = data[data.columns].apply(lambda row:getres3(row),axis=1)
data['neg4'] = data[data.columns].apply(lambda row:getres4(row),axis=1)
data['neg5'] = data[data.columns].apply(lambda row:getres5(row),axis=1)
```

填充缺失值，在这里我采取的方式是将缺失值补全，使用fillna(value)，其中value的数值根据具体的情况来确定。例如将大部分缺失信息认为是零，将家庭成员数认为是1，将家庭收入这个特征认为是66365，即所有家庭的收入平均值。部分实现代码如下：


```python
#填充缺失值 共25列 去掉4列 填充21列
#以下的列都是缺省的，视情况填补
data['work_status'] = data['work_status'].fillna(0)
data['work_yr'] = data['work_yr'].fillna(0)
data['work_manage'] = data['work_manage'].fillna(0)
data['work_type'] = data['work_type'].fillna(0)

data['edu_yr'] = data['edu_yr'].fillna(0)
data['edu_status'] = data['edu_status'].fillna(0)

data['s_work_type'] = data['s_work_type'].fillna(0)
data['s_work_status'] = data['s_work_status'].fillna(0)
data['s_political'] = data['s_political'].fillna(0)
data['s_hukou'] = data['s_hukou'].fillna(0)
data['s_income'] = data['s_income'].fillna(0)
data['s_birth'] = data['s_birth'].fillna(0)
data['s_edu'] = data['s_edu'].fillna(0)
data['s_work_exper'] = data['s_work_exper'].fillna(0)

data['minor_child'] = data['minor_child'].fillna(0)
data['marital_now'] = data['marital_now'].fillna(0)
data['marital_1st'] = data['marital_1st'].fillna(0)
data['social_neighbor']=data['social_neighbor'].fillna(0)
data['social_friend']=data['social_friend'].fillna(0)
data['hukou_loc']=data['hukou_loc'].fillna(1) #最少为1，表示户口
data['family_income']=data['family_income'].fillna(66365) #删除问题值后的平均值
```

除此之外，还有特殊格式的信息需要另外处理，比如与时间有关的信息，这里主要分为两部分进行处理：首先是将“连续”的年龄，进行分层处理，即划分年龄段，具体地在这里我们将年龄分为了6个区间。其次是计算具体的年龄，在Excel表格中，只有出生年月以及调查时间等信息，我们根据此计算出每一位调查者的真实年龄。具体实现代码如下：


```python
#144+1 =145
#继续进行特殊的列进行数据处理
#读happiness_index.xlsx
data['survey_time'] = pd.to_datetime(data['survey_time'], format='%Y-%m-%d',errors='coerce')#防止时间格式不同的报错errors='coerce‘
data['survey_time'] = data['survey_time'].dt.year #仅仅是year，方便计算年龄
data['age'] = data['survey_time']-data['birth']
# print(data['age'],data['survey_time'],data['birth'])
#年龄分层 145+1=146
bins = [0,17,26,34,50,63,100]
data['age_bin'] = pd.cut(data['age'], bins, labels=[0,1,2,3,4,5]) 
```

在这里因为家庭的收入是连续值，所以不能再使用取众数的方法进行处理，这里就直接使用了均值进行缺失值的补全。第三种方法是使用我们日常生活中的真实情况，例如“宗教信息”特征为负数的认为是“不信仰宗教”，并认为“参加宗教活动的频率”为1，即没有参加过宗教活动，主观的进行补全，这也是我在这一步骤中使用最多的一种方式。就像我自己填表一样，这里我全部都使用了我自己的想法进行缺省值的补全。


```python
#对‘宗教’处理
data.loc[data['religion']<0,'religion'] = 1 #1为不信仰宗教
data.loc[data['religion_freq']<0,'religion_freq'] = 1 #1为从来没有参加过
#对‘教育程度’处理
data.loc[data['edu']<0,'edu'] = 4 #初中
data.loc[data['edu_status']<0,'edu_status'] = 0
data.loc[data['edu_yr']<0,'edu_yr'] = 0
#对‘个人收入’处理
data.loc[data['income']<0,'income'] = 0 #认为无收入
#对‘政治面貌’处理
data.loc[data['political']<0,'political'] = 1 #认为是群众
#对体重处理
data.loc[(data['weight_jin']<=80)&(data['height_cm']>=160),'weight_jin']= data['weight_jin']*2
data.loc[data['weight_jin']<=60,'weight_jin']= data['weight_jin']*2  #个人的想法，哈哈哈，没有60斤的成年人吧
#对身高处理
data.loc[data['height_cm']<150,'height_cm'] = 150 #成年人的实际情况
#对‘健康’处理
data.loc[data['health']<0,'health'] = 4 #认为是比较健康
data.loc[data['health_problem']<0,'health_problem'] = 4
#对‘沮丧’处理
data.loc[data['depression']<0,'depression'] = 4 #一般人都是很少吧
#对‘媒体’处理
data.loc[data['media_1']<0,'media_1'] = 1 #都是从不
data.loc[data['media_2']<0,'media_2'] = 1
data.loc[data['media_3']<0,'media_3'] = 1
data.loc[data['media_4']<0,'media_4'] = 1
data.loc[data['media_5']<0,'media_5'] = 1
data.loc[data['media_6']<0,'media_6'] = 1
#对‘空闲活动’处理
data.loc[data['leisure_1']<0,'leisure_1'] = 1 #都是根据自己的想法
data.loc[data['leisure_2']<0,'leisure_2'] = 5
data.loc[data['leisure_3']<0,'leisure_3'] = 3
```

使用众数（代码中使用mode()来实现异常值的修正），由于这里的特征是空闲活动，所以采用众数对于缺失值进行处理比较合理。具体的代码参考如下：


```python
data.loc[data['leisure_4']<0,'leisure_4'] = data['leisure_4'].mode() #取众数
data.loc[data['leisure_5']<0,'leisure_5'] = data['leisure_5'].mode()
data.loc[data['leisure_6']<0,'leisure_6'] = data['leisure_6'].mode()
data.loc[data['leisure_7']<0,'leisure_7'] = data['leisure_7'].mode()
data.loc[data['leisure_8']<0,'leisure_8'] = data['leisure_8'].mode()
data.loc[data['leisure_9']<0,'leisure_9'] = data['leisure_9'].mode()
data.loc[data['leisure_10']<0,'leisure_10'] = data['leisure_10'].mode()
data.loc[data['leisure_11']<0,'leisure_11'] = data['leisure_11'].mode()
data.loc[data['leisure_12']<0,'leisure_12'] = data['leisure_12'].mode()
data.loc[data['socialize']<0,'socialize'] = 2 #很少
data.loc[data['relax']<0,'relax'] = 4 #经常
data.loc[data['learn']<0,'learn'] = 1 #从不，哈哈哈哈
#对‘社交’处理
data.loc[data['social_neighbor']<0,'social_neighbor'] = 0
data.loc[data['social_friend']<0,'social_friend'] = 0
data.loc[data['socia_outing']<0,'socia_outing'] = 1
data.loc[data['neighbor_familiarity']<0,'social_neighbor']= 4
#对‘社会公平性’处理
data.loc[data['equity']<0,'equity'] = 4
#对‘社会等级’处理
data.loc[data['class_10_before']<0,'class_10_before'] = 3
data.loc[data['class']<0,'class'] = 5
data.loc[data['class_10_after']<0,'class_10_after'] = 5
data.loc[data['class_14']<0,'class_14'] = 2
#对‘工作情况’处理
data.loc[data['work_status']<0,'work_status'] = 0
data.loc[data['work_yr']<0,'work_yr'] = 0
data.loc[data['work_manage']<0,'work_manage'] = 0
data.loc[data['work_type']<0,'work_type'] = 0
#对‘社会保障’处理
data.loc[data['insur_1']<0,'insur_1'] = 1
data.loc[data['insur_2']<0,'insur_2'] = 1
data.loc[data['insur_3']<0,'insur_3'] = 1
data.loc[data['insur_4']<0,'insur_4'] = 1
data.loc[data['insur_1']==0,'insur_1'] = 0
data.loc[data['insur_2']==0,'insur_2'] = 0
data.loc[data['insur_3']==0,'insur_3'] = 0
data.loc[data['insur_4']==0,'insur_4'] = 0
```

取均值进行缺失值的补全（代码实现为means()），在这里因为家庭的收入是连续值，所以不能再使用取众数的方法进行处理，这里就直接使用了均值进行缺失值的补全。具体的代码参考如下：


```python
#对家庭情况处理
family_income_mean = data['family_income'].mean()
data.loc[data['family_income']<0,'family_income'] = family_income_mean
data.loc[data['family_m']<0,'family_m'] = 2
data.loc[data['family_status']<0,'family_status'] = 3
data.loc[data['house']<0,'house'] = 1
data.loc[data['car']<0,'car'] = 0
data.loc[data['car']==2,'car'] = 0
data.loc[data['son']<0,'son'] = 1
data.loc[data['daughter']<0,'daughter'] = 0
data.loc[data['minor_child']<0,'minor_child'] = 0
#对‘婚姻’处理
data.loc[data['marital_1st']<0,'marital_1st'] = 0
data.loc[data['marital_now']<0,'marital_now'] = 0
#对‘配偶’处理
data.loc[data['s_birth']<0,'s_birth'] = 0
data.loc[data['s_edu']<0,'s_edu'] = 0
data.loc[data['s_political']<0,'s_political'] = 0
data.loc[data['s_hukou']<0,'s_hukou'] = 0
data.loc[data['s_income']<0,'s_income'] = 0
data.loc[data['s_work_type']<0,'s_work_type'] = 0
data.loc[data['s_work_status']<0,'s_work_status'] = 0
data.loc[data['s_work_exper']<0,'s_work_exper'] = 0
#对‘父母情况’处理
data.loc[data['f_birth']<0,'f_birth'] = 1945
data.loc[data['f_edu']<0,'f_edu'] = 1
data.loc[data['f_political']<0,'f_political'] = 1
data.loc[data['f_work_14']<0,'f_work_14'] = 2
data.loc[data['m_birth']<0,'m_birth'] = 1940
data.loc[data['m_edu']<0,'m_edu'] = 1
data.loc[data['m_political']<0,'m_political'] = 1
data.loc[data['m_work_14']<0,'m_work_14'] = 2
#和同龄人相比社会经济地位
data.loc[data['status_peer']<0,'status_peer'] = 2
#和3年前比社会经济地位
data.loc[data['status_3_before']<0,'status_3_before'] = 2
#对‘观点’处理
data.loc[data['view']<0,'view'] = 4
#对期望年收入处理
data.loc[data['inc_ability']<=0,'inc_ability']= 2
inc_exp_mean = data['inc_exp'].mean()
data.loc[data['inc_exp']<=0,'inc_exp']= inc_exp_mean #取均值

#部分特征处理，取众数
for i in range(1,9+1):
    data.loc[data['public_service_'+str(i)]<0,'public_service_'+str(i)] = data['public_service_'+str(i)].dropna().mode().values
for i in range(1,13+1):
    data.loc[data['trust_'+str(i)]<0,'trust_'+str(i)] = data['trust_'+str(i)].dropna().mode().values
```

### 数据增广

这一步，我们需要进一步分析每一个特征之间的关系，从而进行数据增广。经过思考，这里我添加了如下的特征：第一次结婚年龄、最近结婚年龄、是否再婚、配偶年龄、配偶年龄差、各种收入比（与配偶之间的收入比、十年后预期收入与现在收入之比等等）、收入与住房面积比（其中也包括10年后期望收入等等各种情况）、社会阶级（10年后的社会阶级、14年后的社会阶级等等）、悠闲指数、满意指数、信任指数等等。除此之外，我还考虑了对于同一省、市、县进行了归一化。例如同一省市内的收入的平均值等以及一个个体相对于同省、市、县其他人的各个指标的情况。同时也考虑了对于同龄人之间的相互比较，即在同龄人中的收入情况、健康情况等等。具体的实现代码如下：


```python
#第一次结婚年龄 147
data['marital_1stbir'] = data['marital_1st'] - data['birth'] 
#最近结婚年龄 148
data['marital_nowtbir'] = data['marital_now'] - data['birth'] 
#是否再婚 149
data['mar'] = data['marital_nowtbir'] - data['marital_1stbir']
#配偶年龄 150
data['marital_sbir'] = data['marital_now']-data['s_birth']
#配偶年龄差 151
data['age_'] = data['marital_nowtbir'] - data['marital_sbir'] 

#收入比 151+7 =158
data['income/s_income'] = data['income']/(data['s_income']+1)
data['income+s_income'] = data['income']+(data['s_income']+1)
data['income/family_income'] = data['income']/(data['family_income']+1)
data['all_income/family_income'] = (data['income']+data['s_income'])/(data['family_income']+1)
data['income/inc_exp'] = data['income']/(data['inc_exp']+1)
data['family_income/m'] = data['family_income']/(data['family_m']+0.01)
data['income/m'] = data['income']/(data['family_m']+0.01)

#收入/面积比 158+4=162
data['income/floor_area'] = data['income']/(data['floor_area']+0.01)
data['all_income/floor_area'] = (data['income']+data['s_income'])/(data['floor_area']+0.01)
data['family_income/floor_area'] = data['family_income']/(data['floor_area']+0.01)
data['floor_area/m'] = data['floor_area']/(data['family_m']+0.01)

#class 162+3=165
data['class_10_diff'] = (data['class_10_after'] - data['class'])
data['class_diff'] = data['class'] - data['class_10_before']
data['class_14_diff'] = data['class'] - data['class_14']
#悠闲指数 166
leisure_fea_lis = ['leisure_'+str(i) for i in range(1,13)]
data['leisure_sum'] = data[leisure_fea_lis].sum(axis=1) #skew
#满意指数 167
public_service_fea_lis = ['public_service_'+str(i) for i in range(1,10)]
data['public_service_sum'] = data[public_service_fea_lis].sum(axis=1) #skew

#信任指数 168
trust_fea_lis = ['trust_'+str(i) for i in range(1,14)]
data['trust_sum'] = data[trust_fea_lis].sum(axis=1) #skew

#province mean 168+13=181
data['province_income_mean'] = data.groupby(['province'])['income'].transform('mean').values
data['province_family_income_mean'] = data.groupby(['province'])['family_income'].transform('mean').values
data['province_equity_mean'] = data.groupby(['province'])['equity'].transform('mean').values
data['province_depression_mean'] = data.groupby(['province'])['depression'].transform('mean').values
data['province_floor_area_mean'] = data.groupby(['province'])['floor_area'].transform('mean').values
data['province_health_mean'] = data.groupby(['province'])['health'].transform('mean').values
data['province_class_10_diff_mean'] = data.groupby(['province'])['class_10_diff'].transform('mean').values
data['province_class_mean'] = data.groupby(['province'])['class'].transform('mean').values
data['province_health_problem_mean'] = data.groupby(['province'])['health_problem'].transform('mean').values
data['province_family_status_mean'] = data.groupby(['province'])['family_status'].transform('mean').values
data['province_leisure_sum_mean'] = data.groupby(['province'])['leisure_sum'].transform('mean').values
data['province_public_service_sum_mean'] = data.groupby(['province'])['public_service_sum'].transform('mean').values
data['province_trust_sum_mean'] = data.groupby(['province'])['trust_sum'].transform('mean').values

#city   mean 181+13=194
data['city_income_mean'] = data.groupby(['city'])['income'].transform('mean').values
data['city_family_income_mean'] = data.groupby(['city'])['family_income'].transform('mean').values
data['city_equity_mean'] = data.groupby(['city'])['equity'].transform('mean').values
data['city_depression_mean'] = data.groupby(['city'])['depression'].transform('mean').values
data['city_floor_area_mean'] = data.groupby(['city'])['floor_area'].transform('mean').values
data['city_health_mean'] = data.groupby(['city'])['health'].transform('mean').values
data['city_class_10_diff_mean'] = data.groupby(['city'])['class_10_diff'].transform('mean').values
data['city_class_mean'] = data.groupby(['city'])['class'].transform('mean').values
data['city_health_problem_mean'] = data.groupby(['city'])['health_problem'].transform('mean').values
data['city_family_status_mean'] = data.groupby(['city'])['family_status'].transform('mean').values
data['city_leisure_sum_mean'] = data.groupby(['city'])['leisure_sum'].transform('mean').values
data['city_public_service_sum_mean'] = data.groupby(['city'])['public_service_sum'].transform('mean').values
data['city_trust_sum_mean'] = data.groupby(['city'])['trust_sum'].transform('mean').values

#county  mean 194 + 13 = 207
data['county_income_mean'] = data.groupby(['county'])['income'].transform('mean').values
data['county_family_income_mean'] = data.groupby(['county'])['family_income'].transform('mean').values
data['county_equity_mean'] = data.groupby(['county'])['equity'].transform('mean').values
data['county_depression_mean'] = data.groupby(['county'])['depression'].transform('mean').values
data['county_floor_area_mean'] = data.groupby(['county'])['floor_area'].transform('mean').values
data['county_health_mean'] = data.groupby(['county'])['health'].transform('mean').values
data['county_class_10_diff_mean'] = data.groupby(['county'])['class_10_diff'].transform('mean').values
data['county_class_mean'] = data.groupby(['county'])['class'].transform('mean').values
data['county_health_problem_mean'] = data.groupby(['county'])['health_problem'].transform('mean').values
data['county_family_status_mean'] = data.groupby(['county'])['family_status'].transform('mean').values
data['county_leisure_sum_mean'] = data.groupby(['county'])['leisure_sum'].transform('mean').values
data['county_public_service_sum_mean'] = data.groupby(['county'])['public_service_sum'].transform('mean').values
data['county_trust_sum_mean'] = data.groupby(['county'])['trust_sum'].transform('mean').values

#ratio 相比同省 207 + 13 =220
data['income/province'] = data['income']/(data['province_income_mean'])                                      
data['family_income/province'] = data['family_income']/(data['province_family_income_mean'])   
data['equity/province'] = data['equity']/(data['province_equity_mean'])       
data['depression/province'] = data['depression']/(data['province_depression_mean'])                                                
data['floor_area/province'] = data['floor_area']/(data['province_floor_area_mean'])
data['health/province'] = data['health']/(data['province_health_mean'])
data['class_10_diff/province'] = data['class_10_diff']/(data['province_class_10_diff_mean'])
data['class/province'] = data['class']/(data['province_class_mean'])
data['health_problem/province'] = data['health_problem']/(data['province_health_problem_mean'])
data['family_status/province'] = data['family_status']/(data['province_family_status_mean'])
data['leisure_sum/province'] = data['leisure_sum']/(data['province_leisure_sum_mean'])
data['public_service_sum/province'] = data['public_service_sum']/(data['province_public_service_sum_mean'])
data['trust_sum/province'] = data['trust_sum']/(data['province_trust_sum_mean']+1)

#ratio 相比同市 220 + 13 =233
data['income/city'] = data['income']/(data['city_income_mean'])                                      
data['family_income/city'] = data['family_income']/(data['city_family_income_mean'])   
data['equity/city'] = data['equity']/(data['city_equity_mean'])       
data['depression/city'] = data['depression']/(data['city_depression_mean'])                                                
data['floor_area/city'] = data['floor_area']/(data['city_floor_area_mean'])
data['health/city'] = data['health']/(data['city_health_mean'])
data['class_10_diff/city'] = data['class_10_diff']/(data['city_class_10_diff_mean'])
data['class/city'] = data['class']/(data['city_class_mean'])
data['health_problem/city'] = data['health_problem']/(data['city_health_problem_mean'])
data['family_status/city'] = data['family_status']/(data['city_family_status_mean'])
data['leisure_sum/city'] = data['leisure_sum']/(data['city_leisure_sum_mean'])
data['public_service_sum/city'] = data['public_service_sum']/(data['city_public_service_sum_mean'])
data['trust_sum/city'] = data['trust_sum']/(data['city_trust_sum_mean'])

#ratio 相比同个地区 233 + 13 =246
data['income/county'] = data['income']/(data['county_income_mean'])                                      
data['family_income/county'] = data['family_income']/(data['county_family_income_mean'])   
data['equity/county'] = data['equity']/(data['county_equity_mean'])       
data['depression/county'] = data['depression']/(data['county_depression_mean'])                                                
data['floor_area/county'] = data['floor_area']/(data['county_floor_area_mean'])
data['health/county'] = data['health']/(data['county_health_mean'])
data['class_10_diff/county'] = data['class_10_diff']/(data['county_class_10_diff_mean'])
data['class/county'] = data['class']/(data['county_class_mean'])
data['health_problem/county'] = data['health_problem']/(data['county_health_problem_mean'])
data['family_status/county'] = data['family_status']/(data['county_family_status_mean'])
data['leisure_sum/county'] = data['leisure_sum']/(data['county_leisure_sum_mean'])
data['public_service_sum/county'] = data['public_service_sum']/(data['county_public_service_sum_mean'])
data['trust_sum/county'] = data['trust_sum']/(data['county_trust_sum_mean'])

#age   mean 246+ 13 =259
data['age_income_mean'] = data.groupby(['age'])['income'].transform('mean').values
data['age_family_income_mean'] = data.groupby(['age'])['family_income'].transform('mean').values
data['age_equity_mean'] = data.groupby(['age'])['equity'].transform('mean').values
data['age_depression_mean'] = data.groupby(['age'])['depression'].transform('mean').values
data['age_floor_area_mean'] = data.groupby(['age'])['floor_area'].transform('mean').values
data['age_health_mean'] = data.groupby(['age'])['health'].transform('mean').values
data['age_class_10_diff_mean'] = data.groupby(['age'])['class_10_diff'].transform('mean').values
data['age_class_mean'] = data.groupby(['age'])['class'].transform('mean').values
data['age_health_problem_mean'] = data.groupby(['age'])['health_problem'].transform('mean').values
data['age_family_status_mean'] = data.groupby(['age'])['family_status'].transform('mean').values
data['age_leisure_sum_mean'] = data.groupby(['age'])['leisure_sum'].transform('mean').values
data['age_public_service_sum_mean'] = data.groupby(['age'])['public_service_sum'].transform('mean').values
data['age_trust_sum_mean'] = data.groupby(['age'])['trust_sum'].transform('mean').values

# 和同龄人相比259 + 13 =272
data['income/age'] = data['income']/(data['age_income_mean'])                                      
data['family_income/age'] = data['family_income']/(data['age_family_income_mean'])   
data['equity/age'] = data['equity']/(data['age_equity_mean'])       
data['depression/age'] = data['depression']/(data['age_depression_mean'])                                                
data['floor_area/age'] = data['floor_area']/(data['age_floor_area_mean'])
data['health/age'] = data['health']/(data['age_health_mean'])
data['class_10_diff/age'] = data['class_10_diff']/(data['age_class_10_diff_mean'])
data['class/age'] = data['class']/(data['age_class_mean'])
data['health_problem/age'] = data['health_problem']/(data['age_health_problem_mean'])
data['family_status/age'] = data['family_status']/(data['age_family_status_mean'])
data['leisure_sum/age'] = data['leisure_sum']/(data['age_leisure_sum_mean'])
data['public_service_sum/age'] = data['public_service_sum']/(data['age_public_service_sum_mean'])
data['trust_sum/age'] = data['trust_sum']/(data['age_trust_sum_mean'])
```

经过如上的操作后，最终我们的特征从一开始的131维，扩充为了272维的特征。接下来考虑特征工程、训练模型以及模型融合的工作。


```python
print('shape',data.shape)
data.head()
```

    shape (10956, 272)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>survey_type</th>
      <th>province</th>
      <th>city</th>
      <th>county</th>
      <th>survey_time</th>
      <th>gender</th>
      <th>birth</th>
      <th>nationality</th>
      <th>religion</th>
      <th>...</th>
      <th>depression/age</th>
      <th>floor_area/age</th>
      <th>health/age</th>
      <th>class_10_diff/age</th>
      <th>class/age</th>
      <th>health_problem/age</th>
      <th>family_status/age</th>
      <th>leisure_sum/age</th>
      <th>public_service_sum/age</th>
      <th>trust_sum/age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>12</td>
      <td>32</td>
      <td>59</td>
      <td>2015</td>
      <td>1</td>
      <td>1959</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1.285211</td>
      <td>0.410351</td>
      <td>0.848837</td>
      <td>0.000000</td>
      <td>0.683307</td>
      <td>0.521429</td>
      <td>0.733668</td>
      <td>0.724620</td>
      <td>0.666638</td>
      <td>0.925941</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>18</td>
      <td>52</td>
      <td>85</td>
      <td>2015</td>
      <td>1</td>
      <td>1992</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0.733333</td>
      <td>0.952824</td>
      <td>1.179337</td>
      <td>1.012552</td>
      <td>1.344444</td>
      <td>0.891344</td>
      <td>1.359551</td>
      <td>1.011792</td>
      <td>1.130778</td>
      <td>1.188442</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>29</td>
      <td>83</td>
      <td>126</td>
      <td>2015</td>
      <td>2</td>
      <td>1967</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1.343537</td>
      <td>0.972328</td>
      <td>1.150485</td>
      <td>1.190955</td>
      <td>1.195762</td>
      <td>1.055679</td>
      <td>1.190955</td>
      <td>0.966470</td>
      <td>1.193204</td>
      <td>0.803693</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>10</td>
      <td>28</td>
      <td>51</td>
      <td>2015</td>
      <td>2</td>
      <td>1943</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1.111663</td>
      <td>0.642329</td>
      <td>1.276353</td>
      <td>4.977778</td>
      <td>1.199143</td>
      <td>1.188329</td>
      <td>1.162630</td>
      <td>0.899346</td>
      <td>1.153810</td>
      <td>1.300950</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>7</td>
      <td>18</td>
      <td>36</td>
      <td>2015</td>
      <td>2</td>
      <td>1994</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0.750000</td>
      <td>0.587284</td>
      <td>1.177106</td>
      <td>0.000000</td>
      <td>0.236957</td>
      <td>1.116803</td>
      <td>1.093645</td>
      <td>1.045313</td>
      <td>0.728161</td>
      <td>1.117428</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 272 columns</p>
</div>



我们还应该删去有效样本数很少的特征，例如负值太多的特征或者是缺失值太多的特征，这里我一共删除了包括“目前的最高教育程度”在内的9类特征，得到了最终的263维的特征


```python
#272-9=263
#删除数值特别少的和之前用过的特征
del_list=['id','survey_time','edu_other','invest_other','property_other','join_party','province','city','county']
use_feature = [clo for clo in data.columns if clo not in del_list]
data.fillna(0,inplace=True) #还是补0
train_shape = train.shape[0] #一共的数据量，训练集
features = data[use_feature].columns #删除后所有的特征
X_train_263 = data[:train_shape][use_feature].values
y_train = target
X_test_263 = data[train_shape:][use_feature].values
X_train_263.shape #最终一种263个特征
```




    (7988, 263)



这里选择了最重要的49个特征，作为除了以上263维特征外的另外一组特征


```python
imp_fea_49 = ['equity','depression','health','class','family_status','health_problem','class_10_after',
           'equity/province','equity/city','equity/county',
           'depression/province','depression/city','depression/county',
           'health/province','health/city','health/county',
           'class/province','class/city','class/county',
           'family_status/province','family_status/city','family_status/county',
           'family_income/province','family_income/city','family_income/county',
           'floor_area/province','floor_area/city','floor_area/county',
           'leisure_sum/province','leisure_sum/city','leisure_sum/county',
           'public_service_sum/province','public_service_sum/city','public_service_sum/county',
           'trust_sum/province','trust_sum/city','trust_sum/county',
           'income/m','public_service_sum','class_diff','status_3_before','age_income_mean','age_floor_area_mean',
           'weight_jin','height_cm',
           'health/age','depression/age','equity/age','leisure_sum/age'
          ]
train_shape = train.shape[0]
X_train_49 = data[:train_shape][imp_fea_49].values
X_test_49 = data[train_shape:][imp_fea_49].values
X_train_49.shape #最重要的49个特征
```




    (7988, 49)



选择需要进行onehot编码的离散变量进行one-hot编码，再合成为第三类特征，共383维。


```python
cat_fea = ['survey_type','gender','nationality','edu_status','political','hukou','hukou_loc','work_exper','work_status','work_type',
           'work_manage','marital','s_political','s_hukou','s_work_exper','s_work_status','s_work_type','f_political','f_work_14',
           'm_political','m_work_14']
noc_fea = [clo for clo in use_feature if clo not in cat_fea]

onehot_data = data[cat_fea].values
enc = preprocessing.OneHotEncoder(categories = 'auto')
oh_data=enc.fit_transform(onehot_data).toarray()
oh_data.shape #变为onehot编码格式

X_train_oh = oh_data[:train_shape,:]
X_test_oh = oh_data[train_shape:,:]
X_train_oh.shape #其中的训练集

X_train_383 = np.column_stack([data[:train_shape][noc_fea].values,X_train_oh])#先是noc，再是cat_fea
X_test_383 = np.column_stack([data[train_shape:][noc_fea].values,X_test_oh])
X_train_383.shape
```




    (7988, 383)



基于此，我们构建完成了三种特征工程（训练数据集），其一是上面提取的最重要的49中特征，其中包括健康程度、社会阶级、在同龄人中的收入情况等等特征。其二是扩充后的263维特征（这里可以认为是初始特征）。其三是使用One-hot编码后的特征，这里要使用One-hot进行编码的原因在于，有部分特征为分离值，例如性别中男女，男为1，女为2，我们想使用One-hot将其变为男为0，女为1，来增强机器学习算法的鲁棒性能；再如民族这个特征，原本是1-56这56个数值，如果直接分类会让分类器的鲁棒性变差，所以使用One-hot编码将其变为6个特征进行非零即一的处理。

#### 特征建模

首先我们对于原始的263维的特征，使用lightGBM进行处理，这里我们使用5折交叉验证的方法：

1.lightGBM


```python
##### lgb_263 #
#lightGBM决策树
lgb_263_param = {
'num_leaves': 7, 
'min_data_in_leaf': 20, #叶子可能具有的最小记录数
'objective':'regression',
'max_depth': -1,
'learning_rate': 0.003,
"boosting": "gbdt", #用gbdt算法
"feature_fraction": 0.18, #例如 0.18时，意味着在每次迭代中随机选择18％的参数来建树
"bagging_freq": 1,
"bagging_fraction": 0.55, #每次迭代时用的数据比例
"bagging_seed": 14,
"metric": 'mse',
"lambda_l1": 0.1005,
"lambda_l2": 0.1996, 
"verbosity": -1}
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)   #交叉切分：5
oof_lgb_263 = np.zeros(len(X_train_263))
predictions_lgb_263 = np.zeros(len(X_test_263))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_263, y_train)):

    print("fold n°{}".format(fold_+1))
    trn_data = lgb.Dataset(X_train_263[trn_idx], y_train[trn_idx])
    val_data = lgb.Dataset(X_train_263[val_idx], y_train[val_idx])#train:val=4:1

    num_round = 10000
    lgb_263 = lgb.train(lgb_263_param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=500, early_stopping_rounds = 800)
    oof_lgb_263[val_idx] = lgb_263.predict(X_train_263[val_idx], num_iteration=lgb_263.best_iteration)
    predictions_lgb_263 += lgb_263.predict(X_test_263, num_iteration=lgb_263.best_iteration) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb_263, target)))
```

    fold n°1
    Training until validation scores don't improve for 800 rounds
    [500]	training's l2: 0.507058	valid_1's l2: 0.50963
    [1000]	training's l2: 0.458952	valid_1's l2: 0.469564
    [1500]	training's l2: 0.433197	valid_1's l2: 0.453234
    [2000]	training's l2: 0.415242	valid_1's l2: 0.444839
    [2500]	training's l2: 0.400993	valid_1's l2: 0.440086
    [3000]	training's l2: 0.388937	valid_1's l2: 0.436924
    [3500]	training's l2: 0.378101	valid_1's l2: 0.434974
    [4000]	training's l2: 0.368159	valid_1's l2: 0.43406
    [4500]	training's l2: 0.358827	valid_1's l2: 0.433151
    [5000]	training's l2: 0.350291	valid_1's l2: 0.432544
    [5500]	training's l2: 0.342368	valid_1's l2: 0.431821
    [6000]	training's l2: 0.334675	valid_1's l2: 0.431331
    [6500]	training's l2: 0.327275	valid_1's l2: 0.431014
    [7000]	training's l2: 0.320398	valid_1's l2: 0.431087
    [7500]	training's l2: 0.31352	valid_1's l2: 0.430819
    [8000]	training's l2: 0.307021	valid_1's l2: 0.430848
    [8500]	training's l2: 0.300811	valid_1's l2: 0.430688
    [9000]	training's l2: 0.294787	valid_1's l2: 0.430441
    [9500]	training's l2: 0.288993	valid_1's l2: 0.430433
    Early stopping, best iteration is:
    [9119]	training's l2: 0.293371	valid_1's l2: 0.430308
    fold n°2
    Training until validation scores don't improve for 800 rounds
    [500]	training's l2: 0.49895	valid_1's l2: 0.52945
    [1000]	training's l2: 0.450107	valid_1's l2: 0.496478
    [1500]	training's l2: 0.424394	valid_1's l2: 0.483286
    [2000]	training's l2: 0.40666	valid_1's l2: 0.476764
    [2500]	training's l2: 0.392432	valid_1's l2: 0.472668
    [3000]	training's l2: 0.380438	valid_1's l2: 0.470481
    [3500]	training's l2: 0.369872	valid_1's l2: 0.468919
    [4000]	training's l2: 0.36014	valid_1's l2: 0.467318
    [4500]	training's l2: 0.351175	valid_1's l2: 0.466438
    [5000]	training's l2: 0.342705	valid_1's l2: 0.466284
    [5500]	training's l2: 0.334778	valid_1's l2: 0.466151
    [6000]	training's l2: 0.3273	valid_1's l2: 0.466016
    [6500]	training's l2: 0.320121	valid_1's l2: 0.466013
    Early stopping, best iteration is:
    [5915]	training's l2: 0.328534	valid_1's l2: 0.465918
    fold n°3
    Training until validation scores don't improve for 800 rounds
    [500]	training's l2: 0.499658	valid_1's l2: 0.528985
    [1000]	training's l2: 0.450356	valid_1's l2: 0.497264
    [1500]	training's l2: 0.424109	valid_1's l2: 0.485403
    [2000]	training's l2: 0.405965	valid_1's l2: 0.479513
    [2500]	training's l2: 0.391747	valid_1's l2: 0.47646
    [3000]	training's l2: 0.379601	valid_1's l2: 0.474691
    [3500]	training's l2: 0.368915	valid_1's l2: 0.473648
    [4000]	training's l2: 0.359218	valid_1's l2: 0.47316
    [4500]	training's l2: 0.350338	valid_1's l2: 0.473043
    [5000]	training's l2: 0.341842	valid_1's l2: 0.472719
    [5500]	training's l2: 0.333851	valid_1's l2: 0.472779
    Early stopping, best iteration is:
    [4942]	training's l2: 0.342828	valid_1's l2: 0.472642
    fold n°4
    Training until validation scores don't improve for 800 rounds
    [500]	training's l2: 0.505224	valid_1's l2: 0.508238
    [1000]	training's l2: 0.456198	valid_1's l2: 0.473992
    [1500]	training's l2: 0.430167	valid_1's l2: 0.461419
    [2000]	training's l2: 0.412084	valid_1's l2: 0.454843
    [2500]	training's l2: 0.397714	valid_1's l2: 0.450999
    [3000]	training's l2: 0.385456	valid_1's l2: 0.448697
    [3500]	training's l2: 0.374527	valid_1's l2: 0.446993
    [4000]	training's l2: 0.364711	valid_1's l2: 0.44597
    [4500]	training's l2: 0.355626	valid_1's l2: 0.445132
    [5000]	training's l2: 0.347108	valid_1's l2: 0.44466
    [5500]	training's l2: 0.339146	valid_1's l2: 0.444226
    [6000]	training's l2: 0.331478	valid_1's l2: 0.443992
    [6500]	training's l2: 0.324231	valid_1's l2: 0.444014
    Early stopping, best iteration is:
    [5874]	training's l2: 0.333372	valid_1's l2: 0.443868
    fold n°5
    Training until validation scores don't improve for 800 rounds
    [500]	training's l2: 0.504304	valid_1's l2: 0.515256
    [1000]	training's l2: 0.456062	valid_1's l2: 0.478544
    [1500]	training's l2: 0.430298	valid_1's l2: 0.463847
    [2000]	training's l2: 0.412591	valid_1's l2: 0.456182
    [2500]	training's l2: 0.398635	valid_1's l2: 0.451783
    [3000]	training's l2: 0.386609	valid_1's l2: 0.449154
    [3500]	training's l2: 0.375948	valid_1's l2: 0.447265
    [4000]	training's l2: 0.366291	valid_1's l2: 0.445796
    [4500]	training's l2: 0.357236	valid_1's l2: 0.445098
    [5000]	training's l2: 0.348637	valid_1's l2: 0.444364
    [5500]	training's l2: 0.340736	valid_1's l2: 0.443998
    [6000]	training's l2: 0.333154	valid_1's l2: 0.443622
    [6500]	training's l2: 0.325783	valid_1's l2: 0.443226
    [7000]	training's l2: 0.318802	valid_1's l2: 0.442986
    [7500]	training's l2: 0.312164	valid_1's l2: 0.442928
    [8000]	training's l2: 0.305691	valid_1's l2: 0.442696
    [8500]	training's l2: 0.29935	valid_1's l2: 0.442521
    [9000]	training's l2: 0.293242	valid_1's l2: 0.442655
    Early stopping, best iteration is:
    [8594]	training's l2: 0.298201	valid_1's l2: 0.44238
    CV score: 0.45102656


接着，我使用已经训练完的lightGBM的模型进行特征重要性的判断以及可视化，从结果我们可以看出，排在重要性第一位的是health/age，就是同龄人中的健康程度，与我们主观的看法基本一致。


```python
#---------------特征重要性
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
df = pd.DataFrame(data[use_feature].columns.tolist(), columns=['feature'])
df['importance']=list(lgb_263.feature_importance())
df = df.sort_values(by='importance',ascending=False)
plt.figure(figsize=(14,28))
sns.barplot(x="importance", y="feature", data=df.head(50))
plt.title('Features importance (averaged/folds)')
plt.tight_layout()
```


​    
![png](%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0-%E6%A1%88%E4%BE%8B%E5%88%86%E6%9E%901_files/%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0-%E6%A1%88%E4%BE%8B%E5%88%86%E6%9E%901_33_0.png)
​    


后面，我们使用常见的机器学习方法，对于263维特征进行建模：

2.xgboost


```python
##### xgb_263
#xgboost
xgb_263_params = {'eta': 0.02,  #lr
              'max_depth': 6,  
              'min_child_weight':3,#最小叶子节点样本权重和
              'gamma':0, #指定节点分裂所需的最小损失函数下降值。
              'subsample': 0.7,  #控制对于每棵树，随机采样的比例
              'colsample_bytree': 0.3,  #用来控制每棵随机采样的列数的占比 (每一列是一个特征)。
              'lambda':2,
              'objective': 'reg:linear', 
              'eval_metric': 'rmse', 
              'silent': True, 
              'nthread': -1}


folds = KFold(n_splits=5, shuffle=True, random_state=2019)
oof_xgb_263 = np.zeros(len(X_train_263))
predictions_xgb_263 = np.zeros(len(X_test_263))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_263, y_train)):
    print("fold n°{}".format(fold_+1))
    trn_data = xgb.DMatrix(X_train_263[trn_idx], y_train[trn_idx])
    val_data = xgb.DMatrix(X_train_263[val_idx], y_train[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    xgb_263 = xgb.train(dtrain=trn_data, num_boost_round=3000, evals=watchlist, early_stopping_rounds=600, verbose_eval=500, params=xgb_263_params)
    oof_xgb_263[val_idx] = xgb_263.predict(xgb.DMatrix(X_train_263[val_idx]), ntree_limit=xgb_263.best_ntree_limit)
    predictions_xgb_263 += xgb_263.predict(xgb.DMatrix(X_test_263), ntree_limit=xgb_263.best_ntree_limit) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb_263, target)))
```

    fold n°1
    [19:14:55] WARNING: /Users/travis/build/dmlc/xgboost/src/objective/regression_obj.cu:170: reg:linear is now deprecated in favor of reg:squarederror.
    [19:14:55] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:480: 
    Parameters: { silent } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.


​    
    [0]	train-rmse:3.40426	valid_data-rmse:3.38329
    Multiple eval metrics have been passed: 'valid_data-rmse' will be used for early stopping.
    
    Will train until valid_data-rmse hasn't improved in 600 rounds.
    [500]	train-rmse:0.40805	valid_data-rmse:0.70588
    [1000]	train-rmse:0.27046	valid_data-rmse:0.70760
    Stopping. Best iteration:
    [663]	train-rmse:0.35644	valid_data-rmse:0.70521
    
    [19:15:46] WARNING: /Users/travis/build/dmlc/xgboost/src/objective/regression_obj.cu:170: reg:linear is now deprecated in favor of reg:squarederror.
    fold n°2
    [19:15:46] WARNING: /Users/travis/build/dmlc/xgboost/src/objective/regression_obj.cu:170: reg:linear is now deprecated in favor of reg:squarederror.
    [19:15:46] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:480: 
    Parameters: { silent } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.


​    
    [0]	train-rmse:3.39811	valid_data-rmse:3.40788
    Multiple eval metrics have been passed: 'valid_data-rmse' will be used for early stopping.
    
    Will train until valid_data-rmse hasn't improved in 600 rounds.
    [500]	train-rmse:0.40719	valid_data-rmse:0.69456
    [1000]	train-rmse:0.27402	valid_data-rmse:0.69501
    Stopping. Best iteration:
    [551]	train-rmse:0.39079	valid_data-rmse:0.69403
    
    [19:16:31] WARNING: /Users/travis/build/dmlc/xgboost/src/objective/regression_obj.cu:170: reg:linear is now deprecated in favor of reg:squarederror.
    fold n°3
    [19:16:31] WARNING: /Users/travis/build/dmlc/xgboost/src/objective/regression_obj.cu:170: reg:linear is now deprecated in favor of reg:squarederror.
    [19:16:31] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:480: 
    Parameters: { silent } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.


​    
    [0]	train-rmse:3.40181	valid_data-rmse:3.39295
    Multiple eval metrics have been passed: 'valid_data-rmse' will be used for early stopping.
    
    Will train until valid_data-rmse hasn't improved in 600 rounds.
    [500]	train-rmse:0.41334	valid_data-rmse:0.66250
    Stopping. Best iteration:
    [333]	train-rmse:0.47284	valid_data-rmse:0.66178
    
    [19:17:07] WARNING: /Users/travis/build/dmlc/xgboost/src/objective/regression_obj.cu:170: reg:linear is now deprecated in favor of reg:squarederror.
    fold n°4
    [19:17:08] WARNING: /Users/travis/build/dmlc/xgboost/src/objective/regression_obj.cu:170: reg:linear is now deprecated in favor of reg:squarederror.
    [19:17:08] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:480: 
    Parameters: { silent } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.


​    
    [0]	train-rmse:3.40240	valid_data-rmse:3.39012
    Multiple eval metrics have been passed: 'valid_data-rmse' will be used for early stopping.
    
    Will train until valid_data-rmse hasn't improved in 600 rounds.
    [500]	train-rmse:0.41021	valid_data-rmse:0.66575
    [1000]	train-rmse:0.27491	valid_data-rmse:0.66431
    Stopping. Best iteration:
    [863]	train-rmse:0.30689	valid_data-rmse:0.66358
    
    [19:18:06] WARNING: /Users/travis/build/dmlc/xgboost/src/objective/regression_obj.cu:170: reg:linear is now deprecated in favor of reg:squarederror.
    fold n°5
    [19:18:07] WARNING: /Users/travis/build/dmlc/xgboost/src/objective/regression_obj.cu:170: reg:linear is now deprecated in favor of reg:squarederror.
    [19:18:07] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:480: 
    Parameters: { silent } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.


​    
    [0]	train-rmse:3.39347	valid_data-rmse:3.42628
    Multiple eval metrics have been passed: 'valid_data-rmse' will be used for early stopping.
    
    Will train until valid_data-rmse hasn't improved in 600 rounds.
    [500]	train-rmse:0.41704	valid_data-rmse:0.64937
    [1000]	train-rmse:0.27907	valid_data-rmse:0.64914
    Stopping. Best iteration:
    [598]	train-rmse:0.38625	valid_data-rmse:0.64856
    
    [19:18:55] WARNING: /Users/travis/build/dmlc/xgboost/src/objective/regression_obj.cu:170: reg:linear is now deprecated in favor of reg:squarederror.
    CV score: 0.45559329


3. RandomForestRegressor随机森林


```python
#RandomForestRegressor随机森林
folds = KFold(n_splits=5, shuffle=True, random_state=2019)
oof_rfr_263 = np.zeros(len(X_train_263))
predictions_rfr_263 = np.zeros(len(X_test_263))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_263, y_train)):
    print("fold n°{}".format(fold_+1))
    tr_x = X_train_263[trn_idx]
    tr_y = y_train[trn_idx]
    rfr_263 = rfr(n_estimators=1600,max_depth=9, min_samples_leaf=9, min_weight_fraction_leaf=0.0,
            max_features=0.25,verbose=1,n_jobs=-1)
    #verbose = 0 为不在标准输出流输出日志信息
#verbose = 1 为输出进度条记录
#verbose = 2 为每个epoch输出一行记录
    rfr_263.fit(tr_x,tr_y)
    oof_rfr_263[val_idx] = rfr_263.predict(X_train_263[val_idx])
    
    predictions_rfr_263 += rfr_263.predict(X_test_263) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_rfr_263, target)))
```

    fold n°1


    [Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    0.6s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:    2.6s
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:    6.5s
    [Parallel(n_jobs=-1)]: Done 784 tasks      | elapsed:   11.8s
    [Parallel(n_jobs=-1)]: Done 1234 tasks      | elapsed:   18.9s
    [Parallel(n_jobs=-1)]: Done 1600 out of 1600 | elapsed:   25.6s finished
    [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 184 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 434 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 784 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 1234 tasks      | elapsed:    0.2s
    [Parallel(n_jobs=8)]: Done 1600 out of 1600 | elapsed:    0.2s finished
    [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 184 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 434 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 784 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 1234 tasks      | elapsed:    0.2s
    [Parallel(n_jobs=8)]: Done 1600 out of 1600 | elapsed:    0.2s finished


    fold n°2


    [Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    0.6s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:    2.8s
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:    6.9s
    [Parallel(n_jobs=-1)]: Done 784 tasks      | elapsed:   12.9s
    [Parallel(n_jobs=-1)]: Done 1234 tasks      | elapsed:   21.0s
    [Parallel(n_jobs=-1)]: Done 1600 out of 1600 | elapsed:   27.5s finished
    [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 184 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 434 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 784 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 1234 tasks      | elapsed:    0.2s
    [Parallel(n_jobs=8)]: Done 1600 out of 1600 | elapsed:    0.2s finished
    [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 184 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 434 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 784 tasks      | elapsed:    0.2s
    [Parallel(n_jobs=8)]: Done 1234 tasks      | elapsed:    0.2s
    [Parallel(n_jobs=8)]: Done 1600 out of 1600 | elapsed:    0.3s finished


    fold n°3


    [Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    0.6s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:    3.4s
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:    7.6s
    [Parallel(n_jobs=-1)]: Done 784 tasks      | elapsed:   13.7s
    [Parallel(n_jobs=-1)]: Done 1234 tasks      | elapsed:   21.0s
    [Parallel(n_jobs=-1)]: Done 1600 out of 1600 | elapsed:   26.9s finished
    [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 184 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 434 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 784 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 1234 tasks      | elapsed:    0.2s
    [Parallel(n_jobs=8)]: Done 1600 out of 1600 | elapsed:    0.2s finished
    [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 184 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 434 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 784 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 1234 tasks      | elapsed:    0.2s
    [Parallel(n_jobs=8)]: Done 1600 out of 1600 | elapsed:    0.2s finished


    fold n°4


    [Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    0.8s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:    3.5s
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:    7.9s
    [Parallel(n_jobs=-1)]: Done 784 tasks      | elapsed:   13.3s
    [Parallel(n_jobs=-1)]: Done 1234 tasks      | elapsed:   20.6s
    [Parallel(n_jobs=-1)]: Done 1600 out of 1600 | elapsed:   26.1s finished
    [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 184 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 434 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 784 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 1234 tasks      | elapsed:    0.2s
    [Parallel(n_jobs=8)]: Done 1600 out of 1600 | elapsed:    0.2s finished
    [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 184 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 434 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 784 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 1234 tasks      | elapsed:    0.2s
    [Parallel(n_jobs=8)]: Done 1600 out of 1600 | elapsed:    0.2s finished


    fold n°5


    [Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    0.6s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:    2.7s
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:    6.8s
    [Parallel(n_jobs=-1)]: Done 784 tasks      | elapsed:   12.2s
    [Parallel(n_jobs=-1)]: Done 1234 tasks      | elapsed:   19.2s
    [Parallel(n_jobs=-1)]: Done 1600 out of 1600 | elapsed:   25.1s finished
    [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 184 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 434 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 784 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 1234 tasks      | elapsed:    0.2s
    [Parallel(n_jobs=8)]: Done 1600 out of 1600 | elapsed:    0.2s finished
    [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 184 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 434 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 784 tasks      | elapsed:    0.1s


    CV score: 0.47804209


    [Parallel(n_jobs=8)]: Done 1234 tasks      | elapsed:    0.2s
    [Parallel(n_jobs=8)]: Done 1600 out of 1600 | elapsed:    0.3s finished


4. GradientBoostingRegressor梯度提升决策树


```python
#GradientBoostingRegressor梯度提升决策树
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
oof_gbr_263 = np.zeros(train_shape)
predictions_gbr_263 = np.zeros(len(X_test_263))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_263, y_train)):
    print("fold n°{}".format(fold_+1))
    tr_x = X_train_263[trn_idx]
    tr_y = y_train[trn_idx]
    gbr_263 = gbr(n_estimators=400, learning_rate=0.01,subsample=0.65,max_depth=7, min_samples_leaf=20,
            max_features=0.22,verbose=1)
    gbr_263.fit(tr_x,tr_y)
    oof_gbr_263[val_idx] = gbr_263.predict(X_train_263[val_idx])
    
    predictions_gbr_263 += gbr_263.predict(X_test_263) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_gbr_263, target)))
```

    fold n°1
          Iter       Train Loss      OOB Improve   Remaining Time 
             1           0.6419           0.0036           24.34s
             2           0.6564           0.0031           23.18s
             3           0.6693           0.0031           22.69s
             4           0.6589           0.0031           22.78s
             5           0.6522           0.0027           22.58s
             6           0.6521           0.0031           22.40s
             7           0.6370           0.0029           22.23s
             8           0.6343           0.0030           22.06s
             9           0.6447           0.0029           21.87s
            10           0.6397           0.0028           21.75s
            20           0.5955           0.0019           20.93s
            30           0.5695           0.0016           20.09s
            40           0.5460           0.0015           19.34s
            50           0.5121           0.0011           18.65s
            60           0.4994           0.0012           18.03s
            70           0.4912           0.0010           17.44s
            80           0.4719           0.0010           16.76s
            90           0.4310           0.0007           16.28s
           100           0.4437           0.0006           15.84s
           200           0.3424           0.0002           10.15s
           300           0.3063          -0.0000            4.94s
           400           0.2759          -0.0000            0.00s
    fold n°2
          Iter       Train Loss      OOB Improve   Remaining Time 
             1           0.6836           0.0034           24.61s
             2           0.6613           0.0030           22.86s
             3           0.6500           0.0031           24.11s
             4           0.6621           0.0036           23.15s
             5           0.6356           0.0031           23.49s
             6           0.6460           0.0029           23.13s
             7           0.6263           0.0032           22.83s
             8           0.6149           0.0029           22.72s
             9           0.6350           0.0030           22.83s
            10           0.6325           0.0026           22.65s
            20           0.6064           0.0025           21.62s
            30           0.5812           0.0018           20.59s
            40           0.5460           0.0018           19.98s
            50           0.5016           0.0014           19.52s
            60           0.4991           0.0010           18.84s
            70           0.4645           0.0009           18.24s
            80           0.4621           0.0007           17.76s
            90           0.4497           0.0007           17.20s
           100           0.4374           0.0005           16.51s
           200           0.3420           0.0001           10.35s
           300           0.3032          -0.0000            4.95s
           400           0.2710          -0.0000            0.00s
    fold n°3
          Iter       Train Loss      OOB Improve   Remaining Time 
             1           0.6692           0.0036           24.95s
             2           0.6468           0.0031           23.99s
             3           0.6313           0.0034           24.05s
             4           0.6499           0.0032           23.70s
             5           0.6358           0.0033           23.38s
             6           0.6343           0.0029           23.05s
             7           0.6312           0.0036           22.71s
             8           0.6180           0.0032           22.47s
             9           0.6275           0.0035           22.57s
            10           0.6168           0.0030           22.24s
            20           0.5792           0.0021           20.73s
            30           0.5583           0.0023           20.27s
            40           0.5521           0.0018           19.70s
            50           0.5067           0.0013           18.84s
            60           0.4754           0.0010           18.42s
            70           0.4811           0.0009           17.84s
            80           0.4603           0.0008           17.38s
            90           0.4439           0.0006           16.74s
           100           0.4323           0.0007           16.25s
           200           0.3401           0.0002           10.23s
           300           0.2862          -0.0000            4.84s
           400           0.2690          -0.0000            0.00s
    fold n°4
          Iter       Train Loss      OOB Improve   Remaining Time 
             1           0.6687           0.0032           21.09s
             2           0.6517           0.0031           23.29s
             3           0.6583           0.0031           23.63s
             4           0.6607           0.0033           24.45s
             5           0.6583           0.0029           24.78s
             6           0.6688           0.0028           24.80s
             7           0.6320           0.0030           25.08s
             8           0.6502           0.0026           24.94s
             9           0.6358           0.0026           24.51s
            10           0.6258           0.0027           24.24s
            20           0.5910           0.0023           22.41s
            30           0.5609           0.0020           21.31s
            40           0.5399           0.0017           20.50s
            50           0.4963           0.0013           19.67s
            60           0.4844           0.0012           18.86s
            70           0.4781           0.0008           18.21s
            80           0.4484           0.0010           17.63s
            90           0.4619           0.0006           16.95s
           100           0.4430           0.0005           16.46s
           200           0.3377           0.0001           10.50s
           300           0.3001           0.0001            4.97s
           400           0.2623          -0.0000            0.00s
    fold n°5
          Iter       Train Loss      OOB Improve   Remaining Time 
             1           0.6857           0.0031           23.50s
             2           0.6320           0.0035           24.26s
             3           0.6573           0.0033           23.41s
             4           0.6494           0.0033           24.20s
             5           0.6311           0.0033           24.32s
             6           0.6362           0.0031           24.20s
             7           0.6291           0.0032           24.05s
             8           0.6354           0.0032           23.56s
             9           0.6383           0.0030           23.54s
            10           0.6250           0.0029           23.64s
            20           0.5989           0.0023           21.45s
            30           0.5736           0.0019           20.27s
            40           0.5457           0.0016           19.60s
            50           0.5045           0.0015           18.76s
            60           0.4820           0.0012           18.20s
            70           0.4756           0.0010           17.44s
            80           0.4484           0.0009           16.91s
            90           0.4410           0.0007           16.34s
           100           0.4195           0.0004           15.72s
           200           0.3348           0.0001           10.05s
           300           0.2933          -0.0000            4.76s
           400           0.2658          -0.0000            0.00s
    CV score: 0.45583290


5. ExtraTreesRegressor 极端随机森林回归


```python
#ExtraTreesRegressor 极端随机森林回归
folds = KFold(n_splits=5, shuffle=True, random_state=13)
oof_etr_263 = np.zeros(train_shape)
predictions_etr_263 = np.zeros(len(X_test_263))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_263, y_train)):
    print("fold n°{}".format(fold_+1))
    tr_x = X_train_263[trn_idx]
    tr_y = y_train[trn_idx]
    etr_263 = etr(n_estimators=1000,max_depth=8, min_samples_leaf=12, min_weight_fraction_leaf=0.0,
            max_features=0.4,verbose=1,n_jobs=-1)
    etr_263.fit(tr_x,tr_y)
    oof_etr_263[val_idx] = etr_263.predict(X_train_263[val_idx])
    
    predictions_etr_263 += etr_263.predict(X_test_263) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_etr_263, target)))
```

    fold n°1


    [Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    0.4s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:    1.7s
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:    4.0s
    [Parallel(n_jobs=-1)]: Done 784 tasks      | elapsed:    7.2s
    [Parallel(n_jobs=-1)]: Done 1000 out of 1000 | elapsed:    9.0s finished
    [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 184 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 434 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 784 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 1000 out of 1000 | elapsed:    0.1s finished
    [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 184 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 434 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 784 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 1000 out of 1000 | elapsed:    0.1s finished


    fold n°2


    [Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    0.3s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:    1.6s
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:    3.8s
    [Parallel(n_jobs=-1)]: Done 784 tasks      | elapsed:    6.9s
    [Parallel(n_jobs=-1)]: Done 1000 out of 1000 | elapsed:    8.9s finished
    [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 184 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 434 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 784 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 1000 out of 1000 | elapsed:    0.1s finished
    [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 184 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 434 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 784 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 1000 out of 1000 | elapsed:    0.1s finished


    fold n°3


    [Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    0.4s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:    1.7s
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:    4.1s
    [Parallel(n_jobs=-1)]: Done 784 tasks      | elapsed:    7.6s
    [Parallel(n_jobs=-1)]: Done 1000 out of 1000 | elapsed:    9.6s finished
    [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 184 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 434 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 784 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 1000 out of 1000 | elapsed:    0.1s finished
    [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 184 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 434 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 784 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 1000 out of 1000 | elapsed:    0.1s finished


    fold n°4


    [Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    0.4s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:    1.7s
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:    4.0s
    [Parallel(n_jobs=-1)]: Done 784 tasks      | elapsed:    7.6s
    [Parallel(n_jobs=-1)]: Done 1000 out of 1000 | elapsed:   10.6s finished
    [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 184 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 434 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 784 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 1000 out of 1000 | elapsed:    0.2s finished
    [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 184 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 434 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 784 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 1000 out of 1000 | elapsed:    0.2s finished


    fold n°5


    [Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    0.4s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:    1.9s
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:    4.4s
    [Parallel(n_jobs=-1)]: Done 784 tasks      | elapsed:    8.6s
    [Parallel(n_jobs=-1)]: Done 1000 out of 1000 | elapsed:   10.7s finished
    [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 184 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 434 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 784 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 1000 out of 1000 | elapsed:    0.1s finished
    [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 184 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=8)]: Done 434 tasks      | elapsed:    0.1s


    CV score: 0.48598792


    [Parallel(n_jobs=8)]: Done 784 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=8)]: Done 1000 out of 1000 | elapsed:    0.1s finished


至此，我们得到了以上5种模型的预测结果以及模型架构及参数。其中在每一种特征工程中，进行5折的交叉验证，并重复两次（Kernel Ridge Regression，核脊回归），取得每一个特征数下的模型的结果。


```python
train_stack2 = np.vstack([oof_lgb_263,oof_xgb_263,oof_gbr_263,oof_rfr_263,oof_etr_263]).transpose()
# transpose()函数的作用就是调换x,y,z的位置,也就是数组的索引值
test_stack2 = np.vstack([predictions_lgb_263, predictions_xgb_263,predictions_gbr_263,predictions_rfr_263,predictions_etr_263]).transpose()

#交叉验证:5折，重复2次
folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=7)
oof_stack2 = np.zeros(train_stack2.shape[0])
predictions_lr2 = np.zeros(test_stack2.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack2,target)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack2[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack2[val_idx], target.iloc[val_idx].values
    #Kernel Ridge Regression
    lr2 = kr()
    lr2.fit(trn_data, trn_y)
    
    oof_stack2[val_idx] = lr2.predict(val_data)
    predictions_lr2 += lr2.predict(test_stack2) / 10
    
mean_squared_error(target.values, oof_stack2) 
```

    fold 0
    fold 1
    fold 2
    fold 3
    fold 4
    fold 5
    fold 6
    fold 7
    fold 8
    fold 9





    0.44815130114230267



接下来我们对于49维的数据进行与上述263维数据相同的操作

1.lightGBM


```python
##### lgb_49
lgb_49_param = {
'num_leaves': 9,
'min_data_in_leaf': 23,
'objective':'regression',
'max_depth': -1,
'learning_rate': 0.002,
"boosting": "gbdt",
"feature_fraction": 0.45,
"bagging_freq": 1,
"bagging_fraction": 0.65,
"bagging_seed": 15,
"metric": 'mse',
"lambda_l2": 0.2, 
"verbosity": -1}
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=9)   
oof_lgb_49 = np.zeros(len(X_train_49))
predictions_lgb_49 = np.zeros(len(X_test_49))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_49, y_train)):
    print("fold n°{}".format(fold_+1))
    trn_data = lgb.Dataset(X_train_49[trn_idx], y_train[trn_idx])
    val_data = lgb.Dataset(X_train_49[val_idx], y_train[val_idx])

    num_round = 12000
    lgb_49 = lgb.train(lgb_49_param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 1000)
    oof_lgb_49[val_idx] = lgb_49.predict(X_train_49[val_idx], num_iteration=lgb_49.best_iteration)
    predictions_lgb_49 += lgb_49.predict(X_test_49, num_iteration=lgb_49.best_iteration) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb_49, target)))
```

    fold n°1
    Training until validation scores don't improve for 1000 rounds
    [1000]	training's l2: 0.46958	valid_1's l2: 0.500767
    [2000]	training's l2: 0.429395	valid_1's l2: 0.482214
    [3000]	training's l2: 0.406748	valid_1's l2: 0.477959
    [4000]	training's l2: 0.388735	valid_1's l2: 0.476283
    [5000]	training's l2: 0.373399	valid_1's l2: 0.475506
    [6000]	training's l2: 0.359798	valid_1's l2: 0.475435
    Early stopping, best iteration is:
    [5429]	training's l2: 0.367348	valid_1's l2: 0.475325
    fold n°2
    Training until validation scores don't improve for 1000 rounds
    [1000]	training's l2: 0.469767	valid_1's l2: 0.496741
    [2000]	training's l2: 0.428546	valid_1's l2: 0.479198
    [3000]	training's l2: 0.405733	valid_1's l2: 0.475903
    [4000]	training's l2: 0.388021	valid_1's l2: 0.474891
    [5000]	training's l2: 0.372619	valid_1's l2: 0.474262
    [6000]	training's l2: 0.358826	valid_1's l2: 0.47449
    Early stopping, best iteration is:
    [5002]	training's l2: 0.372597	valid_1's l2: 0.47425
    fold n°3
    Training until validation scores don't improve for 1000 rounds
    [1000]	training's l2: 0.47361	valid_1's l2: 0.4839
    [2000]	training's l2: 0.433064	valid_1's l2: 0.462219
    [3000]	training's l2: 0.410658	valid_1's l2: 0.457989
    [4000]	training's l2: 0.392859	valid_1's l2: 0.456091
    [5000]	training's l2: 0.377706	valid_1's l2: 0.455416
    [6000]	training's l2: 0.364058	valid_1's l2: 0.455285
    Early stopping, best iteration is:
    [5815]	training's l2: 0.3665	valid_1's l2: 0.455119
    fold n°4
    Training until validation scores don't improve for 1000 rounds
    [1000]	training's l2: 0.471715	valid_1's l2: 0.496877
    [2000]	training's l2: 0.431956	valid_1's l2: 0.472828
    [3000]	training's l2: 0.409505	valid_1's l2: 0.467016
    [4000]	training's l2: 0.391659	valid_1's l2: 0.464929
    [5000]	training's l2: 0.376239	valid_1's l2: 0.464048
    [6000]	training's l2: 0.36213	valid_1's l2: 0.463628
    [7000]	training's l2: 0.349338	valid_1's l2: 0.463767
    Early stopping, best iteration is:
    [6272]	training's l2: 0.358584	valid_1's l2: 0.463542
    fold n°5
    Training until validation scores don't improve for 1000 rounds
    [1000]	training's l2: 0.466349	valid_1's l2: 0.507696
    [2000]	training's l2: 0.425606	valid_1's l2: 0.492745
    [3000]	training's l2: 0.403731	valid_1's l2: 0.488917
    [4000]	training's l2: 0.386479	valid_1's l2: 0.487113
    [5000]	training's l2: 0.371358	valid_1's l2: 0.485881
    [6000]	training's l2: 0.357821	valid_1's l2: 0.485185
    [7000]	training's l2: 0.345577	valid_1's l2: 0.484535
    [8000]	training's l2: 0.33415	valid_1's l2: 0.484483
    Early stopping, best iteration is:
    [7649]	training's l2: 0.338078	valid_1's l2: 0.484416
    CV score: 0.47052692


2. xgboost


```python
##### xgb_49
xgb_49_params = {'eta': 0.02, 
              'max_depth': 5, 
              'min_child_weight':3,
              'gamma':0,
              'subsample': 0.7, 
              'colsample_bytree': 0.35, 
              'lambda':2,
              'objective': 'reg:linear', 
              'eval_metric': 'rmse', 
              'silent': True, 
              'nthread': -1}


folds = KFold(n_splits=5, shuffle=True, random_state=2019)
oof_xgb_49 = np.zeros(len(X_train_49))
predictions_xgb_49 = np.zeros(len(X_test_49))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_49, y_train)):
    print("fold n°{}".format(fold_+1))
    trn_data = xgb.DMatrix(X_train_49[trn_idx], y_train[trn_idx])
    val_data = xgb.DMatrix(X_train_49[val_idx], y_train[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    xgb_49 = xgb.train(dtrain=trn_data, num_boost_round=3000, evals=watchlist, early_stopping_rounds=600, verbose_eval=500, params=xgb_49_params)
    oof_xgb_49[val_idx] = xgb_49.predict(xgb.DMatrix(X_train_49[val_idx]), ntree_limit=xgb_49.best_ntree_limit)
    predictions_xgb_49 += xgb_49.predict(xgb.DMatrix(X_test_49), ntree_limit=xgb_49.best_ntree_limit) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb_49, target)))
```

    fold n°1
    [19:25:31] WARNING: /Users/travis/build/dmlc/xgboost/src/objective/regression_obj.cu:170: reg:linear is now deprecated in favor of reg:squarederror.
    [19:25:31] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:480: 
    Parameters: { silent } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.


​    
    [0]	train-rmse:3.40431	valid_data-rmse:3.38307
    Multiple eval metrics have been passed: 'valid_data-rmse' will be used for early stopping.
    
    Will train until valid_data-rmse hasn't improved in 600 rounds.
    [500]	train-rmse:0.52770	valid_data-rmse:0.72110
    [1000]	train-rmse:0.43563	valid_data-rmse:0.72245
    Stopping. Best iteration:
    [690]	train-rmse:0.49010	valid_data-rmse:0.72044
    
    [19:25:44] WARNING: /Users/travis/build/dmlc/xgboost/src/objective/regression_obj.cu:170: reg:linear is now deprecated in favor of reg:squarederror.
    fold n°2
    [19:25:44] WARNING: /Users/travis/build/dmlc/xgboost/src/objective/regression_obj.cu:170: reg:linear is now deprecated in favor of reg:squarederror.
    [19:25:44] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:480: 
    Parameters: { silent } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.


​    
    [0]	train-rmse:3.39815	valid_data-rmse:3.40784
    Multiple eval metrics have been passed: 'valid_data-rmse' will be used for early stopping.
    
    Will train until valid_data-rmse hasn't improved in 600 rounds.
    [500]	train-rmse:0.52871	valid_data-rmse:0.70336
    [1000]	train-rmse:0.43793	valid_data-rmse:0.70446
    Stopping. Best iteration:
    [754]	train-rmse:0.47982	valid_data-rmse:0.70278
    
    [19:25:57] WARNING: /Users/travis/build/dmlc/xgboost/src/objective/regression_obj.cu:170: reg:linear is now deprecated in favor of reg:squarederror.
    fold n°3
    [19:25:57] WARNING: /Users/travis/build/dmlc/xgboost/src/objective/regression_obj.cu:170: reg:linear is now deprecated in favor of reg:squarederror.
    [19:25:57] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:480: 
    Parameters: { silent } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.


​    
    [0]	train-rmse:3.40183	valid_data-rmse:3.39291
    Multiple eval metrics have been passed: 'valid_data-rmse' will be used for early stopping.
    
    Will train until valid_data-rmse hasn't improved in 600 rounds.
    [500]	train-rmse:0.53169	valid_data-rmse:0.66896
    [1000]	train-rmse:0.44129	valid_data-rmse:0.67058
    Stopping. Best iteration:
    [452]	train-rmse:0.54177	valid_data-rmse:0.66871
    
    [19:26:07] WARNING: /Users/travis/build/dmlc/xgboost/src/objective/regression_obj.cu:170: reg:linear is now deprecated in favor of reg:squarederror.
    fold n°4
    [19:26:07] WARNING: /Users/travis/build/dmlc/xgboost/src/objective/regression_obj.cu:170: reg:linear is now deprecated in favor of reg:squarederror.
    [19:26:07] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:480: 
    Parameters: { silent } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.


​    
    [0]	train-rmse:3.40240	valid_data-rmse:3.39014
    Multiple eval metrics have been passed: 'valid_data-rmse' will be used for early stopping.
    
    Will train until valid_data-rmse hasn't improved in 600 rounds.
    [500]	train-rmse:0.53218	valid_data-rmse:0.67783
    [1000]	train-rmse:0.44361	valid_data-rmse:0.67978
    Stopping. Best iteration:
    [566]	train-rmse:0.51924	valid_data-rmse:0.67765
    
    [19:26:18] WARNING: /Users/travis/build/dmlc/xgboost/src/objective/regression_obj.cu:170: reg:linear is now deprecated in favor of reg:squarederror.
    fold n°5
    [19:26:19] WARNING: /Users/travis/build/dmlc/xgboost/src/objective/regression_obj.cu:170: reg:linear is now deprecated in favor of reg:squarederror.
    [19:26:19] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:480: 
    Parameters: { silent } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.


​    
    [0]	train-rmse:3.39345	valid_data-rmse:3.42619
    Multiple eval metrics have been passed: 'valid_data-rmse' will be used for early stopping.
    
    Will train until valid_data-rmse hasn't improved in 600 rounds.
    [500]	train-rmse:0.53565	valid_data-rmse:0.66150
    [1000]	train-rmse:0.44204	valid_data-rmse:0.66241
    Stopping. Best iteration:
    [747]	train-rmse:0.48554	valid_data-rmse:0.66016
    
    [19:26:32] WARNING: /Users/travis/build/dmlc/xgboost/src/objective/regression_obj.cu:170: reg:linear is now deprecated in favor of reg:squarederror.
    CV score: 0.47102840


3. GradientBoostingRegressor梯度提升决策树


```python
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
oof_gbr_49 = np.zeros(train_shape)
predictions_gbr_49 = np.zeros(len(X_test_49))
#GradientBoostingRegressor梯度提升决策树
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_49, y_train)):
    print("fold n°{}".format(fold_+1))
    tr_x = X_train_49[trn_idx]
    tr_y = y_train[trn_idx]
    gbr_49 = gbr(n_estimators=600, learning_rate=0.01,subsample=0.65,max_depth=6, min_samples_leaf=20,
            max_features=0.35,verbose=1)
    gbr_49.fit(tr_x,tr_y)
    oof_gbr_49[val_idx] = gbr_49.predict(X_train_49[val_idx])
    
    predictions_gbr_49 += gbr_49.predict(X_test_49) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_gbr_49, target)))
```

    fold n°1
          Iter       Train Loss      OOB Improve   Remaining Time 
             1           0.6529           0.0032            9.69s
             2           0.6736           0.0029            9.55s
             3           0.6522           0.0029            9.29s
             4           0.6393           0.0034            9.49s
             5           0.6454           0.0032            9.36s
             6           0.6467           0.0031            9.22s
             7           0.6650           0.0026            9.23s
             8           0.6225           0.0030            9.20s
             9           0.6350           0.0028            9.09s
            10           0.6311           0.0028            9.25s
            20           0.6074           0.0022            8.67s
            30           0.5790           0.0017            8.19s
            40           0.5443           0.0016            7.89s
            50           0.5405           0.0013            7.63s
            60           0.5141           0.0010            7.47s
            70           0.4991           0.0008            7.28s
            80           0.4791           0.0007            7.12s
            90           0.4707           0.0006            6.92s
           100           0.4632           0.0006            6.74s
           200           0.4013           0.0001            5.09s
           300           0.3924          -0.0001            3.62s
           400           0.3526          -0.0000            2.32s
           500           0.3355          -0.0000            1.12s
           600           0.3201          -0.0000            0.00s
    fold n°2
          Iter       Train Loss      OOB Improve   Remaining Time 
             1           0.6518           0.0034            8.83s
             2           0.6618           0.0033            8.42s
             3           0.6483           0.0032            8.28s
             4           0.6592           0.0029            8.27s
             5           0.6386           0.0030            8.18s
             6           0.6438           0.0031            8.16s
             7           0.6477           0.0033            8.12s
             8           0.6593           0.0029            8.15s
             9           0.6182           0.0029            8.19s
            10           0.6358           0.0028            8.32s
            20           0.5810           0.0025            7.91s
            30           0.5816           0.0020            7.74s
            40           0.5529           0.0013            7.53s
            50           0.5402           0.0011            7.38s
            60           0.5096           0.0011            7.17s
            70           0.4883           0.0010            7.03s
            80           0.4980           0.0007            6.84s
            90           0.4706           0.0006            6.71s
           100           0.4704           0.0004            6.55s
           200           0.3867           0.0001            5.01s
           300           0.3686          -0.0000            3.60s
           400           0.3363          -0.0000            2.32s
           500           0.3357          -0.0000            1.13s
           600           0.3160          -0.0000            0.00s
    fold n°3
          Iter       Train Loss      OOB Improve   Remaining Time 
             1           0.6457           0.0038            8.04s
             2           0.6687           0.0033            8.08s
             3           0.6462           0.0036            8.04s
             4           0.6587           0.0035            8.02s
             5           0.6430           0.0031            7.99s
             6           0.6540           0.0029            7.95s
             7           0.6377           0.0030            7.93s
             8           0.6414           0.0030            7.97s
             9           0.6399           0.0030            8.07s
            10           0.6375           0.0028            8.07s
            20           0.5949           0.0025            7.67s
            30           0.5854           0.0019            7.72s
            40           0.5386           0.0016            7.46s
            50           0.5156           0.0013            7.32s
            60           0.5080           0.0011            7.17s
            70           0.5021           0.0009            7.04s
            80           0.4654           0.0008            6.85s
            90           0.4712           0.0006            6.72s
           100           0.4740           0.0006            6.53s
           200           0.3924           0.0000            4.96s
           300           0.3568          -0.0000            3.58s
           400           0.3400          -0.0001            2.31s
           500           0.3283          -0.0001            1.12s
           600           0.3044          -0.0000            0.00s
    fold n°4
          Iter       Train Loss      OOB Improve   Remaining Time 
             1           0.6606           0.0032            8.27s
             2           0.6878           0.0030            8.37s
             3           0.6490           0.0031            8.37s
             4           0.6564           0.0032            8.29s
             5           0.6568           0.0027            8.27s
             6           0.6496           0.0030            8.27s
             7           0.6451           0.0029            8.22s
             8           0.6210           0.0031            8.21s
             9           0.6239           0.0028            8.35s
            10           0.6535           0.0025            8.35s
            20           0.6038           0.0022            7.92s
            30           0.6032           0.0019            7.76s
            40           0.5492           0.0018            7.55s
            50           0.5333           0.0011            7.37s
            60           0.4973           0.0010            7.24s
            70           0.4942           0.0009            7.09s
            80           0.4753           0.0008            6.92s
            90           0.4806           0.0005            6.76s
           100           0.4659           0.0005            6.58s
           200           0.4046           0.0000            4.99s
           300           0.3647          -0.0000            3.59s
           400           0.3561          -0.0000            2.32s
           500           0.3330          -0.0000            1.12s
           600           0.3152          -0.0000            0.00s
    fold n°5
          Iter       Train Loss      OOB Improve   Remaining Time 
             1           0.6721           0.0036            8.28s
             2           0.6822           0.0034            8.41s
             3           0.6634           0.0033            8.26s
             4           0.6584           0.0032            8.21s
             5           0.6574           0.0030            8.40s
             6           0.6544           0.0033            8.31s
             7           0.6533           0.0028            8.30s
             8           0.6196           0.0029            8.27s
             9           0.6530           0.0028            8.43s
            10           0.6108           0.0032            8.49s
            20           0.6107           0.0027            7.91s
            30           0.5649           0.0020            7.70s
            40           0.5555           0.0016            7.55s
            50           0.5156           0.0014            7.40s
            60           0.5144           0.0010            7.21s
            70           0.5001           0.0009            7.05s
            80           0.4908           0.0007            6.88s
            90           0.4820           0.0008            6.73s
           100           0.4617           0.0007            6.55s
           200           0.3993          -0.0000            5.01s
           300           0.3678          -0.0000            3.61s
           400           0.3399          -0.0000            2.31s
           500           0.3182          -0.0000            1.12s
           600           0.3238          -0.0000            0.00s
    CV score: 0.46724198


至此，我们得到了以上3种模型的基于49个特征的预测结果以及模型架构及参数。其中在每一种特征工程中，进行5折的交叉验证，并重复两次（Kernel Ridge Regression，核脊回归），取得每一个特征数下的模型的结果。


```python
train_stack3 = np.vstack([oof_lgb_49,oof_xgb_49,oof_gbr_49]).transpose()
test_stack3 = np.vstack([predictions_lgb_49, predictions_xgb_49,predictions_gbr_49]).transpose()
#
folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=7)
oof_stack3 = np.zeros(train_stack3.shape[0])
predictions_lr3 = np.zeros(test_stack3.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack3,target)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack3[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack3[val_idx], target.iloc[val_idx].values
        #Kernel Ridge Regression
    lr3 = kr()
    lr3.fit(trn_data, trn_y)
    
    oof_stack3[val_idx] = lr3.predict(val_data)
    predictions_lr3 += lr3.predict(test_stack3) / 10
    
mean_squared_error(target.values, oof_stack3) 

```

    fold 0
    fold 1
    fold 2
    fold 3
    fold 4
    fold 5
    fold 6
    fold 7
    fold 8
    fold 9





    0.4662728551415085



接下来我们对于383维的数据进行与上述263以及49维数据相同的操作

1. Kernel Ridge Regression 基于核的岭回归


```python
folds = KFold(n_splits=5, shuffle=True, random_state=13)
oof_kr_383 = np.zeros(train_shape)
predictions_kr_383 = np.zeros(len(X_test_383))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_383, y_train)):
    print("fold n°{}".format(fold_+1))
    tr_x = X_train_383[trn_idx]
    tr_y = y_train[trn_idx]
    #Kernel Ridge Regression 岭回归
    kr_383 = kr()
    kr_383.fit(tr_x,tr_y)
    oof_kr_383[val_idx] = kr_383.predict(X_train_383[val_idx])
    
    predictions_kr_383 += kr_383.predict(X_test_383) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_kr_383, target)))
```

    fold n°1
    fold n°2
    fold n°3
    fold n°4
    fold n°5
    CV score: 0.51412085


2. 使用普通岭回归


```python
folds = KFold(n_splits=5, shuffle=True, random_state=13)
oof_ridge_383 = np.zeros(train_shape)
predictions_ridge_383 = np.zeros(len(X_test_383))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_383, y_train)):
    print("fold n°{}".format(fold_+1))
    tr_x = X_train_383[trn_idx]
    tr_y = y_train[trn_idx]
    #使用岭回归
    ridge_383 = Ridge(alpha=1200)
    ridge_383.fit(tr_x,tr_y)
    oof_ridge_383[val_idx] = ridge_383.predict(X_train_383[val_idx])
    
    predictions_ridge_383 += ridge_383.predict(X_test_383) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_ridge_383, target)))
```

    fold n°1
    fold n°2
    fold n°3
    fold n°4
    fold n°5
    CV score: 0.48687670


3. 使用ElasticNet 弹性网络


```python
folds = KFold(n_splits=5, shuffle=True, random_state=13)
oof_en_383 = np.zeros(train_shape)
predictions_en_383 = np.zeros(len(X_test_383))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_383, y_train)):
    print("fold n°{}".format(fold_+1))
    tr_x = X_train_383[trn_idx]
    tr_y = y_train[trn_idx]
    #ElasticNet 弹性网络
    en_383 = en(alpha=1.0,l1_ratio=0.06)
    en_383.fit(tr_x,tr_y)
    oof_en_383[val_idx] = en_383.predict(X_train_383[val_idx])
    
    predictions_en_383 += en_383.predict(X_test_383) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_en_383, target)))
```

    fold n°1
    fold n°2
    fold n°3
    fold n°4
    fold n°5
    CV score: 0.53296555


4. 使用BayesianRidge 贝叶斯岭回归


```python
folds = KFold(n_splits=5, shuffle=True, random_state=13)
oof_br_383 = np.zeros(train_shape)
predictions_br_383 = np.zeros(len(X_test_383))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_383, y_train)):
    print("fold n°{}".format(fold_+1))
    tr_x = X_train_383[trn_idx]
    tr_y = y_train[trn_idx]
    #BayesianRidge 贝叶斯回归
    br_383 = br()
    br_383.fit(tr_x,tr_y)
    oof_br_383[val_idx] = br_383.predict(X_train_383[val_idx])
    
    predictions_br_383 += br_383.predict(X_test_383) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_br_383, target)))
```

    fold n°1
    fold n°2
    fold n°3
    fold n°4
    fold n°5
    CV score: 0.48717310


至此，我们得到了以上4种模型的基于383个特征的预测结果以及模型架构及参数。其中在每一种特征工程中，进行5折的交叉验证，并重复两次（LinearRegression简单的线性回归），取得每一个特征数下的模型的结果。


```python
train_stack1 = np.vstack([oof_br_383,oof_kr_383,oof_en_383,oof_ridge_383]).transpose()
test_stack1 = np.vstack([predictions_br_383, predictions_kr_383,predictions_en_383,predictions_ridge_383]).transpose()

folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=7)
oof_stack1 = np.zeros(train_stack1.shape[0])
predictions_lr1 = np.zeros(test_stack1.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack1,target)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack1[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack1[val_idx], target.iloc[val_idx].values
    # LinearRegression简单的线性回归
    lr1 = lr()
    lr1.fit(trn_data, trn_y)
    
    oof_stack1[val_idx] = lr1.predict(val_data)
    predictions_lr1 += lr1.predict(test_stack1) / 10
    
mean_squared_error(target.values, oof_stack1) 

```

    fold 0
    fold 1
    fold 2
    fold 3
    fold 4
    fold 5
    fold 6
    fold 7
    fold 8
    fold 9





    0.4878202780283125



由于49维的特征是最重要的特征，所以这里考虑增加更多的模型进行49维特征的数据的构建工作。
1. KernelRidge 核岭回归


```python
folds = KFold(n_splits=5, shuffle=True, random_state=13)
oof_kr_49 = np.zeros(train_shape)
predictions_kr_49 = np.zeros(len(X_test_49))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_49, y_train)):
    print("fold n°{}".format(fold_+1))
    tr_x = X_train_49[trn_idx]
    tr_y = y_train[trn_idx]
    kr_49 = kr()
    kr_49.fit(tr_x,tr_y)
    oof_kr_49[val_idx] = kr_49.predict(X_train_49[val_idx])
    
    predictions_kr_49 += kr_49.predict(X_test_49) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_kr_49, target)))
```

    fold n°1
    fold n°2
    fold n°3
    fold n°4
    fold n°5
    CV score: 0.50254410


2. Ridge 岭回归


```python
folds = KFold(n_splits=5, shuffle=True, random_state=13)
oof_ridge_49 = np.zeros(train_shape)
predictions_ridge_49 = np.zeros(len(X_test_49))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_49, y_train)):
    print("fold n°{}".format(fold_+1))
    tr_x = X_train_49[trn_idx]
    tr_y = y_train[trn_idx]
    ridge_49 = Ridge(alpha=6)
    ridge_49.fit(tr_x,tr_y)
    oof_ridge_49[val_idx] = ridge_49.predict(X_train_49[val_idx])
    
    predictions_ridge_49 += ridge_49.predict(X_test_49) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_ridge_49, target)))
```

    fold n°1
    fold n°2
    fold n°3
    fold n°4
    fold n°5
    CV score: 0.49451286


3. BayesianRidge 贝叶斯岭回归


```python
folds = KFold(n_splits=5, shuffle=True, random_state=13)
oof_br_49 = np.zeros(train_shape)
predictions_br_49 = np.zeros(len(X_test_49))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_49, y_train)):
    print("fold n°{}".format(fold_+1))
    tr_x = X_train_49[trn_idx]
    tr_y = y_train[trn_idx]
    br_49 = br()
    br_49.fit(tr_x,tr_y)
    oof_br_49[val_idx] = br_49.predict(X_train_49[val_idx])
    
    predictions_br_49 += br_49.predict(X_test_49) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_br_49, target)))
```

    fold n°1
    fold n°2
    fold n°3
    fold n°4
    fold n°5
    CV score: 0.49534595


4. ElasticNet 弹性网络


```python
folds = KFold(n_splits=5, shuffle=True, random_state=13)
oof_en_49 = np.zeros(train_shape)
predictions_en_49 = np.zeros(len(X_test_49))
#
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_49, y_train)):
    print("fold n°{}".format(fold_+1))
    tr_x = X_train_49[trn_idx]
    tr_y = y_train[trn_idx]
    en_49 = en(alpha=1.0,l1_ratio=0.05)
    en_49.fit(tr_x,tr_y)
    oof_en_49[val_idx] = en_49.predict(X_train_49[val_idx])
    
    predictions_en_49 += en_49.predict(X_test_49) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_en_49, target)))
```

    fold n°1
    fold n°2
    fold n°3
    fold n°4
    fold n°5
    CV score: 0.53841695


我们得到了以上4种新模型的基于49个特征的预测结果以及模型架构及参数。其中在每一种特征工程中，进行5折的交叉验证，并重复两次（LinearRegression简单的线性回归），取得每一个特征数下的模型的结果。


```python
train_stack4 = np.vstack([oof_br_49,oof_kr_49,oof_en_49,oof_ridge_49]).transpose()
test_stack4 = np.vstack([predictions_br_49, predictions_kr_49,predictions_en_49,predictions_ridge_49]).transpose()

folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=7)
oof_stack4 = np.zeros(train_stack4.shape[0])
predictions_lr4 = np.zeros(test_stack4.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack4,target)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack4[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack4[val_idx], target.iloc[val_idx].values
    #LinearRegression
    lr4 = lr()
    lr4.fit(trn_data, trn_y)
    
    oof_stack4[val_idx] = lr4.predict(val_data)
    predictions_lr4 += lr4.predict(test_stack1) / 10
    
mean_squared_error(target.values, oof_stack4) 

```

    fold 0
    fold 1
    fold 2
    fold 3
    fold 4
    fold 5
    fold 6
    fold 7
    fold 8
    fold 9





    0.49491439094008133



### 模型融合

这里对于上述四种集成学习的模型的预测结果进行加权的求和，得到最终的结果，当然这种方式是很不准确的。


```python
#和下面作对比
mean_squared_error(target.values, 0.7*(0.6*oof_stack2 + 0.4*oof_stack3)+0.3*(0.55*oof_stack1+0.45*oof_stack4))
```




    0.4527515432292745



更好的方式是将以上的4中集成学习模型再次进行集成学习的训练，这里直接使用LinearRegression简单线性回归的进行集成。


```python
train_stack5 = np.vstack([oof_stack1,oof_stack2,oof_stack3,oof_stack4]).transpose()
test_stack5 = np.vstack([predictions_lr1, predictions_lr2,predictions_lr3,predictions_lr4]).transpose()

folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=7)
oof_stack5 = np.zeros(train_stack5.shape[0])
predictions_lr5= np.zeros(test_stack5.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack5,target)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack5[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack5[val_idx], target.iloc[val_idx].values
    #LinearRegression
    lr5 = lr()
    lr5.fit(trn_data, trn_y)
    
    oof_stack5[val_idx] = lr5.predict(val_data)
    predictions_lr5 += lr5.predict(test_stack5) / 10
    
mean_squared_error(target.values, oof_stack5) 

```

    fold 0
    fold 1
    fold 2
    fold 3
    fold 4
    fold 5
    fold 6
    fold 7
    fold 8
    fold 9





    0.4480223491250565



### 结果保存

进行index的读取工作


```python
submit_example = pd.read_csv('submit_example.csv',sep=',',encoding='latin-1')

submit_example['happiness'] = predictions_lr5

submit_example.happiness.describe()
```




    count    2968.000000
    mean        3.879322
    std         0.462290
    min         1.636433
    25%         3.667859
    50%         3.954825
    75%         4.185277
    max         5.051027
    Name: happiness, dtype: float64



进行结果保存，这里我们预测出的值是1-5的连续值，但是我们的ground truth是整数值，所以为了进一步优化我们的结果，我们对于结果进行了整数解的近似，并保存到了csv文件中。


```python
submit_example.loc[submit_example['happiness']>4.96,'happiness']= 5
submit_example.loc[submit_example['happiness']<=1.04,'happiness']= 1
submit_example.loc[(submit_example['happiness']>1.96)&(submit_example['happiness']<2.04),'happiness']= 2

submit_example.to_csv("submision.csv",index=False)
submit_example.happiness.describe()
```




    count    2968.000000
    mean        3.879330
    std         0.462127
    min         1.636433
    25%         3.667859
    50%         3.954825
    75%         4.185277
    max         5.000000
    Name: happiness, dtype: float64



大家可以对于model的参数进行更进一步的调整，例如使用网格搜索的方法。这留给大家做进一步的思考喽～